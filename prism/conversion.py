"""Split the official fused Prism checkpoint into standalone native components."""
from __future__ import annotations

from collections import defaultdict
from contextlib import ExitStack
import json
from pathlib import Path
import re

import torch
from safetensors import safe_open

from . import FORMAT_VERSION, UPSTREAM_COMMIT
from .format import COMPONENTS, DTYPES, Component, StreamingWriter, TensorReader, tokenizer_metadata
from .quantization import best_group_size, decode_config, quantize


def split_fused_key(key, fused_layers):
    match = re.fullmatch(r"fusion_blocks\.(\d+)\.(video_block|audio_block|a2v_conditioner|v2a_conditioner)\.(.+)", key)
    if match:
        index, part, suffix = match.groups()
        kind, prefix = {"video_block": ("video_dit", "blocks"),
                        "audio_block": ("audio_dit", "blocks"),
                        "a2v_conditioner": ("dual_tower_bridge", "audio_to_video_conditioners"),
                        "v2a_conditioner": ("dual_tower_bridge", "video_to_audio_conditioners")}[part]
        return kind, f"{prefix}.{index}.{suffix}"
    match = re.fullmatch(r"remaining_video_blocks\.(\d+)\.(.+)", key)
    if match:
        return "video_dit", f"blocks.{int(match[1]) + fused_layers}.{match[2]}"
    for kind in ("video_dit_2", "video_dit", "audio_dit", "dual_tower_bridge"):
        if key.startswith(kind + "."):
            return kind, key[len(kind) + 1:]
    raise ValueError(f"Unrecognized official checkpoint key: {key}")


def eligible_linear(kind, key, shape):
    if not key.endswith(".weight") or len(shape) != 2 or min(shape) < 16 or best_group_size(shape[1]) is None:
        return False
    if kind in ("video_vae", "audio_vae"):
        return False  # ConvRot applies to Linear GEMMs, not codec convolution weights.
    if kind == "text_encoder":
        return bool(re.search(r"\.block\.\d+\.layer\.\d+\.(?:SelfAttention\.(?:q|k|v|o)|DenseReluDense\.(?:wi|wi_0|wi_1|wo))\.weight$", key))
    if kind == "dual_tower_bridge":
        return "conditioners." in key and ".inner." in key
    return key.startswith("blocks.") and (".self_attn." in key or ".cross_attn." in key or ".ffn." in key)


def read_config(folder, name="config.json"):
    with open(Path(folder) / name, encoding="utf-8") as source:
        return json.load(source)


def find_shards(folder):
    folder = Path(folder)
    indices = list(folder.glob("*.safetensors.index.json"))
    if len(indices) > 1:
        raise ValueError(f"Ambiguous shard indices in {folder}")
    if indices:
        weight_map = read_config(folder, indices[0].name)["weight_map"]
        names = sorted(set(weight_map.values()))
        for name in names:
            if Path(name).name != name:
                raise ValueError(f"Invalid shard filename {name}")
        paths = [folder / name for name in names]
    else:
        paths = sorted(folder.glob("*.safetensors"))
    if not paths:
        raise FileNotFoundError(f"No safetensors weights in {folder}")
    return paths


def index_sources(paths):
    result = {}
    for path in paths:
        with safe_open(str(path), framework="pt", device="cpu") as reader:
            for key in reader.keys():
                if key in result:
                    raise ValueError(f"Duplicate source key: {key}")
                result[key] = (Path(path), key, reader.get_slice(key).get_shape())
    return result


def validate_plan(kind, config, plan):
    from .loading import make_module
    with torch.device("meta"):
        module = make_module(Component(Path("."), kind, config, {}))
    expected = module.state_dict()
    unknown = set(plan) - expected.keys()
    if unknown:
        raise ValueError(f"Unexpected source tensors for {kind}: {sorted(unknown)[:8]}")
    for key, (_, _, shape) in plan.items():
        if list(expected[key].shape) != list(shape):
            raise ValueError(f"{kind}:{key} source shape {shape} does not match native config {list(expected[key].shape)}")
    missing = set(expected) - plan.keys()
    # Shared UMT5 embeddings are stored once in some official shard layouts.
    names = dict(module.named_parameters(remove_duplicate=False))
    missing = {key for key in missing if key not in names or not any(
        other in plan and names.get(other) is names[key] for other in names)}
    if missing:
        raise ValueError(f"Incomplete source {kind}: {sorted(missing)[:8]}")


def validate_converted_component(path, kind, config, metadata, plan, quantized):
    """Check an existing output's recipe and layout before explicit recovery.

    This checks native shapes, storage dtypes and quantization records. It does
    not establish equivalence to a source whose weight values have been changed.
    """
    component = Component.inspect(path, kind)
    for key, value in metadata.items():
        actual = component.metadata.get(key)
        if key in ("prism.config", "prism.scheduler_config", "prism.tokenizer_files"):
            matches = actual is not None and json.loads(actual) == json.loads(value)
        else:
            matches = actual == value
        if not matches:
            raise ValueError(f"Cannot resume {Path(path).name}: metadata differs for {key}")
    from .loading import make_module
    with torch.device("meta"):
        expected = make_module(Component(Path(path), kind, config, metadata)).state_dict()
    quantized = set(quantized)
    keys = set(plan)
    for key in quantized:
        prefix = key[:-len(".weight")]
        keys.update((prefix + ".weight_scale", prefix + ".comfy_quant"))
    with TensorReader(path) as reader:
        header = reader._header
        if set(header) != keys:
            raise ValueError(f"Cannot resume {Path(path).name}: tensor keys differ")
        for key, (_, _, shape) in plan.items():
            dtype = "I8" if key in quantized else "BF16" if expected[key].is_floating_point() else DTYPES[expected[key].dtype]
            if header[key]["shape"] != list(shape) or header[key]["dtype"] != dtype:
                raise ValueError(f"Cannot resume {Path(path).name}: tensor layout differs for {key}")
        for key in quantized:
            prefix = key[:-len(".weight")]
            marker = prefix + ".comfy_quant"
            quant_config = decode_config(reader.get_tensor(marker))
            if quant_config["convrot_groupsize"] != best_group_size(plan[key][2][1]):
                raise ValueError(f"Cannot resume {Path(path).name}: ConvRot group differs for {key}")
            scale = reader.get_tensor(prefix + ".weight_scale")
            if (scale.dtype != torch.float32 or list(scale.shape) != [plan[key][2][0], 1]
                    or not torch.isfinite(scale).all() or not (scale > 0).all()):
                raise ValueError(f"Cannot resume {Path(path).name}: invalid ConvRot scale for {key}")
    return component


def convert_bundle(base, preview, output, *, variant="alpha", precision="int8_convrot",
                   mseclip=False, device="cpu", overwrite=False, dry_run=False, components=None, resume=False):
    base, output = Path(base), Path(output)
    if variant not in ("alpha", "beta") or precision not in ("int8_convrot", "bf16"):
        raise ValueError("Invalid variant or precision")
    selected = set(components or COMPONENTS)
    if selected - set(COMPONENTS):
        raise ValueError(f"Unknown components: {sorted(selected - set(COMPONENTS))}")
    configs = {kind: read_config(base / kind) for kind in COMPONENTS if (base / kind / "config.json").exists()}
    missing = set(COMPONENTS) - {"video_dit_2"} - configs.keys()
    if missing:
        raise ValueError(f"Missing official component configs: {sorted(missing)}")
    if any(configs[k].get("has_image_input", False) for k in ("video_dit", "audio_dit")):
        raise ValueError("This Prism pipeline does not implement external CLIP image embedding checkpoints")
    if configs["audio_dit"].get("vae_type") != "dac":
        raise ValueError("Only the officially released DAC preview architecture is supported")
    fused_layers = min(configs["video_dit"]["num_layers"], configs["audio_dit"]["num_layers"])
    sources = defaultdict(dict)
    # Official preview is a complete fused transformer. No base DiT weights needed.
    if selected.intersection(("video_dit", "video_dit_2", "audio_dit", "dual_tower_bridge")):
        if preview is None:
            raise ValueError("Transformer conversion requires --preview")
        for source_key, source in index_sources([preview]).items():
            kind, key = split_fused_key(source_key, fused_layers)
            if key in sources[kind]:
                raise ValueError(f"Ambiguous fused mapping {kind}:{key}")
            sources[kind][key] = source
    for kind in ("video_dit", "audio_dit", "dual_tower_bridge"):
        if kind in selected and not sources[kind]:
            raise ValueError(f"Incomplete preview: missing {kind}")
    for kind in ("text_encoder", "video_vae", "audio_vae"):
        if kind in selected:
            sources[kind] = index_sources(find_shards(base / kind))
    model_index = read_config(base, "model_index.json") if (base / "model_index.json").exists() else {}
    scheduler_path = base / "scheduler/scheduler_config.json"
    scheduler = read_config(scheduler_path.parent, scheduler_path.name) if scheduler_path.exists() else {"shift": 5.0}
    tokenizer = tokenizer_metadata(base / "tokenizer")
    bundle_id = f"prism-preview-{variant}:{UPSTREAM_COMMIT}"
    report = {"bundle_id": bundle_id, "precision": precision, "mseclip": mseclip, "components": {}}
    report_path = output / f"prism_{variant}_conversion.json"
    if report_path.exists():
        previous = json.loads(report_path.read_text(encoding="utf-8"))
        if any(previous.get(key) != report[key] for key in ("bundle_id", "precision", "mseclip")):
            raise ValueError("Output manifest contains a different conversion recipe; choose a new output directory")
        report["components"].update(previous["components"])
    def save_report():
        output.mkdir(parents=True, exist_ok=True)
        report["complete"] = set(COMPONENTS).issubset(report["components"]) and all(
            (output / entry["file"]).is_file() for entry in report["components"].values())
        temporary = report_path.with_suffix(".json.part")
        temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
        temporary.replace(report_path)
    for kind in COMPONENTS:
        if kind not in selected or not sources[kind]:
            continue
        if kind not in configs:
            raise ValueError(f"Checkpoint contains {kind} but no matching config")
        plan = sources[kind]
        validate_plan(kind, configs[kind], plan)
        quantized = [key for key, (_, _, shape) in plan.items() if precision == "int8_convrot" and eligible_linear(kind, key, shape)]
        filename = f"prism_{variant}_{kind}_{'int8_convrot' if quantized else 'bf16'}.safetensors"
        report["components"][kind] = {"file": filename, "source_tensors": len(plan), "quantized_linears": len(quantized)}
        print(f"{kind}: {len(plan)} tensors, {len(quantized)} ConvRot linears -> {filename}", flush=True)
        if dry_run:
            continue
        metadata = {"prism.format_version": FORMAT_VERSION, "prism.component": kind,
                    "prism.config": json.dumps(configs[kind]), "prism.bundle_id": bundle_id,
                    "prism.upstream_commit": UPSTREAM_COMMIT, "prism.variant": variant,
                    "prism.precision": "int8_convrot" if quantized else "bf16",
                    "prism.quant_recipe": "block-linear-regular-hadamard-mseclip" if mseclip else "block-linear-regular-hadamard-absmax"}
        if kind == "text_encoder":
            metadata["prism.tokenizer_files"] = tokenizer
        if kind == "dual_tower_bridge":
            metadata["prism.scheduler_config"] = json.dumps(scheduler)
            metadata["prism.boundary_ratio"] = str(model_index.get("boundary_ratio", 0.9))
        quantized_set = set(quantized)
        if resume and not overwrite and (output / filename).exists():
            validate_converted_component(output / filename, kind, configs[kind], metadata, plan, quantized)
            save_report()
            print(f"  {kind}: reused validated component and recovered its manifest entry", flush=True)
            continue
        with StreamingWriter(output / filename, metadata, overwrite=overwrite) as writer, ExitStack() as reader_stack:
            # Read one owned tensor, avoiding a 60+GiB private PyTorch mmap on Windows.
            readers = {}
            for index, (key, (path, original_key, _)) in enumerate(sorted(plan.items())):
                if path not in readers:
                    readers[path] = reader_stack.enter_context(TensorReader(path, copy=True))
                tensor = readers[path].get_tensor(original_key)
                if tensor.is_floating_point() and not torch.isfinite(tensor).all():
                    raise ValueError(f"Non-finite source weights: {original_key}")
                if key in quantized_set:
                    q, scale, config = quantize(tensor, mseclip=mseclip, device=device)
                    prefix = key[:-len(".weight")]
                    writer.add(key, q)
                    writer.add(prefix + ".weight_scale", scale)
                    writer.add(prefix + ".comfy_quant", config)
                else:
                    stored = tensor.to(torch.bfloat16) if tensor.is_floating_point() else tensor
                    if stored is not tensor and not torch.isfinite(stored).all():
                        raise ValueError(f"Non-finite source weights after BF16 conversion: {original_key}")
                    writer.add(key, stored)
                if (index + 1) % 50 == 0:
                    print(f"  {kind}: {index + 1}/{len(plan)}", flush=True)
            writer.finish()
        # Commit each successfully installed component so a later interruption
        # can resume without overwriting earlier multi-gigabyte outputs.
        save_report()
    if not dry_run:
        save_report()
    return report
