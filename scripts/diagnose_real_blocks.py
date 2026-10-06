"""Capture one real INT8 positive forward, then compare isolated native DiTBlocks.

No command runs a BF16 Transformer, denoising update, VAE decode, or video sampler.
`check` is CPU-only and verifies structure/serialization, not model quality.
`capture` and `compare` explicitly require a free CUDA GPU; run them separately.
Complex tensors are stored with view_as_real in safetensors plus JSON type metadata.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import inspect
import json
import math
from pathlib import Path
import struct
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from safetensors.torch import save_file

from prism.format import Component, COMPONENTS, TensorReader, load_tokenizer
from prism.loading import load_component, set_tensor
from prism.quantization import ConvRotLinear, decode_config

BLOCKS = (0, 19, 39)
VARIANTS = ("bf16", "int8", "self_qk_bf16", "all_qk_bf16", "cross_attn_bf16")
PROMPT = (
    "Medium shot on a fishing boat in daylight. A smiling man wearing a gray cap, "
    "blue sunglasses and a gray shirt holds a large grouper. The boat gently rocks, "
    "the fish moves slightly in his hands, and sunlight glints on the ocean. "
    "The camera stays steady. <sfx>Gentle waves and water lapping around the boat.</sfx>"
)
CORE_FILES = (
    "prism/runtime.py", "prism/offload.py", "prism/loading.py", "prism/quantization.py",
    "prism/native/models/modules/mova.py", "prism/native/models/modules/wan_video_dit.py",
    "prism/native/models/modules/wan_audio_dit.py", "prism/native/models/modules/interactionv2.py",
    "prism/native/diffusion/pipelines/mova_pipeline.py",
    "prism/native/diffusion/schedulers/flow_match_pair.py",
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def provenance(component):
    # Hash the small header, not a complete multi-GB component, to keep this bounded.
    with component.path.open("rb") as stream:
        header_size = struct.unpack("<Q", stream.read(8))[0]
        header_hash = hashlib.sha256(stream.read(header_size)).hexdigest()
    stat = component.path.stat()
    return {"path": str(component.path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
            "header_sha256": header_hash, "bundle_id": component.metadata["prism.bundle_id"],
            "precision": component.metadata["prism.precision"]}


def code_hashes():
    return {name: sha256(ROOT / name) for name in CORE_FILES}


def pack(path, value):
    """Serialize tensors and exact Python container types without pickle."""
    path = Path(path)
    if path.exists() or path.with_suffix(".json").exists():
        raise FileExistsError(f"Capture already exists: {path}")
    tensors = {}

    def encode(item):
        if isinstance(item, torch.Tensor):
            name = f"tensor_{len(tensors):04d}"
            item = item.detach().cpu().contiguous()
            is_complex = item.is_complex()
            tensors[name] = (torch.view_as_real(item) if is_complex else item).clone().contiguous()
            return {"type": "tensor", "key": name, "complex": is_complex,
                    "dtype": str(item.dtype), "shape": list(item.shape)}
        if item is None or type(item) in (str, bool, int, float):
            if isinstance(item, float) and not math.isfinite(item):
                raise ValueError("Non-finite scalar metadata")
            return {"type": "scalar", "value": item}
        if type(item) in (tuple, list):
            return {"type": type(item).__name__, "items": [encode(part) for part in item]}
        if type(item) is dict and all(type(key) is str for key in item):
            return {"type": "dict", "items": {key: encode(part) for key, part in item.items()}}
        raise TypeError(f"Unsupported capture metadata: {type(item)}")

    schema = encode(value)
    if not tensors:
        raise ValueError("A tensor capture must contain a tensor")
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(path), metadata={"prism.diagnostic": "real_block_inputs_v1"})
    write_json(path.with_suffix(".json"), {"format": "real_block_inputs_v1", "schema": schema,
                                           "safetensors_sha256": sha256(path)})


def unpack(path, device="cpu"):
    path = Path(path)
    info = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    if info["format"] != "real_block_inputs_v1" or sha256(path) != info["safetensors_sha256"]:
        raise ValueError(f"Capture format/hash mismatch: {path}")
    with TensorReader(path, copy=True) as reader:
        tensors = {name: reader.get_tensor(name) for name in reader.keys()}
    used = set()

    def decode(item):
        kind = item["type"]
        if kind == "tensor":
            used.add(item["key"])
            tensor = tensors[item["key"]]
            if item["complex"]:
                tensor = torch.view_as_complex(tensor.contiguous())
            if str(tensor.dtype) != item["dtype"] or list(tensor.shape) != item["shape"]:
                raise ValueError("Captured tensor shape/dtype mismatch")
            return tensor.to(device)
        if kind == "scalar":
            return item["value"]
        if kind in ("tuple", "list"):
            parts = [decode(part) for part in item["items"]]
            return tuple(parts) if kind == "tuple" else parts
        if kind == "dict":
            return {key: decode(part) for key, part in item["items"].items()}
        raise ValueError(f"Unsupported schema node: {kind}")

    result = decode(info["schema"])
    if used != tensors.keys():
        raise ValueError("Unused/missing tensor in capture")
    return result


def component_pair(args):
    quant = Component.inspect(Path(args.models) / "prism_alpha_video_dit_int8_convrot.safetensors", "video_dit")
    baseline = Component.inspect(Path(args.baseline) / "prism_alpha_video_dit_bf16.safetensors", "video_dit")
    if quant.metadata["prism.precision"] != "int8_convrot" or baseline.metadata["prism.precision"] != "bf16":
        raise ValueError("Expected original INT8 ConvRot and unquantized BF16 video components")
    if quant.metadata["prism.bundle_id"] != baseline.metadata["prism.bundle_id"] or quant.config != baseline.config:
        raise ValueError("INT8/BF16 components must have identical bundle/config provenance")
    if quant.config["num_layers"] <= max(BLOCKS):
        raise ValueError("This diagnostic expects the real high expert with block 39")
    return quant, baseline


def empty_block(config):
    from prism.native.models.modules.wan_video_dit import DiTBlock
    with torch.device("meta"):
        return DiTBlock(has_image_input=config["has_image_input"], dim=config["dim"],
                        num_heads=config["num_heads"], ffn_dim=config["ffn_dim"], eps=config["eps"])


def bf16_layer(variant, name):
    if variant == "bf16":
        return True
    if variant == "self_qk_bf16":
        return name in ("self_attn.q", "self_attn.k")
    if variant == "all_qk_bf16":
        return name in ("self_attn.q", "self_attn.k", "cross_attn.q", "cross_attn.k")
    return variant == "cross_attn_bf16" and name.startswith("cross_attn.")


def load_one_block(quant, baseline, index, variant, backend="portable"):
    """Construct ONE DiTBlock, read ONLY its tensors; never construct a BF16 WanModel."""
    if variant not in VARIANTS or index not in BLOCKS:
        raise ValueError("Unsupported comparison block/variant")
    block = empty_block(quant.config)
    expected = set(block.state_dict())
    prefix = f"blocks.{index}."
    source = baseline if variant == "bf16" else quant
    with TensorReader(source.path) as reader, TensorReader(baseline.path) as bf:
        keys = {key[len(prefix):] for key in reader.keys() if key.startswith(prefix)}
        logical = {key for key in keys if not key.endswith((".weight_scale", ".comfy_quant"))}
        if logical != expected:
            raise ValueError(f"Block {index}: source native keys differ from the actual DiTBlock")
        consumed = set()
        for marker in sorted(key for key in keys if key.endswith(".comfy_quant")):
            name = marker[:-len(".comfy_quant")]
            layer = block.get_submodule(name)
            if not isinstance(layer, torch.nn.Linear):
                raise ValueError(f"Expected Linear: {name}")
            cfg = decode_config(reader.get_tensor(prefix + marker))
            weight, scale, bias = name + ".weight", name + ".weight_scale", name + ".bias"
            if scale not in keys:
                raise ValueError(f"Missing ConvRot scale: {name}")
            if bf16_layer(variant, name):
                set_tensor(block, weight, bf.get_tensor(prefix + weight), torch.bfloat16)
                if bias in keys:
                    set_tensor(block, bias, bf.get_tensor(prefix + bias), torch.bfloat16)
            else:
                value = reader.get_tensor(prefix + bias).to(torch.bfloat16) if bias in keys else None
                replacement = ConvRotLinear(reader.get_tensor(prefix + weight), reader.get_tensor(prefix + scale),
                                             cfg, value, backend)
                if (replacement.in_features, replacement.out_features) != (layer.in_features, layer.out_features):
                    raise ValueError(f"Quantized shape mismatch: {name}")
                if (layer.bias is None) != (value is None):
                    raise ValueError(f"Quantized bias mismatch: {name}")
                parent, _, attr = name.rpartition(".")
                (block.get_submodule(parent) if parent else block)._modules[attr] = replacement
            consumed.update((marker, weight, scale))
            if bias in keys:
                consumed.add(bias)
        for key in sorted(keys - consumed):
            value = reader.get_tensor(prefix + key)
            if not value.is_floating_point():
                raise ValueError(f"Unmarked integer block weight: {key}")
            set_tensor(block, key, value, torch.bfloat16)
    if any(value.is_meta for value in list(block.parameters()) + list(block.buffers())):
        raise ValueError(f"Block {index}: incomplete standalone loading")
    return block.eval().requires_grad_(False)


def block_metrics(actual, reference, input_x=None):
    actual, reference = actual.detach().cpu(), reference.detach().cpu()
    if actual.shape != reference.shape or not torch.isfinite(actual).all() or not torch.isfinite(reference).all():
        raise ValueError("Block output shape mismatch/non-finite output")
    a, b = actual.reshape(-1), reference.reshape(-1)
    inp = input_x.detach().cpu().reshape(-1) if input_x is not None else None
    count = a.numel()
    error_sq = reference_sq = absolute = dot = actual_sq = residual_sq = 0.0
    max_abs = 0.0
    for start in range(0, count, 1_000_000):
        end = start + 1_000_000
        av, bv = a[start:end].float(), b[start:end].float()
        diff = av - bv
        error_sq += diff.double().square().sum().item()
        reference_sq += bv.double().square().sum().item()
        actual_sq += av.double().square().sum().item()
        dot += (av.double() * bv.double()).sum().item()
        absolute += diff.double().abs().sum().item()
        max_abs = max(max_abs, diff.abs().max().item())
        if inp is not None:
            residual_sq += (bv - inp[start:end].float()).double().square().sum().item()
    result = {"relative_l2_pct": 100 * math.sqrt(error_sq / max(reference_sq, 1e-30)),
              "rmse": math.sqrt(error_sq / count), "mean_abs_error": absolute / count,
              "max_abs_error": max_abs,
              "cosine_similarity": dot / max(math.sqrt(reference_sq * actual_sq), 1e-30),
              "exact_equal": torch.equal(actual, reference)}
    if inp is not None:
        result["error_relative_to_bf16_block_update_pct"] = 100 * math.sqrt(error_sq / max(residual_sq, 1e-30))
        result["bf16_block_update_rms"] = math.sqrt(residual_sq / count)
    # Native 320x192/49 video patch grid: [13,12,20], f-major sequence order.
    if actual.ndim == 3 and actual.shape[1] == 13 * 12 * 20:
        result["per_latent_frame_relative_l2_pct"] = [
            block_metrics(actual[:, frame * 240:(frame + 1) * 240],
                          reference[:, frame * 240:(frame + 1) * 240])["relative_l2_pct"]
            for frame in range(13)]
    return result


def capture(args):
    from PIL import Image
    from prism.offload import FrozenOffloadModule, ManagedTransformer
    from prism.runtime import check_bundle, crop_reference, dense_attention
    from prism.settings import GENERATION_DEFAULTS, validate_generation, configure_sparse
    from prism.native.models.modules.mova import MOVABridge
    from prism.native.diffusion.pipelines.mova_pipeline import MOVAPipeline
    from prism.native.diffusion.schedulers.flow_match_pair import FlowMatchPairScheduler

    if not torch.cuda.is_available():
        raise RuntimeError("capture requires a free CUDA GPU; check is CPU-only")
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("Real capture requires CUDA")
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError("Use a new output directory; existing captures are never overwritten")
    output.mkdir(parents=True)
    quant, baseline = component_pair(args)
    parts = {}
    for kind in COMPONENTS:
        precision = "bf16" if kind.endswith("vae") else "int8_convrot"
        parts[kind] = Component.inspect(Path(args.models) / f"prism_alpha_{kind}_{precision}.safetensors", kind)
        if parts[kind].metadata["prism.precision"] != precision:
            raise ValueError(f"Capture refuses a BF16 Transformer/UMT5: {kind}")
    check_bundle(parts)
    settings = validate_generation({**GENERATION_DEFAULTS, "mode": "i2va", "prompt": args.prompt,
        "audio_prompt": "", "width": 320, "height": 192, "num_frames": 49, "fps": 24.,
        "steps": 50, "seed": 42, "cfg": 5., "visual_shift": 9., "audio_shift": 7.,
        "offload": "block", "attention": "sdpa", "int8_backend": args.int8_backend, "vae_tiling": False})
    image_path = Path(args.image).resolve()
    image = crop_reference(Image.open(image_path), settings["height"], settings["width"])
    report = {"status": "preparing", "scope": "exactly one INT8 positive forward at first timestep",
              "blocks": list(BLOCKS), "settings": settings, "code_hashes": code_hashes(),
              "script_sha256": sha256(__file__), "reference": str(image_path),
              "reference_sha256": sha256(image_path), "int8_component": provenance(quant),
              "components": {kind: provenance(part) for kind, part in parts.items()},
              "bf16_component_for_later_single_blocks_only": provenance(baseline),
              "loaded_bf16_transformer_components": 0, "transformer_calls": 0,
              "scheduler_update_calls": 0, "vae_decode_calls": 0, "captured": {},
              "negative_forward_calls": 0, "video_produced": False}
    write_json(output / "capture.json", report)
    modules, handles, managed = {}, [], None
    start = time.monotonic()

    class CaptureComplete(Exception):
        pass

    def forbidden_update(*unused, **unused_kwargs):
        report["scheduler_update_calls"] += 1
        raise RuntimeError("Denoising update forbidden by this single-forward diagnostic")

    def forbidden_decode(*unused, **unused_kwargs):
        report["vae_decode_calls"] += 1
        raise RuntimeError("VAE decode forbidden by this single-forward diagnostic")

    class FirstPositive(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def to(self, *positional, **keywords):
            self.inner.to(*positional, **keywords)
            return self

        def forward(self, *positional, **keywords):
            report["transformer_calls"] += 1
            if report["transformer_calls"] != 1 or positional or keywords.get("use_video_dit_2"):
                raise RuntimeError("Only the first positive high-expert forward is permitted")
            if keywords["visual_latents"].shape != (1, 36, 13, 24, 40):
                raise ValueError("Unexpected real video/reference latent shape")
            if keywords["timestep"].item() != 1000. or keywords["audio_timestep"].item() != 1000.:
                raise ValueError("Expected actual first pair of the 50-step native schedule")
            pack(output / "first-transformer-input.safetensors", keywords)
            result = self.inner(**keywords)
            pack(output / "first-transformer-output.safetensors", result)
            raise CaptureComplete()

    try:
        # The low expert cannot execute at this timestep; do not even load it.
        for kind in COMPONENTS:
            if kind == "video_dit_2":
                continue
            print(f"capture: loading {kind} ({parts[kind].metadata['prism.precision']})", flush=True)
            module = load_component(parts[kind], dtype=torch.float32 if kind == "audio_vae" else torch.bfloat16,
                                    backend=args.int8_backend)
            if kind in ("video_vae", "audio_vae", "text_encoder"):
                module = FrozenOffloadModule(module)
            modules[kind] = module
        boundary = float(parts["dual_tower_bridge"].metadata["prism.boundary_ratio"])
        bridge = MOVABridge(modules["video_dit"], None, modules["audio_dit"], modules["dual_tower_bridge"],
                            boundary_ratio=boundary)
        configure_sparse(bridge, {})
        native_blocks = [fused.video_block for fused in bridge.fusion_blocks] + list(bridge.remaining_video_blocks)
        for index in BLOCKS:
            block = native_blocks[index]
            signature = inspect.signature(block.forward)

            def before(module, positional, keywords, index=index, signature=signature):
                entry = report["captured"].setdefault(str(index), {"input_calls": 0, "output_calls": 0})
                entry["input_calls"] += 1
                if entry["input_calls"] != 1:
                    raise RuntimeError("A selected block was executed more than once")
                bound = signature.bind(*positional, **keywords)
                bound.apply_defaults()
                pack(output / f"block-{index:03d}-input.safetensors", dict(bound.arguments))

            def after(module, positional, result, index=index):
                if not isinstance(result, torch.Tensor):
                    raise TypeError("Expected native DiTBlock tensor output")
                report["captured"][str(index)]["output_calls"] += 1
                pack(output / f"block-{index:03d}-output.safetensors", result)

            handles.append(block.register_forward_pre_hook(before, with_kwargs=True))
            handles.append(block.register_forward_hook(after))
        managed = ManagedTransformer(bridge, block_offload=True)
        scheduler = FlowMatchPairScheduler.from_config(json.loads(parts["dual_tower_bridge"].metadata["prism.scheduler_config"]))
        scheduler.step_from_to = forbidden_update
        modules["video_vae"].module.decode = forbidden_decode
        modules["audio_vae"].module.decode = forbidden_decode
        pipe = MOVAPipeline(FirstPositive(managed), modules["video_vae"], modules["audio_vae"], modules["text_encoder"],
                            load_tokenizer(parts["text_encoder"]), scheduler, boundary_ratio=boundary, device=device)
        pipe.enable_cpu_offload(device)
        with torch.inference_mode(), dense_attention("sdpa"):
            try:
                pipe(prompt=settings["prompt"], image=image, audio_prompt=None,
                     negative_prompt=settings["negative_prompt"], seed=42, height=192, width=320,
                     num_frames=49, video_fps=24., num_inference_steps=50, visual_shift=9., audio_shift=7.,
                     cfg_scale=5., enable_vae_tiling=False)
            except CaptureComplete:
                pass
            else:
                raise RuntimeError("Native pipeline unexpectedly returned; no completed sampler is permitted")
        if set(report["captured"]) != {str(index) for index in BLOCKS}:
            raise RuntimeError("Missing selected real block activations")
        if any(entry != {"input_calls": 1, "output_calls": 1} for entry in report["captured"].values()):
            raise RuntimeError("Incomplete real block capture")
        if report["transformer_calls"] != 1 or report["scheduler_update_calls"] or report["vae_decode_calls"]:
            raise RuntimeError("Forbidden diagnostic execution scope")
        report["actual_schedule"] = {"pair_count": int(scheduler.get_pairs().shape[0]),
                                     "pairs": scheduler.get_pairs().tolist(),
                                     "scheduler_config": dict(scheduler.config)}
        if report["actual_schedule"]["pair_count"] != 50:
            raise RuntimeError("Expected the actual native 50-step paired schedule")
        report["status"] = "complete"
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        for handle in handles:
            handle.remove()
        if managed is not None:
            managed.close()
        for module in modules.values():
            module.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()
        report["seconds"] = time.monotonic() - start
        write_json(output / "capture.json", report)
    print(json.dumps({"capture": str(output), "status": report["status"], "transformer_calls": 1,
                      "latent_updates": 0, "vae_decode_calls": 0}), flush=True)


def compare(args):
    from prism.runtime import dense_attention
    if not torch.cuda.is_available() or torch.device(args.device).type != "cuda":
        raise RuntimeError("Real isolated-block comparison requires a free CUDA GPU")
    folder = Path(args.capture).resolve()
    captured = json.loads((folder / "capture.json").read_text(encoding="utf-8"))
    quant, baseline = component_pair(args)
    if captured["status"] != "complete" or captured["transformer_calls"] != 1:
        raise ValueError("Only a completed first-positive capture may be compared")
    if captured["code_hashes"] != code_hashes():
        raise ValueError("Native/loading/quantization code changed since capture; make a new capture")
    if args.int8_backend != captured["settings"]["int8_backend"]:
        raise ValueError("Replay backend must match the actual INT8 capture backend")
    if captured["int8_component"] != provenance(quant) or captured["bf16_component_for_later_single_blocks_only"] != provenance(baseline):
        raise ValueError("Checkpoint provenance changed since capture")
    variants = args.variants.split(",")
    if len(set(variants)) != len(variants) or set(variants) - set(VARIANTS) or not {"bf16", "int8"} <= set(variants):
        raise ValueError("Comparison must include bf16 and int8, with unique supported variants")
    path = Path(args.report).resolve() if args.report else folder / "single-block-comparison.json"
    if path.exists():
        raise FileExistsError("Comparison report already exists; choose a new --report path")
    path.parent.mkdir(parents=True, exist_ok=True)
    report = {"status": "running", "capture": str(folder), "scope": "one DiTBlock at a time with identical real INT8-captured input",
              "loaded_complete_bf16_transformers": 0, "denoising_updates": 0, "video_produced": False,
              "variants": variants, "blocks": {}, "script_sha256": sha256(__file__),
              "note": "First high-noise positive step only; this does not establish full-video quality or late-step/low-expert accuracy."}
    start = time.monotonic()
    try:
        for index in BLOCKS:
            cpu_input = unpack(folder / f"block-{index:03d}-input.safetensors")
            observed = unpack(folder / f"block-{index:03d}-output.safetensors")
            arguments = unpack(folder / f"block-{index:03d}-input.safetensors", device=args.device)
            entry = report["blocks"].setdefault(str(index), {})
            reference = None
            for variant in ["bf16"] + [name for name in variants if name != "bf16"]:
                print(f"compare: block {index}, {variant} (ONE block only)", flush=True)
                block = load_one_block(quant, baseline, index, variant, args.int8_backend).to(args.device)
                try:
                    with torch.inference_mode(), dense_attention("sdpa"):
                        result = block(**arguments).detach().cpu()
                    if variant == "bf16":
                        reference = result
                    entry[variant] = block_metrics(result, reference, cpu_input["x"])
                    entry[variant]["quantized_linears"] = sum(isinstance(layer, ConvRotLinear) for layer in block.modules())
                    if variant == "int8":
                        entry[variant]["replay_vs_original_int8_capture"] = block_metrics(result, observed)
                    write_json(path, report)
                finally:
                    del block
                    gc.collect()
                    torch.cuda.empty_cache()
            del arguments, cpu_input, observed, reference, result
            gc.collect()
            torch.cuda.empty_cache()
        report["status"] = "complete"
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report["seconds"] = time.monotonic() - start
        write_json(path, report)
    print(json.dumps({"comparison": str(path), "status": report["status"]}), flush=True)


def check(args):
    """Header/meta and small CPU serialization/loader verification; no quality inference."""
    initialized_before = torch.cuda.is_initialized()
    if initialized_before:
        raise RuntimeError("CPU structure check must start without CUDA initialization")
    quant, baseline = component_pair(args)
    expected = empty_block(quant.config).state_dict()
    rows = {}
    for component in (quant, baseline):
        with TensorReader(component.path) as reader:
            per_block = {}
            for index in BLOCKS:
                prefix = f"blocks.{index}."
                keys = {key[len(prefix):] for key in reader.keys() if key.startswith(prefix)}
                logical = {key for key in keys if not key.endswith((".weight_scale", ".comfy_quant"))}
                if logical != expected.keys():
                    raise ValueError("Real component block keys do not match native block constructor")
                quantized = 0
                for key in logical:
                    header = reader._header[prefix + key]
                    if list(expected[key].shape) != header["shape"]:
                        raise ValueError(f"Real component header shape mismatch: {key}")
                    if header["dtype"] == "I8":
                        quantized += 1
                        name = key[:-len(".weight")]
                        cfg = decode_config(reader.get_tensor(prefix + name + ".comfy_quant"))
                        scale = reader._header[prefix + name + ".weight_scale"]
                        if cfg.get("convrot") is not True or header["shape"][1] % cfg["convrot_groupsize"] or scale["dtype"] != "F32" or scale["shape"] != [header["shape"][0], 1]:
                            raise ValueError("Invalid real ConvRot header/config")
                    elif header["dtype"] != "BF16":
                        raise ValueError("Expected actual BF16 passthrough tensor")
                per_block[str(index)] = {"native_tensors": len(logical), "quantized_linears": quantized}
            rows[component.metadata["prism.precision"]] = per_block
    cpu_loads = {}
    expected_quantized = {"bf16": 0, "int8": 10, "self_qk_bf16": 8,
                          "all_qk_bf16": 6, "cross_attn_bf16": 6}
    for index in BLOCKS:
        cpu_loads[str(index)] = {}
        for variant in VARIANTS:
            # mmap the real single-block weights, inspect residency; do not forward.
            block = load_one_block(quant, baseline, index, variant)
            count = sum(isinstance(layer, ConvRotLinear) for layer in block.modules())
            assert count == expected_quantized[variant]
            assert all(value.device.type == "cpu" for value in list(block.parameters()) + list(block.buffers()))
            cpu_loads[str(index)][variant] = {"quantized_linears": count, "forward_calls": 0}
            del block
            gc.collect()
    with tempfile.TemporaryDirectory(prefix="prism_real_block_cpu_check_") as tmp:
        path = Path(tmp) / "roundtrip.safetensors"
        sample = {"x": torch.arange(6, dtype=torch.bfloat16).reshape(2, 3),
                  "freqs": torch.polar(torch.ones(3, dtype=torch.float64), torch.arange(3, dtype=torch.float64)),
                  "metadata": (None, True, 0.9, {"grid_size": [13, 12, 20]})}
        pack(path, sample)
        restored = unpack(path)
        assert torch.equal(sample["x"], restored["x"]) and torch.equal(sample["freqs"], restored["freqs"])
        assert type(restored["metadata"]) is tuple and sample["metadata"] == restored["metadata"]
    known = block_metrics(torch.tensor([2., 4.]), torch.tensor([1., 2.]), torch.zeros(2))
    assert math.isclose(known["relative_l2_pct"], 100.) and math.isclose(known["cosine_similarity"], 1.)
    assert math.isclose(known["error_relative_to_bf16_block_update_pct"], 100.)
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU structure check unexpectedly initialized CUDA")
    print(json.dumps({"status": "cpu_structure_verified", "quality_evidence": False,
                      "cuda_initialized": False, "real_block_headers": rows,
                      "real_single_block_cpu_loads": cpu_loads,
                      "known_error_metric_checked": True,
                      "complex_and_metadata_safetensors_roundtrip": True,
                      "mixed_bf16_layers": {name: [layer for layer in ("self_attn.q", "self_attn.k", "self_attn.v", "self_attn.o", "cross_attn.q", "cross_attn.k", "cross_attn.v", "cross_attn.o", "ffn.0", "ffn.2") if bf16_layer(name, layer)] for name in VARIANTS}}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "capture", "compare"))
    parser.add_argument("--models", default="models/standalone")
    parser.add_argument("--baseline", default="models/baseline_bf16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--image", default="examples/prism_official_case5.png")
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--output", help="New capture directory, required by capture")
    parser.add_argument("--capture", help="Completed capture directory, required by compare")
    parser.add_argument("--report", help="Optional new comparison JSON path; reuses the same captured activation files")
    parser.add_argument("--variants", default="bf16,int8,self_qk_bf16,cross_attn_bf16",
                        help="Comma-separated isolated block variants; all_qk_bf16 is also available")
    parser.add_argument("--int8-backend", choices=("portable", "kitchen"), default="portable")
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.command == "capture" and not args.output:
        parser.error("capture requires --output")
    if args.command == "compare" and not args.capture:
        parser.error("compare requires --capture")
    {"check": check, "capture": capture, "compare": compare}[args.command](args)


if __name__ == "__main__":
    main()
