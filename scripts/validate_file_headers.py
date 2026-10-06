"""Verify seven standalone file layouts/configs/quant markers without allocating model weights.

This is a structural check. Sampling and complete finite-weight checks belong to
the native loader and the real ComfyUI execution, not this header-only report.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from prism import UPSTREAM_COMMIT
from prism.conversion import validate_plan
from prism.format import COMPONENTS, Component, TensorReader
from prism.loading import make_module
from prism.quantization import decode_config
from prism.runtime import check_bundle


def validate_headers(folder, variant="alpha"):
    folder = Path(folder)
    manifest = json.loads((folder / f"prism_{variant}_conversion.json").read_text(encoding="utf-8"))
    if not manifest.get("complete") or set(manifest["components"]) != set(COMPONENTS):
        raise ValueError("Manifest does not describe all seven complete components")
    parts = {}
    details = {}
    for kind in COMPONENTS:
        entry = manifest["components"][kind]
        if Path(entry["file"]).name != entry["file"]:
            raise ValueError(f"Component filename must be local: {entry['file']}")
        component = Component.inspect(folder / entry["file"], kind)
        if component.metadata.get("prism.upstream_commit") != UPSTREAM_COMMIT:
            raise ValueError(f"Incorrect upstream provenance: {kind}")
        if component.metadata.get("prism.variant") != variant:
            raise ValueError(f"Incorrect preview variant: {kind}")
        parts[kind] = component
        with TensorReader(component.path) as reader:
            header = reader._header
            quant = [key for key in header if key.endswith(".comfy_quant")]
            plan = {key: (component.path, key, info["shape"]) for key, info in header.items()
                    if not key.endswith((".comfy_quant", ".weight_scale"))}
            validate_plan(kind, component.config, plan)
            with torch.device("meta"):
                model = make_module(component)
            for marker in quant:
                prefix = marker[:-len(".comfy_quant")]
                module = model.get_submodule(prefix)
                if not isinstance(module, torch.nn.Linear):
                    raise ValueError(f"ConvRot marker does not reference a native Linear: {kind}:{prefix}")
                config = decode_config(reader.get_tensor(marker))
                weight = header.get(prefix + ".weight")
                scale_name = prefix + ".weight_scale"
                if not weight or weight["dtype"] != "I8" or scale_name not in header:
                    raise ValueError(f"Incomplete INT8 ConvRot record: {kind}:{prefix}")
                if weight["shape"][1] % config["convrot_groupsize"]:
                    raise ValueError(f"Invalid ConvRot group size: {kind}:{prefix}")
                scale = reader.get_tensor(scale_name)
                if (scale.dtype != torch.float32 or scale.shape not in
                    (torch.Size([]), torch.Size([1]), torch.Size([weight["shape"][0], 1])) or
                    not torch.isfinite(scale).all() or not (scale > 0).all()):
                    raise ValueError(f"Invalid ConvRot scale: {kind}:{prefix}")
            if len(quant) != entry["quantized_linears"] or len(plan) != entry["source_tensors"]:
                raise ValueError(f"Manifest tensor counts disagree with {kind}")
            details[kind] = {"file": component.path.name, "bytes": component.path.stat().st_size,
                             "tensors": len(header), "native_tensors": len(plan),
                             "quantized_linears": len(quant), "native_shapes_valid": True,
                             "all_quant_scales_finite_positive": True}
    check_bundle(parts)
    if next(iter(parts.values())).metadata["prism.bundle_id"] != manifest["bundle_id"]:
        raise ValueError("Manifest bundle ID differs from component bundle ID")
    return {"scope": "structural headers, native shapes and quant markers/scales; not full-weight quality acceptance",
            "bundle_id": manifest["bundle_id"], "components": details, "complete": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default="models/standalone")
    parser.add_argument("--variant", default="alpha", choices=("alpha", "beta"))
    parser.add_argument("--output", default="outputs/file-header-validation.json")
    args = parser.parse_args()
    torch.set_num_threads(4)
    report = validate_headers(args.models, args.variant)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".json.part")
    temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
    temporary.replace(output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
