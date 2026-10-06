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
from prism.conversion import eligible_linear, validate_plan
from prism.format import COMPONENTS, DTYPES, Component, TensorReader
from prism.loading import make_module
from prism.quantization import best_group_size, decode_config
from prism.runtime import check_bundle


def validate_headers(folder, variant="alpha", require_complete=True):
    folder = Path(folder)
    manifest = json.loads((folder / f"prism_{variant}_conversion.json").read_text(encoding="utf-8"))
    if not isinstance(manifest.get("components"), dict) or set(manifest["components"]) - set(COMPONENTS):
        raise ValueError("Invalid conversion manifest components")
    if require_complete and (not manifest.get("complete") or set(manifest["components"]) != set(COMPONENTS)):
        raise ValueError("Manifest does not describe all seven complete components")
    if manifest.get("precision") not in ("int8_convrot", "bf16") or type(manifest.get("mseclip")) is not bool:
        raise ValueError("Invalid conversion manifest recipe")
    recipe = "block-linear-regular-hadamard-mseclip" if manifest["mseclip"] else "block-linear-regular-hadamard-absmax"
    parts = {}
    details = {}
    for kind in COMPONENTS:
        if kind not in manifest["components"]:
            continue
        entry = manifest["components"][kind]
        if Path(entry["file"]).name != entry["file"]:
            raise ValueError(f"Component filename must be local: {entry['file']}")
        path = folder / entry["file"]
        if not path.exists() and not require_complete:
            continue
        component = Component.inspect(path, kind)
        if component.metadata.get("prism.upstream_commit") != UPSTREAM_COMMIT:
            raise ValueError(f"Incorrect upstream provenance: {kind}")
        if component.metadata.get("prism.variant") != variant:
            raise ValueError(f"Incorrect preview variant: {kind}")
        if component.metadata.get("prism.bundle_id") != manifest["bundle_id"]:
            raise ValueError(f"Manifest bundle ID differs from {kind}")
        if component.metadata.get("prism.quant_recipe") != recipe:
            raise ValueError(f"Incorrect quantization recipe: {kind}")
        parts[kind] = component
        with TensorReader(component.path) as reader:
            header = reader._header
            quant = [key for key in header if key.endswith(".comfy_quant")]
            scales = {key for key in header if key.endswith(".weight_scale")}
            expected_scales = {key[:-len(".comfy_quant")] + ".weight_scale" for key in quant}
            if scales != expected_scales:
                raise ValueError(f"Unpaired ConvRot scale records in {kind}")
            plan = {key: (component.path, key, info["shape"]) for key, info in header.items()
                    if not key.endswith((".comfy_quant", ".weight_scale"))}
            validate_plan(kind, component.config, plan)
            expected_quant = {key for key, (_, _, shape) in plan.items()
                              if manifest["precision"] == "int8_convrot" and eligible_linear(kind, key, shape)}
            if {key[:-len(".comfy_quant")] + ".weight" for key in quant} != expected_quant:
                raise ValueError(f"Quantized tensor set disagrees with the conversion recipe: {kind}")
            expected_precision = "int8_convrot" if expected_quant else "bf16"
            if component.metadata.get("prism.precision") != expected_precision:
                raise ValueError(f"Incorrect component storage precision: {kind}")
            with torch.device("meta"):
                model = make_module(component)
            expected_state = model.state_dict()
            for key, info in header.items():
                if key not in plan or key in expected_quant:
                    continue
                expected_dtype = "BF16" if expected_state[key].is_floating_point() else DTYPES[expected_state[key].dtype]
                if info["dtype"] != expected_dtype:
                    raise ValueError(f"Incorrect storage dtype for {kind}:{key}")
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
                if config["convrot_groupsize"] != best_group_size(weight["shape"][1]):
                    raise ValueError(f"ConvRot group size disagrees with the conversion recipe: {kind}:{prefix}")
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
    complete = set(parts) == set(COMPONENTS)
    if complete:
        check_bundle(parts)
    return {"scope": "structural headers, native shapes and quant markers/scales; not full-weight quality acceptance",
            "bundle_id": manifest["bundle_id"], "components": details, "complete": complete}


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
