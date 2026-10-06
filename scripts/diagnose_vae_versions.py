"""Isolated Diffusers 0.33.0/current Wan VAE comparison; no denoising.

Default mode requires --validate-only and uses CPU fixtures/meta models only.
Actual latent decoding and native reference-plus-zero conditioning encoding are
available with explicit --run-codec after the sampling process releases CUDA.
Install old Diffusers into --old-package with pip --no-deps --target; never
replace the working virtual environment. Results are written to a new directory.
"""
import argparse
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_comparison(old, new):
    """Compare active numerical bodies, without annotations/docstrings.

    Cache arguments changed from positional to named arguments. first_chunk is
    ignored by ordinary WanUpBlock; normalizing these *call arguments* is
    separately disclosed rather than counted as a literal source match.
    """
    def methods(path):
        tree = ast.parse(Path(path).read_text(encoding="utf-8"))
        return {f"{node.name}.{method.name}": method for node in tree.body
                if isinstance(node, ast.ClassDef) for method in node.body
                if isinstance(method, ast.FunctionDef)}

    class Normalize(ast.NodeTransformer):
        def visit_AnnAssign(self, node):
            return ast.Assign(targets=[node.target], value=node.value) if node.value else None

        def visit_Call(self, node):
            self.generic_visit(node)
            node.keywords = [kw for kw in node.keywords if kw.arg != "first_chunk"]
            cache_args = {kw.arg: kw.value for kw in node.keywords if kw.arg in ("feat_cache", "feat_idx")}
            if set(cache_args) == {"feat_cache", "feat_idx"} and len(node.args) == 1:
                node.args.extend([cache_args["feat_cache"], cache_args["feat_idx"]])
                node.keywords = [kw for kw in node.keywords if kw.arg not in cache_args]
            return node

    old_methods, new_methods = methods(old), methods(new)
    result = {}
    names = ["WanCausalConv3d", "WanRMS_norm", "WanUpsample", "WanResample",
             "WanResidualBlock", "WanAttentionBlock", "WanMidBlock", "WanEncoder3d",
             "WanUpBlock", "WanDecoder3d"]
    for name in names:
        key = name + ".forward"
        bodies = []
        normalized = []
        for method in (old_methods[key], new_methods[key]):
            body = copy.deepcopy(method.body)
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
                body = body[1:]
            tree = ast.Module(body=body, type_ignores=[])
            bodies.append(ast.dump(tree, include_attributes=False))
            normalized.append(ast.dump(Normalize().visit(tree), include_attributes=False))
        result[key] = {"body_ast_exact": bodies[0] == bodies[1],
                       "body_ast_equal_with_equivalent_cache_call_arguments": normalized[0] == normalized[1]}
    return {"old_file": str(old), "old_sha256": sha256(old), "current_file": str(new),
            "current_sha256": sha256(new), "forward_methods": result,
            "scope": "Active ordinary Wan2.1 forward bodies, not all package methods",
            "nonliteral_differences": [
                "0.37 adds optional residual blocks, patchification, spatial tiling and batch slicing",
                "Cache arguments use identical callee parameter positions in 0.33 and named feat_cache/feat_idx in 0.37",
                "Ordinary WanUpBlock accepts first_chunk but its numerical body does not use it",
                "0.37 caches static causal-convolution counts; 0.33 recomputes the same counts",
                "0.33 _encode splits/reconcatenates mu/logvar; 0.37 returns quant_conv output directly",
                "0.33 has no tiling API; this comparison requires source vae_tiling=False"]}


def worker(args):
    # Version selection occurs before importing torch/diffusers or Prism loaders.
    if args.version == "0.33.0":
        sys.path.insert(0, str(Path(args.old_package).resolve()))
    sys.path.insert(0, str(ROOT))
    os.environ["USE_TF"] = "0"
    import inspect
    import torch
    import diffusers
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file
    from prism.format import Component, TensorReader
    from prism.loading import load_component
    from diffusers import AutoencoderKLWan
    from diagnose_vae_precision import tensor_stats, save_media, latent_distribution, decoded_distribution

    actual_version = diffusers.__version__
    if actual_version != args.version:
        raise RuntimeError(f"Wrong Diffusers selected: {actual_version}, expected {args.version}")
    package_path = Path(diffusers.__file__).resolve()
    if args.version == "0.33.0" and not package_path.is_relative_to(Path(args.old_package).resolve()):
        raise RuntimeError("Old package did not load from the isolated target")
    output = Path(args.output)
    output.mkdir(exist_ok=False)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    component = Component.inspect(args.vae, "video_vae")
    config = dict(component.config)
    if config.get("is_residual", False) or config.get("patch_size") is not None or config.get("decoder_base_dim") not in (None, config["base_dim"]):
        raise ValueError("This comparison is limited to the actual ordinary Prism Wan2.1 configuration")
    if config.get("in_channels", 3) != 3 or config.get("out_channels", 3) != 3:
        raise ValueError("Old Wan implementation fixes RGB channels")
    with torch.device("meta"):
        meta = AutoencoderKLWan.from_config(config)
    expected = {key: list(value.shape) for key, value in meta.state_dict().items()}
    with safe_open(str(component.path), framework="pt", device="cpu") as handle:
        actual = {key: list(handle.get_slice(key).get_shape()) for key in handle.keys()}
    if expected != actual:
        raise RuntimeError("Actual standalone VAE keys/shapes differ from this version's model")
    graph = []
    for name, module in meta.named_modules():
        entry = {"name": name, "class": type(module).__name__}
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Conv3d)):
            entry.update(stride=list(module.stride), padding=list(module.padding), dilation=list(module.dilation),
                         groups=module.groups, kernel_size=list(module.kernel_size))
        if isinstance(module, torch.nn.Upsample):
            entry.update(mode=module.mode, scale_factor=module.scale_factor, align_corners=module.align_corners)
        graph.append(entry)
    source = Path(inspect.getfile(AutoencoderKLWan)).resolve()
    report = {"diffusers_version": actual_version, "package_file": str(package_path),
              "vae_source_file": str(source), "vae_source_sha256": sha256(source),
              "vae_component": str(component.path), "config": config, "state_shapes": expected,
              "all_standalone_keys_shapes_match": True, "native_module_graph": graph,
              "standalone_tensor_count": len(expected), "cases": {}}
    del meta
    if args.worker_mode == "cpu":
        torch.manual_seed(12003)
        small_config = {**config, "base_dim": 4, "num_res_blocks": 2}
        model = AutoencoderKLWan.from_config(small_config).eval()
        fixture = Path(args.fixture)
        if args.version == "0.33.0":
            fixture.mkdir(exist_ok=False)
            save_file({k: v.contiguous() for k, v in model.state_dict().items()}, str(fixture / "weights.safetensors"))
            generator = torch.Generator(device="cpu").manual_seed(3401)
            condition = torch.zeros(1, 3, 9, 16, 24)
            condition[:, :, 0] = torch.rand(1, 3, 16, 24, generator=generator) * 2 - 1
            latent = torch.randn(1, config["z_dim"], 3, 2, 3, generator=generator)
            save_file({"condition": condition, "latent": latent}, str(fixture / "inputs.safetensors"))
        model.load_state_dict(load_file(str(fixture / "weights.safetensors")), strict=True)
        values = load_file(str(fixture / "inputs.safetensors"))
        with torch.inference_mode():
            encoded = model.encode(values["condition"]).latent_dist.mode()
            decoded = model.decode(values["latent"]).sample
            encoded_again = model.encode(values["condition"]).latent_dist.mode()
            decoded_again = model.decode(values["latent"]).sample
        if not torch.equal(encoded, encoded_again) or not torch.equal(decoded, decoded_again):
            raise RuntimeError("Temporal VAE caches did not reset between complete calls")
        save_file({"condition_latent": encoded.contiguous(), "decoded": decoded.contiguous()}, str(output / "cpu-results.safetensors"))
        report["cases"]["cpu_complete_encode_decode"] = {
            "scope": "Small deterministic full Wan2.1 graph with native temporal structure; not real-model visual QA",
            "shared_weights_sha256": sha256(fixture / "weights.safetensors"),
            "shared_inputs_sha256": sha256(fixture / "inputs.safetensors"),
            "condition": tensor_stats(values["condition"]), "latent": tensor_stats(values["latent"]),
            "encoded": tensor_stats(encoded), "decoded": tensor_stats(decoded),
            "cache_reset_exact_after_interleaved_encode_decode": True}
        report["cuda_initialized"] = torch.cuda.is_initialized()
        if report["cuda_initialized"]:
            raise RuntimeError("CPU-only validation unexpectedly initialized CUDA")
    else:
        device = torch.device(args.device)
        if device.type != "cuda" or not torch.cuda.is_available():
            raise ValueError("--run-codec requires the released CUDA GPU")
        source_report = json.loads(Path(args.source_report).read_text(encoding="utf-8"))
        sampler = source_report["sampler"]
        if sampler["vae_tiling"]:
            raise ValueError("Diffusers 0.33 lacks Wan tiling; same untiled source settings are required")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        dtype = torch.bfloat16
        model = load_component(component, dtype=dtype).to(device)
        if hasattr(model, "disable_tiling"):
            model.disable_tiling()
            model.disable_slicing()
        with TensorReader(args.latents, copy=True) as reader:
            latent = reader.get_tensor("latents")
        if latent.dtype != torch.float32 or not torch.isfinite(latent).all() or latent.ndim != 5 or latent.shape[0] != 1 or latent.shape[1] != config["z_dim"]:
            raise ValueError("Expected actual retained finite denormalized FP32 native latent")
        torch.cuda.reset_peak_memory_stats(device)
        latent_gpu = latent.to(device)
        with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
            decoded = model.decode(latent_gpu).sample
        if not torch.isfinite(decoded).all():
            raise RuntimeError("Version decode produced nonfinite returned pixels")
        raw = decoded.cpu().contiguous()
        save_file({"decoded": raw}, str(output / "decoded-video.safetensors"))
        from diffusers.video_processor import VideoProcessor
        processor = VideoProcessor(vae_scale_factor=config.get("scale_factor_spatial", 8))
        frames = processor.postprocess_video(decoded, output_type="pil")[0]
        report["cases"]["actual_latent_decode"], _ = save_media(frames, output, sampler["fps"], actual_version)
        report["cases"]["actual_latent_decode"].update(decoded=decoded_distribution(raw),
            input_distribution=latent_distribution(latent), input_sha256=sha256(args.latents),
            input_unchanged=torch.equal(latent_gpu.cpu(), latent), model_dtype=str(model.dtype),
            decode_autocast=True, returned_output_already_clamped=True)
        del latent_gpu, decoded
        # Match native I2VA prepare_latents: preprocess image, append normalized-domain
        # zeros, encode in VAE dtype without autocast, mode(), then latent normalization.
        from PIL import Image
        image_path = Path(args.image)
        with Image.open(image_path) as original:
            # Local implementation matches crop_reference; avoid loading pipeline/DiT.
            image = original.convert("RGB")
            width, height = sampler["width"], sampler["height"]
            w, h = image.size
            ratio = width / height
            if w / h > ratio:
                crop_width = int(h * ratio)
                offset = (w - crop_width) // 2
                image = image.crop((offset, 0, offset + crop_width, h))
            elif w / h < ratio:
                crop_height = int(w / ratio)
                offset = (h - crop_height) // 2
                image = image.crop((0, offset, w, offset + crop_height))
            image = image.resize((width, height), Image.Resampling.LANCZOS)
        image.save(output / "cropped-reference.png")
        first = processor.preprocess(image, height=height, width=width).to(device, dtype=torch.float32).unsqueeze(2)
        condition = torch.cat([first, first.new_zeros(1, 3, sampler["num_frames"] - 1, height, width)], dim=2).to(dtype)
        with torch.inference_mode(), torch.autocast("cuda", enabled=False):
            condition_latent = model.encode(condition).latent_dist.mode()
        condition_float = condition_latent.float()
        mean = torch.tensor(config["latents_mean"], dtype=torch.float32, device=device).view(1, -1, 1, 1, 1)
        std = torch.tensor(config["latents_std"], dtype=torch.float32, device=device).view(1, -1, 1, 1, 1)
        normalized = (condition_float - mean) * (1.0 / std)
        save_file({"native_mode_latent": condition_latent.cpu().contiguous(), "normalized_condition_latent": normalized.cpu().contiguous()},
                  str(output / "condition-latents.safetensors"))
        save_file({"video_condition": condition.cpu().contiguous()}, str(output / "condition-input.safetensors"))
        report["cases"]["reference_plus_zero_encode"] = {
            "image": str(image_path.resolve()), "image_sha256": sha256(image_path), "encode_autocast": False,
            "condition_dtype": str(condition.dtype), "condition_shape": list(condition.shape),
            "all_frames_after_reference_exact_zero": bool((condition[:, :, 1:] == 0).all()),
            "mode_latent": latent_distribution(condition_latent), "normalized": latent_distribution(normalized)}
        report["peak_allocated_gib"] = torch.cuda.max_memory_allocated(device) / 2 ** 30
        report["tf32_matmul"] = report["tf32_cudnn"] = False
        report["quality_acceptance"] = "Requires direct contact-sheet inspection; numeric equality is codec evidence only"
        del model
        torch.cuda.empty_cache()
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")


def compare_files(left, right):
    import torch
    from safetensors.torch import load_file
    a, b = load_file(str(left)), load_file(str(right))
    if set(a) != set(b):
        raise ValueError("Comparison tensor keys differ")
    result = {}
    for key in a:
        if a[key].shape != b[key].shape or a[key].dtype != b[key].dtype:
            raise ValueError(f"Comparison shape/dtype differs for {key}")
        difference = a[key].double() - b[key].double()
        result[key] = {"exact_equal": torch.equal(a[key], b[key]), "max_abs": difference.abs().max().item(),
                       "relative_l2": torch.linalg.vector_norm(difference).item() / max(torch.linalg.vector_norm(b[key].double()).item(), 1e-30),
                       "shape": list(a[key].shape), "dtype": str(a[key].dtype)}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--validate-only", action="store_true", help="CPU fixture and actual config/header validation; no CUDA")
    mode.add_argument("--run-codec", action="store_true", help="Actual decode and native conditioning encode only; no denoising")
    parser.add_argument("--old-package", default=str(ROOT / ".research/diffusers-0.33.0"))
    parser.add_argument("--vae", default=str(ROOT / "models/standalone/prism_alpha_video_vae_bf16.safetensors"))
    parser.add_argument("--latents", default=str(ROOT / "outputs/quality_int8_latent_probe/video-latents.safetensors"))
    parser.add_argument("--source-report", default=str(ROOT / "outputs/quality_int8_latent_probe/diagnostic.json"))
    parser.add_argument("--image", default=str(ROOT / "examples/prism_official_case5.png"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True)
    parser.add_argument("--version", help=argparse.SUPPRESS)
    parser.add_argument("--worker-mode", choices=("cpu", "cuda"), help=argparse.SUPPRESS)
    parser.add_argument("--fixture", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_mode:
        worker(args)
        return
    if not args.validate_only and not args.run_codec:
        parser.error("Choose --validate-only or explicit --run-codec")
    old_package = Path(args.old_package).resolve()
    if not (old_package / "diffusers/__init__.py").is_file():
        parser.error("Install diffusers==0.33.0 into --old-package using --no-deps --target")
    os.environ["USE_TF"] = "0"
    import inspect
    import diffusers
    from diffusers import AutoencoderKLWan
    current = diffusers.__version__
    if current != "0.37.1":
        parser.error(f"Prepared comparison expects current0.37.1, got {current}")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {"scope": "Only Wan VAE; never a transformer, text encoder or denoising pipeline",
              "script_sha256": sha256(__file__), "mode": "cpu" if args.validate_only else "actual-codec",
              "source_audit": source_comparison(old_package / "diffusers/models/autoencoders/autoencoder_kl_wan.py", inspect.getfile(AutoencoderKLWan)),
              "cases": {}, "existing_environment_not_modified": True}
    for version in ("0.33.0", current):
        folder = output / version
        command = [sys.executable, str(Path(__file__).resolve()), "--worker-mode", "cpu" if args.validate_only else "cuda",
                   "--version", version, "--old-package", str(old_package), "--vae", str(Path(args.vae).resolve()),
                   "--latents", str(Path(args.latents).resolve()), "--source-report", str(Path(args.source_report).resolve()),
                   "--image", str(Path(args.image).resolve()), "--device", args.device,
                   "--output", str(folder), "--fixture", str(output / "cpu-fixture")]
        print(f"Isolated {version} {report['mode']}; no denoising", flush=True)
        subprocess.run(command, check=True, cwd=ROOT)
        report["cases"][version] = json.loads((folder / "report.json").read_text(encoding="utf-8"))
    a, b = report["cases"]["0.33.0"], report["cases"][current]
    report["actual_full_model_graph_exact_equal"] = a["native_module_graph"] == b["native_module_graph"]
    report["actual_full_model_state_shapes_exact_equal"] = a["state_shapes"] == b["state_shapes"]
    if args.validate_only:
        report["cpu_numerics"] = compare_files(output / "0.33.0/cpu-results.safetensors", output / f"{current}/cpu-results.safetensors")
        report["cuda_initialized"] = any(case["cuda_initialized"] for case in report["cases"].values())
    else:
        report["decode_comparison"] = compare_files(output / "0.33.0/decoded-video.safetensors", output / f"{current}/decoded-video.safetensors")
        report["conditioning_input_comparison"] = compare_files(output / "0.33.0/condition-input.safetensors", output / f"{current}/condition-input.safetensors")
        report["conditioning_latent_comparison"] = compare_files(output / "0.33.0/condition-latents.safetensors", output / f"{current}/condition-latents.safetensors")
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Version comparison evidence: {output / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
