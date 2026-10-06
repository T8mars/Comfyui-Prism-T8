"""VAE-only round trips of a real reference image repeated across all frames.

Known identical input frames help locate temporal codec corruption. Both encode
and decode vary with precision here; use diagnose_vae_precision.py for an A/B
that instead holds an actual denoiser-produced latent fixed. No sampling occurs.
"""
import argparse
import json
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image
import torch
from safetensors.torch import save_file

from prism.format import Component
from prism.loading import load_component
from prism.runtime import crop_reference
from diagnose_vae_precision import file_hash, latent_distribution, decode_case, compare_decodes


def encode_reference(component, inputs, precision, device):
    dtype = torch.bfloat16 if precision == "bf16" else torch.float32
    model = load_component(component, dtype=dtype).to(device)
    model.disable_tiling()
    start = time.monotonic()
    try:
        # Native conditioning encode uses VAE dtype and has no surrounding autocast.
        with torch.inference_mode(), torch.autocast("cuda", enabled=False):
            latents = model.encode(inputs.to(device=device, dtype=dtype)).latent_dist.mode()
        torch.cuda.synchronize(device)
        result = latents.detach().cpu().contiguous()
        if not torch.isfinite(result).all():
            raise RuntimeError("Known-input VAE encode produced non-finite latent")
        return result, time.monotonic() - start
    finally:
        del model
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--num-frames", type=int, required=True)
    parser.add_argument("--fps", type=float, required=True)
    parser.add_argument("--vae", default="models/standalone/prism_alpha_video_vae_bf16.safetensors")
    parser.add_argument("--output", required=True, help="New output directory")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    import math
    if min(args.width, args.height, args.num_frames) <= 0 or not math.isfinite(args.fps) or args.fps <= 0:
        parser.error("Positive dimensions/frame count and finite positive FPS required")
    component = Component.inspect(args.vae, "video_vae")
    spatial, temporal = component.config["scale_factor_spatial"], component.config["scale_factor_temporal"]
    if args.width % spatial or args.height % spatial or (args.num_frames - 1) % temporal:
        parser.error("Reference dimensions/frame count must match native VAE spatial/temporal factors")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        parser.error("Run only after the sampling process releases its CUDA GPU")
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    from diffusers.video_processor import VideoProcessor
    image_path, output = Path(args.image).resolve(), Path(args.output).resolve()
    with Image.open(image_path) as original:
        reference = crop_reference(original, args.height, args.width)
    inputs = VideoProcessor(vae_scale_factor=spatial).preprocess_video(
        [reference.copy() for _ in range(args.num_frames)], height=args.height, width=args.width)
    assert torch.equal(inputs, inputs[:, :, :1].expand_as(inputs))
    output.mkdir(parents=True, exist_ok=False)
    reference.save(output / "cropped-reference.png")
    save_file({"video_input": inputs.contiguous()}, str(output / "repeated-input.safetensors"))
    report = {"scope": "Real image repeated identically; standalone VAE encode(mode)/decode only; no denoiser",
              "script_sha256": file_hash(__file__), "torch_version": torch.__version__,
              "image": str(image_path), "image_sha256": file_hash(image_path),
              "vae": str(component.path), "vae_sha256": file_hash(component.path),
              "input_shape": list(inputs.shape), "all_input_frames_identical": True,
              "tf32_matmul": False, "tf32_cudnn": False, "encode_autocast": False,
              "cases": {}, "quality_acceptance": "pending direct visual inspection"}
    decoded, pixels = {}, {}
    for precision in ("bf16", "fp32"):
        folder = output / precision
        folder.mkdir()
        print(f"Reference VAE-only round trip {precision}, {args.num_frames} repeated real frames", flush=True)
        try:
            encoded, seconds = encode_reference(component, inputs, precision, device)
            save_file({"latents": encoded}, str(folder / "encoded-latents.safetensors"))
            result, decoded[precision], pixels[precision] = decode_case(
                component, encoded.float(), precision, folder, device,
                {"fps": args.fps, "vae_tiling": False, "tile_size": None, "tile_stride": None})
            result["encode_seconds"] = seconds
            result["encoded_latent"] = latent_distribution(encoded)
            if pixels[precision] is not None:
                reference_pixels = np.asarray(reference).astype(np.int16)
                error = np.abs(pixels[precision].astype(np.int16) - reference_pixels[None])
                result["per_frame_reference_rgb_mean_abs_error"] = error.mean(axis=(1, 2, 3)).tolist()
                result["per_frame_first_decoded_rgb_mean_abs_difference"] = np.abs(
                    pixels[precision].astype(np.int16) - pixels[precision][:1].astype(np.int16)).mean(axis=(1, 2, 3)).tolist()
            report["cases"][precision] = result
        except Exception as error:
            report["cases"][precision] = {"status": "failed", "error": str(error), "traceback": traceback.format_exc()}
        (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    if len(decoded) == 2 and all(pixels.get(key) is not None for key in ("bf16", "fp32")):
        folder = output / "comparison"
        folder.mkdir()
        report["comparison"] = compare_decodes(decoded["bf16"], decoded["fp32"], pixels["bf16"], pixels["fp32"], folder, args.fps)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Reference round-trip report: {output / 'report.json'}", flush=True)
    if any(value["status"] != "decoded-finite-requires-visual-inspection" for value in report["cases"].values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
