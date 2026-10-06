"""Decode the same retained, already-denormalized Wan latent in BF16 and FP32.

Only the standalone video VAE is loaded; no transformer or sampling is invoked.
Run after the sampling process releases the GPU. Raw tensors and metrics support
inspection; finite values and a playable clip do not establish visual quality.
"""
import argparse
import hashlib
import inspect
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image, ImageDraw
import torch
from safetensors.torch import save_file

from prism.format import Component, TensorReader
from prism.loading import load_component


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_stats(tensor):
    value = tensor.detach().cpu()
    values = (value.float() if value.dtype == torch.bfloat16 else value).numpy().reshape(-1)
    mask = np.isfinite(values)
    finite = values if mask.all() else values[mask]
    result = {"shape": list(tensor.shape), "dtype": str(tensor.dtype), "count": int(values.size),
              "finite": bool(mask.all()), "nonfinite_count": int((~mask).sum()),
              "nan_count": int(np.isnan(values).sum()), "posinf_count": int(np.isposinf(values).sum()),
              "neginf_count": int(np.isneginf(values).sum())}
    if finite.size:
        percentiles = [.1, 1., 5., 50., 95., 99., 99.9]
        result.update(min=float(finite.min()), max=float(finite.max()),
                      mean=float(finite.mean(dtype=np.float64)), std=float(finite.std(dtype=np.float64)),
                      abs_max=float(np.abs(finite).max()),
                      percentiles={str(p): float(v) for p, v in zip(percentiles, np.percentile(finite, percentiles))})
    return result


def latent_distribution(latents):
    return {"global": tensor_stats(latents),
            "per_channel": [dict(channel=index, **tensor_stats(latents[:, index]))
                            for index in range(latents.shape[1])],
            "per_latent_time": [dict(latent_time=index, **tensor_stats(latents[:, :, index]))
                                for index in range(latents.shape[2])]}


def decoded_distribution(decoded):
    def frame_stats(value):
        result = tensor_stats(value)
        result["raw_le_minus_1_fraction"] = float((value <= -1).float().mean())
        result["raw_ge_plus_1_fraction"] = float((value >= 1).float().mean())
        return result
    return {"global": frame_stats(decoded),
            "per_frame": [dict(frame=index, **frame_stats(decoded[:, :, index]))
                          for index in range(decoded.shape[2])]}


def save_sheet(frames, path, fps, label):
    width, height = frames[0].size
    indices = np.linspace(0, len(frames) - 1, 9, dtype=int).tolist()
    sheet = Image.new("RGB", (width * 3, (height + 28) * 3), "#202020")
    draw = ImageDraw.Draw(sheet)
    for cell, index in enumerate(indices):
        x, y = cell % 3 * width, cell // 3 * (height + 28)
        sheet.paste(frames[index].convert("RGB"), (x, y))
        draw.text((x + 5, y + height + 3), f"{label} frame {index} / {index / fps:.3f}s", fill="white")
    sheet.save(path)
    return indices


def save_media(frames, folder, fps, label):
    frame_folder = folder / "frames"
    frame_folder.mkdir()
    for index, frame in enumerate(frames):
        frame.save(frame_folder / f"frame_{index:04d}.png")
    sheet = folder / "contact-sheet.png"
    indices = save_sheet(frames, sheet, fps, label)
    clip = folder / "decode.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-framerate", str(fps), "-i",
                    str(frame_folder / "frame_%04d.png"), "-frames:v", str(len(frames)),
                    "-an", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart", str(clip)], check=True)
    probe = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-count_frames",
                                              "-show_streams", "-of", "json", str(clip)]))
    stream = next(value for value in probe["streams"] if value["codec_type"] == "video")
    raw = subprocess.check_output(["ffmpeg", "-v", "error", "-i", str(clip), "-map", "0:v:0",
                                   "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"])
    width, height = frames[0].size
    expected_bytes = len(frames) * height * width * 3
    if len(raw) != expected_bytes or int(stream["nb_read_frames"]) != len(frames):
        raise RuntimeError("VAE clip full decode frame count mismatch")
    pixels = np.stack([np.asarray(frame.convert("RGB")) for frame in frames])
    return {"all_frames_png": str(frame_folder.resolve()), "contact_sheet": str(sheet.resolve()),
            "sheet_indices": indices, "clip": str(clip.resolve()), "clip_sha256": file_hash(clip),
            "clip_full_decode": True, "clip_frames": len(raw) // (height * width * 3),
            "clip_codec": stream["codec_name"], "clip_fps": stream["avg_frame_rate"],
            "clip_duration": float(stream["duration"]), "audio": "none: VAE-only video diagnostic",
            "width": width, "height": height,
            "per_frame_rgb_mean": pixels.mean(axis=(1, 2)).tolist(),
            "per_frame_zero_fraction": (pixels == 0).mean(axis=(1, 2, 3)).tolist(),
            "per_frame_255_fraction": (pixels == 255).mean(axis=(1, 2, 3)).tolist()}, pixels


def compare_decodes(bf16, fp32, bf16_pixels, fp32_pixels, folder, fps):
    difference = bf16.float() - fp32.float()
    reference_norm = torch.linalg.vector_norm(fp32.double()).item()
    result = {"raw_bf16_minus_fp32": tensor_stats(difference),
              "raw_exact_equal": bool(torch.equal(bf16.float(), fp32.float())),
              "raw_relative_l2": torch.linalg.vector_norm(difference.double()).item() / max(reference_norm, 1e-30),
              "per_frame": [dict(frame=index, **tensor_stats(difference[:, :, index]))
                            for index in range(difference.shape[2])]}
    save_file({"bf16_minus_fp32": difference.contiguous()}, str(folder / "raw-difference.safetensors"))
    pixel_difference = np.abs(bf16_pixels.astype(np.int16) - fp32_pixels.astype(np.int16))
    result["display_uint8_difference"] = {
        "mean_abs": float(pixel_difference.mean()), "max_abs": int(pixel_difference.max()),
        "different_pixel_channel_fraction": float((pixel_difference != 0).mean()),
        "per_frame_mean_abs": pixel_difference.mean(axis=(1, 2, 3)).tolist(),
        "per_frame_max_abs": pixel_difference.max(axis=(1, 2, 3)).tolist()}
    heatmaps = [Image.fromarray(np.clip(frame * 8, 0, 255).astype(np.uint8)) for frame in pixel_difference]
    target = folder / "display-difference-x8-contact-sheet.png"
    save_sheet(heatmaps, target, fps, "abs RGB diff x8")
    result["display_difference_sheet"] = str(target.resolve())
    return result


def decode_case(component, latents, precision, folder, device, settings):
    from diffusers.video_processor import VideoProcessor
    dtype = torch.bfloat16 if precision == "bf16" else torch.float32
    start = time.monotonic()
    model = load_component(component, dtype=dtype).to(device)
    video_input = latents.to(device)  # Identical FP32 values; no BF16 pre-cast or second denormalization.
    torch.cuda.reset_peak_memory_stats(device)
    try:
        if settings["vae_tiling"]:
            tiles = {}
            if settings["tile_size"] is not None:
                tiles["tile_sample_min_height"] = settings["tile_size"]
                tiles["tile_sample_min_width"] = settings["tile_size"]
            if settings["tile_stride"] is not None:
                tiles["tile_sample_stride_height"] = settings["tile_stride"]
                tiles["tile_sample_stride_width"] = settings["tile_stride"]
            model.enable_tiling(**tiles)
        else:
            model.disable_tiling()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=precision == "bf16"):
            decoded = model.decode(video_input).sample
        torch.cuda.synchronize(device)
        raw_cpu = decoded.detach().cpu().contiguous()
        report = {"model_dtype": str(model.dtype), "input_dtype": str(video_input.dtype),
                  "autocast": precision == "bf16", "decoded": decoded_distribution(raw_cpu),
                  "input_unchanged": bool(torch.equal(video_input.cpu(), latents)),
                  "expected_shape_from_latent": [1, 3, (latents.shape[2] - 1) * model.config.scale_factor_temporal + 1,
                                                 latents.shape[3] * model.config.scale_factor_spatial,
                                                 latents.shape[4] * model.config.scale_factor_spatial]}
        save_file({"decoded": raw_cpu}, str(folder / "decoded-video.safetensors"))
        pixels = None
        if report["decoded"]["global"]["finite"]:
            if list(decoded.shape) != report["expected_shape_from_latent"]:
                raise RuntimeError("Unexpected native Wan VAE decoded shape")
            # Preserve native output dtype through the original processor on CUDA.
            processor = VideoProcessor(vae_scale_factor=model.config.scale_factor_spatial)
            with torch.inference_mode():
                frames = processor.postprocess_video(decoded, output_type="pil")[0]
            report["media"], pixels = save_media(frames, folder, settings["fps"], precision.upper())
            report["status"] = "decoded-finite-requires-visual-inspection"
        else:
            report["status"] = "nonfinite-no-pixel-conversion"
        report["seconds"] = time.monotonic() - start
        report["peak_allocated_gib"] = torch.cuda.max_memory_allocated(device) / 2 ** 30
        return report, raw_cpu, pixels
    finally:
        del model, video_input
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latents", required=True, help="Actual video-latents.safetensors retained before VAE.decode")
    parser.add_argument("--vae", default="models/standalone/prism_alpha_video_vae_bf16.safetensors")
    parser.add_argument("--source-report", help="diagnostic.json with sampler settings; defaults beside latent")
    parser.add_argument("--output", required=True, help="New directory; existing results are never overwritten")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--fps", type=float, help="Explicit timing if no source report is available")
    parser.add_argument("--vae-tiling", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--tile-size", type=int)
    parser.add_argument("--tile-stride", type=int)
    args = parser.parse_args()
    latent_path, output = Path(args.latents).resolve(), Path(args.output).resolve()
    source_path = Path(args.source_report).resolve() if args.source_report else latent_path.parent / "diagnostic.json"
    source = json.loads(source_path.read_text(encoding="utf-8")) if source_path.is_file() else None
    sampler = source.get("sampler", {}) if source is not None else {}
    if source is None and (args.fps is None or args.vae_tiling is None):
        parser.error("Without diagnostic.json provide --fps and explicit --vae-tiling/--no-vae-tiling")
    settings = {"fps": args.fps if args.fps is not None else sampler.get("fps"),
                "vae_tiling": args.vae_tiling if args.vae_tiling is not None else sampler.get("vae_tiling", False),
                "tile_size": args.tile_size if args.tile_size is not None else sampler.get("tile_size"),
                "tile_stride": args.tile_stride if args.tile_stride is not None else sampler.get("tile_stride")}
    if not isinstance(settings["fps"], (int, float)) or isinstance(settings["fps"], bool) or not math.isfinite(settings["fps"]) or settings["fps"] <= 0:
        parser.error("A finite positive source FPS is required")
    if not isinstance(settings["vae_tiling"], bool):
        parser.error("Source vae_tiling must be boolean")
    for key in ("tile_size", "tile_stride"):
        if settings[key] is not None and (isinstance(settings[key], bool) or not isinstance(settings[key], int) or settings[key] <= 0):
            parser.error(f"{key} must be a positive integer")
    with TensorReader(latent_path, copy=True) as reader:
        latents = reader.get_tensor("latents")
    if latents.dtype != torch.float32 or latents.ndim != 5 or latents.shape[0] != 1 or min(latents.shape) < 1 or not torch.isfinite(latents).all():
        parser.error("Expected finite actual retained FP32 latents [1,C,T,H,W]")
    component = Component.inspect(args.vae, "video_vae")
    if latents.shape[1] != component.config["z_dim"]:
        parser.error("Latent channel count disagrees with VAE config")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        parser.error("This precision comparison requires the released CUDA GPU")
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    import diffusers
    from diffusers import AutoencoderKLWan
    report = {"scope": "Same actual denormalized latent; standalone Wan VAE only; no sampling",
              "script_sha256": file_hash(__file__), "torch_version": torch.__version__,
              "diffusers_version": diffusers.__version__, "vae_implementation_sha256": file_hash(inspect.getfile(AutoencoderKLWan)),
              "tf32_matmul": torch.backends.cuda.matmul.allow_tf32, "tf32_cudnn": torch.backends.cudnn.allow_tf32,
              "latent_file": str(latent_path), "latent_sha256": file_hash(latent_path),
              "latent_distribution": latent_distribution(latents), "already_denormalized": True,
              "source_report": str(source_path) if source is not None else None,
              "source_sampler": sampler, "source_components": source.get("components") if source else None,
              "vae_file": str(component.path), "vae_sha256": file_hash(component.path), "vae_config": component.config,
              "vae_precision": component.metadata["prism.precision"], "vae_upstream": component.metadata["prism.upstream_commit"],
              "decode_settings": settings, "cases": {}, "quality_acceptance": "pending direct visual inspection"}
    decoded, pixels = {}, {}
    for precision in ("bf16", "fp32"):
        folder = output / precision
        folder.mkdir()
        print(f"Decoding {precision}: VAE only, same retained FP32 latent", flush=True)
        try:
            report["cases"][precision], decoded[precision], pixels[precision] = decode_case(
                component, latents, precision, folder, device, settings)
        except Exception as error:
            report["cases"][precision] = {"status": "failed", "error": str(error), "traceback": traceback.format_exc()}
        (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    if len(decoded) == 2 and all(pixels.get(key) is not None for key in ("bf16", "fp32")):
        folder = output / "comparison"
        folder.mkdir()
        report["comparison"] = compare_decodes(decoded["bf16"], decoded["fp32"], pixels["bf16"], pixels["fp32"], folder, settings["fps"])
    report["latent_file_unchanged"] = file_hash(latent_path) == report["latent_sha256"]
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"report": str(output / "report.json"), "cases": {key: value["status"] for key, value in report["cases"].items()},
                      "comparison_available": "comparison" in report}, ensure_ascii=False, indent=2), flush=True)
    if any(value["status"] != "decoded-finite-requires-visual-inspection" for value in report["cases"].values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
