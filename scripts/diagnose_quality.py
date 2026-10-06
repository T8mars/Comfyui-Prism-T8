"""Same-settings BF16/INT8 quality comparison, retaining actual VAE inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from diffusers import AutoencoderKLWan
from safetensors.torch import save_file
from prism.format import Component, COMPONENTS
from prism.settings import GENERATION_DEFAULTS
from prism.runtime import run
import prism.runtime as runtime
from prism.media import save_video
from review_samples import decode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default="models/standalone")
    parser.add_argument("--baseline-dir", help="Override converted BF16 components when available")
    parser.add_argument("--history", default="outputs/comfyui-review/reports/history.json")
    parser.add_argument("--prompt-id", default="46d565ae-783c-45f4-b8a7-26d6aebfc279")
    parser.add_argument("--output", required=True)
    parser.add_argument("--empty-negative", action="store_true", help="Reproduce the earlier missing official negative prompt")
    parser.add_argument("--int8-backend", choices=("portable", "kitchen"))
    parser.add_argument("--width", type=int, help="Change only the spatial test width")
    parser.add_argument("--height", type=int, help="Change only the spatial test height")
    parser.add_argument("--num-frames", type=int, help="Explicit temporal test override")
    parser.add_argument("--vae-tiling", action=argparse.BooleanOptionalAction, default=None)
    args = parser.parse_args()
    torch.set_num_threads(4)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    code_files = [ROOT / "prism/runtime.py", ROOT / "prism/offload.py",
        ROOT / "prism/quantization.py", ROOT / "prism/native/models/modules/wan_video_dit.py",
        ROOT / "prism/native/models/modules/interactionv2.py"]
    code_hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                   for path in code_files}
    job = json.loads(Path(args.history).read_text(encoding="utf-8"))[args.prompt_id]
    sampler = next(n["inputs"] for n in job["prompt"][2].values() if n["class_type"] == "PrismNativeSampler")
    settings = {k: v for k, v in sampler.items() if k in GENERATION_DEFAULTS}
    settings["negative_prompt"] = "" if args.empty_negative else GENERATION_DEFAULTS["negative_prompt"]
    if args.int8_backend:
        settings["int8_backend"] = args.int8_backend
    for key in ("width", "height", "num_frames", "vae_tiling"):
        value = getattr(args, key)
        if value is not None:
            settings[key] = value
    parts = {}
    for kind in COMPONENTS:
        original_precision = "bf16" if kind.endswith("vae") else "int8_convrot"
        path = Path(args.models) / f"prism_alpha_{kind}_{original_precision}.safetensors"
        if args.baseline_dir:
            baseline = Path(args.baseline_dir) / f"prism_alpha_{kind}_bf16.safetensors"
            if baseline.exists():
                path = baseline
        parts[kind] = Component.inspect(path, kind)
    original = AutoencoderKLWan.decode
    def retain_decode(vae, latents, *positional, **kwargs):
        save_file({"latents": latents.detach().float().cpu().contiguous()}, str(output / "video-latents.safetensors"))
        return original(vae, latents, *positional, **kwargs)
    AutoencoderKLWan.decode = retain_decode
    original_load = runtime.load_component
    def observed_load(component, *positional, **kwargs):
        print(f"loading {component.kind} ({component.metadata['prism.precision']})", flush=True)
        loaded = original_load(component, *positional, **kwargs)
        print(f"loaded {component.kind}", flush=True)
        return loaded
    runtime.load_component = observed_load
    from PIL import Image
    reference = Image.open(ROOT / "examples/prism_official_case5.png")
    start = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    try:
        frames, audio, fps = run(parts, reference, settings, sparse={},
            callback=lambda step, total: print(f"step {step+1}/{total}", flush=True))
    finally:
        AutoencoderKLWan.decode = original
        runtime.load_component = original_load
    save_video(frames, audio, fps, output / "sample.mp4")
    result = {"sampler": settings, "seconds": time.monotonic() - start,
        "peak_vram_gib": torch.cuda.max_memory_allocated() / 2 ** 30,
        "code_at_start": code_hashes,
        "components": {k: {"path": str(c.path), "precision": c.metadata["prism.precision"]} for k, c in parts.items()},
        "media": decode(output / "sample.mp4", settings, output / "review")}
    (output / "diagnostic.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
