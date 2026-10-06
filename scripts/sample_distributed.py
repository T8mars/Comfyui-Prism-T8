"""Optional Linux torchrun entry point using the same standalone component files.

torchrun --nproc_per_node=4 scripts/sample_distributed.py --models models/standalone
    --reference reference.png --sp-size 4 --width 1920 --height 1072
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import torch.distributed as dist
from PIL import Image
from prism.distributed import initialize
from prism.format import Component
from prism.runtime import run
from prism.media import save_video
from prism.settings import GENERATION_DEFAULTS


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", required=True)
    parser.add_argument("--variant", default="alpha", choices=("alpha", "beta"))
    parser.add_argument("--reference", required=True)
    parser.add_argument("--prompt", default="A person standing near the ocean. <sfx>Waves.</sfx>")
    parser.add_argument("--audio-prompt", default="")
    parser.add_argument("--negative-prompt", default=GENERATION_DEFAULTS["negative_prompt"])
    parser.add_argument("--output", default="outputs/distributed.mp4")
    parser.add_argument("--sp-size", type=int, default=1)
    parser.add_argument("--fsdp", action="store_true", help="BF16 components only; INT8 buffers cannot be FSDP sharded")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--frames", type=int, default=205)
    parser.add_argument("--fps", type=float, default=24.)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--cfg", type=float, default=5.)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--visual-shift", type=float, default=9.)
    parser.add_argument("--audio-shift", type=float, default=7.)
    parser.add_argument("--sparse-json", help="Full native sparse settings JSON file")
    parser.add_argument("--int8-backend", choices=("portable", "kitchen"), default="portable")
    parser.add_argument("--vae-tiling", action="store_true")
    args = parser.parse_args(argv)
    if args.sp_size < 1:
        parser.error("--sp-size must be a positive integer")
    return args


def main():
    args = parse_args()
    if not dist.is_nccl_available():
        raise RuntimeError("Native distributed Prism requires CUDA NCCL, normally Linux; use the single-GPU ComfyUI sampler on Windows")
    torch.set_num_threads(4)
    folder = Path(args.models)
    report = json.loads((folder / f"prism_{args.variant}_conversion.json").read_text(encoding="utf-8"))
    parts = {kind: Component.inspect(folder / entry["file"], kind) for kind, entry in report["components"].items()}
    if args.fsdp and any(part.metadata.get("prism.precision") == "int8_convrot" for part in parts.values()):
        raise ValueError("Use --precision bf16 conversion for FSDP, or omit --fsdp for INT8 + SP + block offload")
    device, state = initialize(args.sp_size)
    try:
        sparse = json.loads(Path(args.sparse_json).read_text(encoding="utf-8")) if args.sparse_json else {}
        settings = dict(prompt=args.prompt, audio_prompt=args.audio_prompt, negative_prompt=args.negative_prompt,
            width=args.width, height=args.height, num_frames=args.frames, fps=args.fps, steps=args.steps,
            cfg=args.cfg, seed=args.seed, visual_shift=args.visual_shift, audio_shift=args.audio_shift,
            offload="cpu" if args.fsdp else "block", attention="sdpa", int8_backend=args.int8_backend,
            vae_tiling=args.vae_tiling)
        frames, audio, fps = run(parts, Image.open(args.reference), settings, sparse=sparse, device=device,
            fsdp_mesh=state.fsdp_mesh if args.fsdp else None,
            callback=lambda step, total: print(f"rank={dist.get_rank()} step={step + 1}/{total}", flush=True))
        if dist.get_rank() == 0:
            save_video(frames, audio, fps, args.output)
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
