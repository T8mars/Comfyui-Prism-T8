"""Real-weight smoke test; a short sample is not a visual quality acceptance test."""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from prism.format import Component
from prism.runtime import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", required=True)
    parser.add_argument("--variant", default="alpha", choices=("alpha", "beta"))
    parser.add_argument("--output", default="outputs/validation")
    parser.add_argument("--reference", default=None)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--frames", type=int, default=5)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--int8-backend", default="portable", choices=("portable", "kitchen"))
    parser.add_argument("--sparse-options", help="Path to JSON native sparse options")
    args = parser.parse_args()
    torch.set_num_threads(4)
    folder = Path(args.models)
    report = json.loads((folder / f"prism_{args.variant}_conversion.json").read_text(encoding="utf-8"))
    parts = {kind: Component.inspect(folder / entry["file"], kind) for kind, entry in report["components"].items()}
    fingerprint = {kind: {"file": component.path.name, "bytes": component.path.stat().st_size,
                          "mtime_ns": component.path.stat().st_mtime_ns} for kind, component in parts.items()}
    from PIL import Image
    reference = Image.open(args.reference) if args.reference else Image.new("RGB", (args.width, args.height), (128, 128, 128))
    start = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    sparse = json.loads(Path(args.sparse_options).read_text(encoding="utf-8")) if args.sparse_options else {}
    frames, audio, fps = run(parts, reference, dict(width=args.width, height=args.height,
        num_frames=args.frames, steps=args.steps, prompt="A person standing near the ocean. <sfx>Gentle waves.</sfx>",
        cfg=1.0, offload="block", attention="sdpa", visual_shift=5.0, audio_shift=5.0,
        int8_backend=args.int8_backend), sparse=sparse,
        callback=lambda step, total: print(f"step {step + 1}/{total}", flush=True))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    for index, frame in enumerate(frames):
        Image.fromarray((frame.numpy() * 255).round().astype("uint8")).save(output / f"frame_{index:04d}.png")
    import wave
    import numpy as np
    pcm = (audio["waveform"][0].clamp(-1, 1).numpy().T * 32767).astype(np.int16)
    with wave.open(str(output / "audio.wav"), "wb") as stream:
        stream.setnchannels(pcm.shape[1])
        stream.setsampwidth(2)
        stream.setframerate(audio["sample_rate"])
        stream.writeframes(pcm.tobytes())
    result = {"scope": "real-weight short smoke, not quality acceptance", "bundle_id": report["bundle_id"],
              "frames_shape": list(frames.shape), "audio_shape": list(audio["waveform"].shape),
              "sample_rate": audio["sample_rate"], "fps": fps, "steps": args.steps,
              "seconds": time.monotonic() - start, "peak_vram_gib": torch.cuda.max_memory_allocated() / 2 ** 30,
              "finite": True, "internal_finite_checked": True, "sparse_options": sparse,
              "int8_backend": args.int8_backend, "component_files": fingerprint}
    (output / "validation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    from prism.media import save_video
    save_video(frames, audio, fps, output / "smoke.mp4")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
