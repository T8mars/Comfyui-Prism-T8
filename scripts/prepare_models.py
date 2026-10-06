"""Download, convert missing components, then run a real-weight smoke test."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="checkpoints/official")
    parser.add_argument("--output", default="models/standalone")
    parser.add_argument("--variant", choices=("alpha", "beta"), default="alpha")
    parser.add_argument("--wait-for-preview", action="store_true", help="A download is already running; wait for its atomically completed preview")
    parser.add_argument("--skip-smoke", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    source = (root / args.source).resolve()
    output = (root / args.output).resolve()
    status_path = root / "outputs" / f"{args.variant}_build_status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    def status(phase, **extra):
        data = {"phase": phase, "variant": args.variant, "source": str(source), "models": str(output), **extra}
        temporary = status_path.with_suffix(".json.part")
        temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
        temporary.replace(status_path)
        print(json.dumps(data), flush=True)
    def execute(script, *options):
        subprocess.run([sys.executable, str(root / "scripts" / script), *map(str, options)], cwd=root, check=True)
    try:
        status("downloading")
        preview = source / f"preview_{args.variant}/diffusion_pytorch_model.safetensors"
        if args.wait_for_preview:
            while not preview.exists():
                time.sleep(5)
        else:
            execute("download_models.py", "--output", source, "--variant", args.variant)
        manifest_path = output / f"prism_{args.variant}_conversion.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {"components": {}}
        all_components = ("video_dit", "video_dit_2", "audio_dit", "dual_tower_bridge", "text_encoder", "video_vae", "audio_vae")
        missing = [kind for kind in all_components if kind not in manifest["components"] or not (output / manifest["components"][kind]["file"]).exists()]
        if missing:
            status("converting", components=missing)
            execute("convert_models.py", "--base", source / "pretrained_models/MOVA-360p", "--preview", preview,
                    "--output", output, "--variant", args.variant, "--device", "cuda:0", "--components", *missing)
        reused = False
        if not args.skip_smoke:
            validation_folder = root / "outputs" / f"{args.variant}_validation"
            validation_path = validation_folder / "validation.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            fingerprint = {kind: {"file": entry["file"], "bytes": (output / entry["file"]).stat().st_size,
                                  "mtime_ns": (output / entry["file"]).stat().st_mtime_ns}
                           for kind, entry in manifest["components"].items()}
            if validation_path.exists() and (validation_folder / "smoke.mp4").exists():
                try:
                    previous = json.loads(validation_path.read_text(encoding="utf-8"))
                    reused = (previous.get("finite") is True and previous.get("component_files") == fingerprint
                              and previous.get("bundle_id") == manifest["bundle_id"])
                except (ValueError, OSError):
                    pass
            status("smoke_test", reused=reused)
            if not reused:
                if (validation_folder / "smoke.mp4").exists():
                    validation_folder = validation_folder.with_name(validation_folder.name + "_" + str(time.time_ns()))
                execute("validate_models.py", "--models", output, "--variant", args.variant,
                        "--output", validation_folder)
        status("complete", smoke_test=not args.skip_smoke, smoke_reused=reused, quality_acceptance=False)
    except BaseException as error:
        status("failed", error=str(error))
        raise


if __name__ == "__main__":
    main()
