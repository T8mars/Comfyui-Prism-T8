"""Fetch pinned official weights needed for standalone conversion (skip base DiTs)."""
import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

REPO = "FrancisRing/Prism"
REVISION = "347659c562dcc392c45dfe1673c051e7c62f57ac"
BASE = "pretrained_models/MOVA-360p"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="checkpoints/official")
    parser.add_argument("--variant", choices=("alpha", "beta", "both"), default="alpha")
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    variants = ("alpha", "beta") if args.variant == "both" else (args.variant,)
    info = HfApi().model_info(REPO, revision=REVISION, files_metadata=True)
    selected = []
    for file in info.siblings:
        name = file.rfilename
        config = name.startswith(BASE + "/") and (name.endswith(".json") or "/tokenizer/" in name)
        preview = any(name == f"preview_{variant}/diffusion_pytorch_model.safetensors" for variant in variants)
        frozen = any(name.startswith(f"{BASE}/{kind}/") and name.endswith(".safetensors")
                     for kind in ("text_encoder", "video_vae", "audio_vae"))
        if config or (not args.metadata_only and (preview or frozen)):
            selected.append(name)
    total = sum(file.size or 0 for file in info.siblings if file.rfilename in selected)
    print(json.dumps({"repository": REPO, "revision": REVISION, "bytes": total,
                      "gib": round(total / 2 ** 30, 2), "files": selected}, indent=2), flush=True)
    if args.dry_run:
        return
    folder = snapshot_download(REPO, revision=REVISION, allow_patterns=selected,
                               local_dir=args.output, max_workers=3)
    (Path(folder) / "prism_download.json").write_text(json.dumps({"repo": REPO, "revision": REVISION,
        "files": selected, "bytes": total}, indent=2), encoding="utf-8")
    print(f"Downloaded official sources to {folder}", flush=True)


if __name__ == "__main__":
    main()
