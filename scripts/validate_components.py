"""Validate available real components without requiring the whole transformer."""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from prism.format import Component, load_tokenizer
from prism.loading import load_component


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", required=True)
    parser.add_argument("--variant", default="alpha")
    parser.add_argument("--output", default="outputs/component-validation.json")
    args = parser.parse_args()
    torch.set_num_threads(4)
    folder = Path(args.models)
    manifest = json.loads((folder / f"prism_{args.variant}_conversion.json").read_text(encoding="utf-8"))
    results = {}
    for kind in ("text_encoder", "video_vae", "audio_vae"):
        if kind not in manifest["components"]:
            continue
        start = time.monotonic()
        component = Component.inspect(folder / manifest["components"][kind]["file"], kind)
        model = load_component(component, dtype=torch.float32 if kind == "audio_vae" else torch.bfloat16).cuda()
        with torch.inference_mode():
            if kind == "text_encoder":
                tokenizer = load_tokenizer(component)
                inputs = tokenizer("A surfer on a wave. <sfx>Ocean sounds.</sfx>", padding="max_length", max_length=512,
                                   truncation=True, return_tensors="pt").to("cuda")
                output = model(**inputs).last_hidden_state
            elif kind == "video_vae":
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    latent = model.encode(torch.zeros(1, 3, 5, 64, 64, device="cuda", dtype=torch.bfloat16)).latent_dist.mode()
                    output = model.decode(latent).sample
            else:
                output = model.decode(torch.zeros(1, model.latent_dim, 2, device="cuda"))
        if not torch.isfinite(output).all():
            raise RuntimeError(f"{kind} generated non-finite output")
        results[kind] = {"shape": list(output.shape), "finite": True, "seconds": time.monotonic() - start,
                         "quantized_linears": manifest["components"][kind]["quantized_linears"]}
        print(json.dumps({kind: results[kind]}), flush=True)
        model.cpu()
        del model, output
        torch.cuda.empty_cache()
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
