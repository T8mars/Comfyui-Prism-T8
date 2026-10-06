"""Convert local official Prism weights; no Diffusers-format output is produced."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prism.conversion import convert_bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="Official MOVA-360p folder (configs, text encoder, VAEs and tokenizer)")
    parser.add_argument("--preview", help="Official alpha or beta fused checkpoint (required for transformer components)")
    parser.add_argument("--output", required=True, help="Directory for seven independent component files")
    parser.add_argument("--variant", choices=("alpha", "beta"), default="alpha")
    parser.add_argument("--precision", choices=("int8_convrot", "bf16"), default="int8_convrot")
    parser.add_argument("--device", default="cpu", help="cpu or cuda:0; conversion intermediates are row-chunked")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--mseclip", action="store_true", help="Experimental optimal clipping; 80x extra quantization work")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--components", nargs="+", help="Convert selected components, merging their manifest sequentially")
    args = parser.parse_args()
    import torch
    torch.set_num_threads(args.cpu_threads)
    convert_bundle(args.base, args.preview, args.output, variant=args.variant,
                   precision=args.precision, mseclip=args.mseclip, device=args.device,
                   overwrite=args.overwrite, dry_run=args.dry_run, components=args.components)


if __name__ == "__main__":
    main()
