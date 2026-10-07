"""Prepare the private streaming cache without using CUDA."""
import argparse
import json
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from prism.format import Component
from prism.acceleration.cache import prepare

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', type=Path, default=ROOT / 'models/standalone')
    parser.add_argument('--manifest', type=Path, help='Standalone conversion manifest; defaults to prism_alpha_conversion.json')
    parser.add_argument('--text-encoder', type=Path, help='Explicit standalone BF16 or INT8 T5 override')
    parser.add_argument('--cache-root', type=Path, default=ROOT / 'models/.prism-acceleration-cache')
    args = parser.parse_args()
    manifest = args.manifest or args.models / 'prism_alpha_conversion.json'
    converted = json.loads(manifest.read_text(encoding='utf-8'))
    # Both BF16 and INT8 T5 can exist in this folder. Never let glob order
    # silently choose the model, or combine components from different variants.
    parts = {kind: Component.inspect(args.models / row['file'], kind)
             for kind, row in converted['components'].items()}
    if args.text_encoder:
        parts['text_encoder'] = Component.inspect(args.text_encoder, 'text_encoder')
    loras = []
    for expert in ('HIGH', 'LOW'):
        paths = list((ROOT / 'models/loras').glob(f'*{expert}*260412*rank_256*.safetensors'))
        if len(paths) != 1:
            raise ValueError(f'Select exactly one 260412 {expert} rank-256 LoRA; found {len(paths)}')
        loras.append(paths[0])
    torch.set_num_threads(4)
    cache = prepare(parts, loras, args.cache_root, progress=lambda name: print(name, flush=True))
    print(json.dumps({'cache': str(cache), 'text_encoder': str(parts['text_encoder'].path),
                      'text_precision': parts['text_encoder'].metadata['prism.precision']}), flush=True)


if __name__ == '__main__':
    main()
