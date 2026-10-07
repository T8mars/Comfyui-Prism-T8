"""Bounded, private inference cache built from standalone ConvRot components.

The public inputs remain seven independent safetensors and two independent
LoRAs. This cache is disposable; it never invokes a Diffusers model loader.
"""
import hashlib
import json
from pathlib import Path

import torch
from filelock import FileLock
from safetensors.torch import save_file

from ..format import TensorReader
from ..quantization import decode_config
from ..runtime import check_bundle
from . import FREEVIDEO_COMMIT
from .rotation import activation_mult, convert_weight


def fingerprint(parts, loras):
    paths = [parts[k].path for k in sorted(parts)] + [Path(p) for p in loras]
    rows = [(str(p.resolve()), p.stat().st_size, p.stat().st_mtime_ns) for p in paths]
    return hashlib.sha256(json.dumps([2, FREEVIDEO_COMMIT, rows]).encode()).hexdigest()[:24]


def tensors(reader, mapping):
    output, metadata, used = {}, {}, set()
    for key, target in mapping.items():
        if not key.endswith('.comfy_quant'):
            continue
        source = key[:-12]
        owner = target[:-12]
        group = decode_config(reader.get_tensor(key))['convrot_groupsize']
        weight = reader.get_tensor(source + '.weight')
        n, k = weight.shape
        bias = source + '.bias' in mapping
        output[owner + '.qweight'] = convert_weight(weight, group)
        output[owner + '.w_scale'] = reader.get_tensor(source + '.weight_scale').reshape(n).float().contiguous()
        output[owner + '.act_mult'] = activation_mult(k, group)
        if bias:
            output[owner + '.bias'] = reader.get_tensor(source + '.bias').to(torch.bfloat16).contiguous()
        metadata[owner] = dict(mode='w8a8_int8', in_features=k, out_features=n,
            rot_block=group, has_mult=True, act_asym=False, bias=bias,
            bias_dtype='bfloat16' if bias else None)
        used.update((key, source + '.weight', source + '.weight_scale', source + '.bias'))
    for key, target in mapping.items():
        if key in used:
            continue
        value = reader.get_tensor(key)
        if value.dtype == torch.int8:
            raise ValueError('Unmarked INT8 tensor: ' + key)
        if value.is_floating_point():
            value = value.to(torch.float32 if key.startswith(('time_embedding.', 'time_projection.'))
                             else torch.bfloat16)
        output[target] = value.contiguous()
    return output, metadata


def prepare(parts, loras, cache_root, progress=None):
    check_bundle(parts)
    if 'video_dit_2' not in parts:
        raise ValueError('Acceleration requires both Prism video experts')
    for kind in ('video_dit', 'video_dit_2', 'audio_dit', 'dual_tower_bridge'):
        if parts[kind].metadata.get('prism.precision') != 'int8_convrot':
            raise ValueError(f'FreeVideo acceleration requires an INT8 ConvRot {kind} component')
    # FreeVideo's published INT8 bundle keeps its shared UMT5 encoder in BF16.
    # Text is encoded separately and is never fused into this diffusion cache.
    # Accept the original standalone encoder as well as the smaller INT8 option.
    if parts['text_encoder'].metadata.get('prism.precision') not in ('bf16', 'int8_convrot'):
        raise ValueError('FreeVideo text encoder must be a standalone BF16 or INT8 ConvRot component')
    if len(loras) not in (0, 2):
        raise ValueError('Select both high and low 260412 LoRAs')
    root = Path(cache_root) / fingerprint(parts, loras)
    root.mkdir(parents=True, exist_ok=True)
    with FileLock(str(root / '.prepare.lock')):
        manifest_path = root / 'manifest.json'
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
            for name, row in manifest['files'].items():
                path = root / name
                if not path.is_file() or path.stat().st_size != row['bytes']:
                    raise ValueError(f'Incomplete acceleration cache: {path}. Remove this cache and retry.')
            return root
        files = {}
        def write(relative, values, metadata=None):
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix('.tmp')
            save_file(values, str(temporary), metadata={'prism_qlinear': json.dumps(metadata or {})})
            temporary.replace(path)
            checksum = hashlib.sha256()
            with path.open('rb') as stream:
                while block := stream.read(8 * 2**20):
                    checksum.update(block)
            digest = checksum.hexdigest()
            files[relative] = dict(bytes=path.stat().st_size, sha256=digest)
            if progress:
                progress(relative)
        roots = {}
        folders = {'video_dit': 'expert_high/blocks', 'video_dit_2': 'expert_low/blocks',
                   'audio_dit': 'audio/blocks', 'dual_tower_bridge': 'bridge'}
        for kind, folder in folders.items():
            component = parts[kind]
            cfg = root / 'configs' / kind / 'config.json'
            cfg.parent.mkdir(parents=True, exist_ok=True)
            cfg.write_text(json.dumps(component.config), encoding='utf-8')
            with TensorReader(component.path) as reader:
                groups, root_map = {}, {}
                for key in reader.keys():
                    if key.startswith('blocks.'):
                        _, number, tail = key.split('.', 2)
                        tag = f'{int(number):02d}'
                        role = 'audio' if kind == 'audio_dit' else 'video'
                        groups.setdefault(tag, {})[key] = f'{tag}.{role}.{tail}'
                    elif kind == 'dual_tower_bridge':
                        first, number, tail = key.split('.', 2)
                        role = {'audio_to_video_conditioners': 'a2v', 'video_to_audio_conditioners': 'v2a'}[first]
                        tag = f'{int(number):02d}'
                        groups.setdefault(tag, {})[key] = f'{tag}.{role}.{tail}'
                    else:
                        root_map[key] = kind + '.' + key
                for tag, mapping in sorted(groups.items()):
                    values, metadata = tensors(reader, mapping)
                    write(folder + '/' + tag + '.safetensors', values, metadata)
                    del values
                values, metadata = tensors(reader, root_map)
                if metadata:
                    raise ValueError('Quantized roots need an explicit root adapter')
                roots.update(values)
        write('root.safetensors', roots)
        del roots
        from .vendor.prism_prepare import lora_groups, lora_block_tensors, lora_root_tensors
        for expert, path in zip(('high', 'low'), loras):
            groups = lora_groups(path)
            if not groups:
                raise ValueError('Not a supported LightX2V LoRA: ' + str(path))
            for index in range(parts['video_dit'].config['num_layers']):
                values = lora_block_tensors(groups, index, f'{index:02d}')
                if not values:
                    raise ValueError(f'LoRA missing block {index}')
                write(f'lora_{expert}/blocks/{index:02d}.safetensors', values)
            write(f'lora_{expert}/root.safetensors', lora_root_tensors(groups))
            del groups
        scheduler = root / 'configs/scheduler/scheduler_config.json'
        scheduler.parent.mkdir(parents=True, exist_ok=True)
        scheduler.write_text(parts['dual_tower_bridge'].metadata['prism.scheduler_config'], encoding='utf-8')
        boundary = float(parts['dual_tower_bridge'].metadata['prism.boundary_ratio'])
        (root / 'configs/model_index.json').write_text(json.dumps({'boundary_ratio': boundary}), encoding='utf-8')
        manifest = dict(format='freevideo-prism-prepared', version=1, variant='int8',
            linear_weights='w8a8_int8', layout=dict(video_blocks=parts['video_dit'].config['num_layers'],
            fused_blocks=parts['audio_dit'].config['num_layers']), files=files,
            distill={'kind': 'lora'} if loras else None, source_commit=FREEVIDEO_COMMIT,
            bundle_id=parts['video_dit'].metadata['prism.bundle_id'], rotation='lossless-convrot-PD')
        if fingerprint(parts, loras) != root.name:
            raise ValueError('An input model changed during acceleration preparation; retry with stable files')
        temporary = manifest_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
        temporary.replace(manifest_path)
    return root
