"""Metadata-only resume checks for prepared model groups; no Torch import."""
import hashlib
import json
from pathlib import Path

from .network import hash_file


# An audited, byte-identical I/O migration, pinned at BOTH ends. Editing either
# loader invalidates this permission until its arithmetic is checked again.
# Every other identity field (source, upstream quantizer, Torch, format) matches
# exactly. The source marker retains the original producer's truthful identity.
AUDITED_FP8_IMPLEMENTATION = 'b60f9c865fbc8725bd3389e37fed58972b0557879dbb51bf537b05e7f07318b2'
AUDITED_WEIGHTS_IMPLEMENTATION = 'fb818cff05ec5d7e87a07ffba9998156e4cd6c7ca2f7ee7db0c2d749685d34e1'
COMPATIBLE_FP8_IMPLEMENTATIONS = ('9d5f0fce6eb100b50bd5b669f96cb45677d4141235e66492dbf6a962853165ee',)
COMPATIBLE_WEIGHTS_IMPLEMENTATIONS = ('0e9d79b94265c99cb144fe0d57f46e95c4befca3d7430579a0cc0bc424565e19',)


def source_paths(base, checkpoint):
    return sorted(set((base / 'transformer').glob('*.safetensors')) |
                  {checkpoint / 'linear_branch/model.safetensors'} |
                  set((checkpoint / 'adapters').glob('*/adapter_model.safetensors')))


def expected_groups(base):
    count = json.loads((base / 'transformer/config.json').read_text(encoding='utf-8'))['num_layers']
    if type(count) is not int or count < 1:
        raise ValueError('Invalid transformer layer count')
    return sorted(['root'] + [kind + '/%02d' % i for kind in ('adaln', 'blocks') for i in range(count)])


def cache_location(root, prefix, identity):
    identities = [identity]
    if identity['implementation_sha256'] == AUDITED_FP8_IMPLEMENTATION:
        if (prefix == 'vdn-fp8-streamed-'
                and identity.get('merge_implementation_sha256') == AUDITED_WEIGHTS_IMPLEMENTATION):
            identities += [dict(identity, implementation_sha256=fp8, merge_implementation_sha256=weights)
                           for fp8, weights in zip(COMPATIBLE_FP8_IMPLEMENTATIONS, COMPATIBLE_WEIGHTS_IMPLEMENTATIONS)]
        elif prefix == 'vdn-fp8-':
            identities += [dict(identity, implementation_sha256=old) for old in COMPATIBLE_FP8_IMPLEMENTATIONS]
    elif prefix == 'vdn-bf16-' and identity['implementation_sha256'] == AUDITED_WEIGHTS_IMPLEMENTATION:
        identities += [dict(identity, implementation_sha256=old) for old in COMPATIBLE_WEIGHTS_IMPLEMENTATIONS]
    candidates = []
    for selected in identities:
        key = hashlib.sha256(json.dumps(selected, sort_keys=True).encode()).hexdigest()
        output = Path(root) / (prefix + key[:16])
        candidates.append((output, selected, key))
        marker = output / 'source.json'
        if marker.is_file():
            try:
                if json.loads(marker.read_text(encoding='utf-8')) == selected:
                    return output, selected, key
            except (ValueError, UnicodeError):
                pass
        if output.exists() and selected is identity:
            # Never work around a conflicting current identity by selecting an
            # older directory; the caller must report that conflict.
            return output, selected, key
    return candidates[0]


def reusable_group(output, group):
    """Hash bounded chunks and inspect headers, never deserialize tensor data."""
    destination = output / (group + '.safetensors')
    progress = destination.with_suffix('.json')
    if not destination.is_file() or not progress.is_file():
        return None
    from safetensors import SafetensorError, safe_open
    try:
        saved = json.loads(progress.read_text(encoding='utf-8'))
        if (not isinstance(saved, dict) or saved.get('group') != group
                or saved.get('file') != group + '.safetensors'
                or not isinstance(saved.get('linears'), dict)
                or destination.stat().st_size != saved.get('bytes')
                or hash_file(destination, discard_cache=True) != saved.get('sha256')):
            return None
        # NumPy framework keeps this path independent of Torch and its storage
        # constructors. get_slice().get_shape() reads metadata only, also for FP8.
        with safe_open(destination, framework='np') as stream:
            keys = set(stream.keys())
            if saved.get('tensors') != len(keys):
                return None
            prefixes = {name.removesuffix('.weight_fp8') for name in keys if name.endswith('.weight_fp8')}
            if set(saved['linears']) != prefixes:
                return None
            for name, spec in saved['linears'].items():
                if (not isinstance(spec, dict) or spec.get('input_dtype') not in ('bfloat16', 'float32')
                        or spec.get('weight_shape') != stream.get_slice(name + '.weight_fp8').get_shape()
                        or spec.get('scale_shape') != stream.get_slice(name + '.weight_scale').get_shape()
                        or stream.get_slice(name + '.weight_fp8').get_dtype() != 'F8_E4M3'
                        or stream.get_slice(name + '.weight_scale').get_dtype() != 'F32'):
                    return None
        return saved
    except (OSError, ValueError, TypeError, KeyError, SafetensorError):
        return None
