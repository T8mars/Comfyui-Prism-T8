"""Export verified legacy/v2 AdaLN tables and FP8 weights as a portable model.

This is an explicit model publication operation. It never runs inference, casts
weights, deletes originals, or uses the producer GPU as a compatibility key.
"""
import argparse
import json
import os
from pathlib import Path
import shutil

from . import adaln_assets as assets
from .monitoring import save

# Explicit legacy contract migration, not a live-code hash in the v2 key.
LEGACY_CONTRACTS = {'ddfb84aa2970cd8169b22c18f2005e2098a1d7273ff257e92d49566de9141706'}
LEGACY_MODEL_CONTRACTS = {'0f94a5583d5f3f16f9e2d10f0955eb2f55a92a0b6501f339d2b6445f38614841'}


def embedding_hash(path):
    """Hash only time-embedding tensors, in bounded reads, without Torch."""
    import hashlib
    import struct
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        length = struct.unpack('<Q', stream.read(8))[0]
        if length > 16 * 1024**2:
            raise ValueError('Invalid root tensor header')
        header = json.loads(stream.read(length))
        keys = sorted(k for k in header if k.startswith(('time_embedder.', 'time_proj.')))
        if not keys:
            raise ValueError('No time embedding tensors in model root')
        for key in keys:
            row = header[key]
            h.update(json.dumps([key, row['dtype'], row['shape']], separators=(',', ':')).encode())
            start, stop = row['data_offsets']
            stream.seek(8 + length + start)
            remaining = stop - start
            while remaining:
                block = stream.read(min(remaining, 8 * 1024**2))
                if not block:
                    raise ValueError('Truncated time embedding weight')
                h.update(block)
                remaining -= len(block)
    return h.hexdigest()


def share(source, destination, sha256):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if assets.file_hash(destination) != sha256:
            raise ValueError('Export destination changed; retained: ' + str(destination))
        return
    try:
        os.link(source, destination)
    except OSError:
        if shutil.disk_usage(destination.parent).free < source.stat().st_size + 1024**3:
            raise ValueError('Insufficient disk space for export copy')
        temporary = destination.with_suffix('.partial')
        if temporary.exists():
            raise ValueError('Interrupted export retained: ' + str(temporary))
        shutil.copyfile(source, temporary)
        if assets.file_hash(temporary) != sha256:
            raise ValueError('Export copy failed verification')
        temporary.replace(destination)


def export(cache, output, tables, *, channels=5376, download=None):
    from .adaln import schedule_timesteps
    cache, output = Path(cache).resolve(), Path(output).resolve()
    if output == cache or cache in output.parents:
        raise ValueError('Export to a separate directory; originals are never modified')
    original = json.loads((cache / 'manifest.json').read_text(encoding='utf-8'))
    groups = [r for r in original['groups'] if not r['group'].startswith('adaln/')]
    count = len(groups) - 1
    if {r['group'] for r in groups} != {'root'} | {'blocks/%02d' % i for i in range(count)}:
        raise ValueError('Export needs every transformer block and one root group')
    sources = dict(embedding_sha256=embedding_hash(cache / 'root.safetensors'),
                   groups=assets.projection_groups(original))
    if download:
        sources['download'] = download
    manifest = dict(original, format=assets.SLIM_FORMAT, groups=groups, adaln_sources=sources,
                    adaln_tables=[], modification='Original AdaLN projections omitted; fixed portable modulation constants included. FP8 tensor bytes unchanged.')
    weights = assets.weight_identity(manifest)
    for table_path in tables:
        table_path = Path(table_path).resolve()
        producer = json.loads((table_path / 'identity.json').read_text(encoding='utf-8'))
        if producer.get('format') == assets.FORMAT:
            value = producer
            if value['weights'] != weights:
                raise ValueError('Portable table belongs to different modulation weights')
            provenance = json.loads((table_path / 'producer.json').read_text(encoding='utf-8'))
        else:
            if (producer.get('implementation') not in LEGACY_CONTRACTS
                    or producer.get('original_implementation') not in LEGACY_MODEL_CONTRACTS
                    or producer.get('source_id') != original['source_id']
                    or producer.get('video_shift') != 12. or producer.get('audio_shift') != 3.):
                raise ValueError('Legacy table is not a verified export contract')
            from src.inference.render import KEYFRAME_NOISE_AUG
            if producer.get('task') != 't2va' and producer.get('keyframe_noise_aug') != KEYFRAME_NOISE_AUG:
                raise ValueError('Legacy table keyframe schedule differs')
            times = schedule_timesteps(producer['steps'], device='cpu', task=producer['task'])
            value = assets.identity(weights, [t.tolist() for t in times], channels)
            provenance = dict(producer, migrated_from='legacy-device-cache-v1')
        name = assets.directory(value)
        previous = next((row for row in manifest['adaln_tables'] if row['directory'] == name), None)
        records = []
        for index in range(count):
            path = table_path / ('%02d.safetensors' % index)
            receipt = json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
            row = dict(index=index, file=name + '/' + path.name,
                       bytes=path.stat().st_size, sha256=receipt['sha256'])
            assets.check_table(path, row, value)
            if previous:
                # Equal schedules may deduplicate only if the actual tensor
                # values are equal, never on a similarity threshold.
                from safetensors.torch import load_file
                import torch
                left, right = load_file(path), load_file(output / row['file'])
                if any(not torch.equal(left[k], right[k]) for k in left):
                    raise ValueError('Conflicting fixed tables for the same schedule')
                del left, right
                continue
            share(path, output / row['file'], row['sha256'])
            save((output / row['file']).with_suffix('.json'), dict(bytes=row['bytes'], sha256=row['sha256']))
            records.append(row)
        if not previous:
            save(output / name / 'identity.json', value)
            save(output / name / 'producer.json', provenance)
            manifest['adaln_tables'].append(dict(identity=value, directory=name, files=records, producer=provenance))
    assets.validate_catalog(manifest, count)
    # Hash every source before sharing its immutable inode. The export never
    # credits timestamps as proof of a user-supplied model's content.
    for row in groups:
        source = assets.asset_path(cache, row['file'])
        if source.stat().st_size != row['bytes'] or assets.file_hash(source) != row['sha256']:
            raise ValueError('FP8 export source changed: ' + row['file'])
        share(source, output / row['file'], row['sha256'])
        print(json.dumps({'event': 'export_slim_group', 'group': row['group']}), flush=True)
    manifest['total_bytes'] = sum(r['bytes'] for r in groups) + sum(
        r['bytes'] for t in manifest['adaln_tables'] for r in t['files'])
    manifest['omitted_projection_bytes'] = sum(r['bytes'] for r in sources['groups'])
    save(output / 'manifest.json', manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--tables', required=True, type=Path, nargs='+')
    parser.add_argument('--original-repo')
    parser.add_argument('--original-revision')
    args = parser.parse_args()
    from .paths import add_vdn
    add_vdn()
    download = None
    if args.original_repo or args.original_revision:
        if not args.original_repo or not args.original_revision:
            parser.error('Both original source repo and immutable revision are required')
        download = dict(repo=args.original_repo, revision=args.original_revision, prefix='cache')
    result = export(args.cache, args.out, args.tables, download=download)
    print(json.dumps({'bytes': result['total_bytes'], 'omitted_bytes': result['omitted_projection_bytes']}))


if __name__ == '__main__':
    main()
