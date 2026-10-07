"""Compact installation storage; never remove borrowed models or user outputs."""
import json
import os
from pathlib import Path
import stat
import time

from .monitoring import save


def conversion_source(row):
    name = row['file']
    return name.endswith('.safetensors') and name.startswith(('h3-base/transformer/', 'stage-dmd-step-250/'))


def fingerprint(path):
    info = path.stat()
    result = dict(bytes=info.st_size, mtime_ns=info.st_mtime_ns, ctime_ns=info.st_ctime_ns,
                  inode=info.st_ino, device=info.st_dev)
    if os.name == 'nt':
        from .win32 import file_change_time_ns
        result['change_time_ns'] = file_change_time_ns(path)
    return result


def managed_model(root, path):
    """Only regular, unshared files beneath this installation's models directory."""
    root, path = Path(root).resolve(), Path(path).absolute()
    try:
        path.relative_to(root / 'models')
        if path.resolve() != path:
            return False
        current = path
        while current != root:
            info = current.lstat()
            if stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 0x400:
                return False
            current = current.parent
        info = path.stat()
        return stat.S_ISREG(info.st_mode) and info.st_nlink == 1
    except (OSError, ValueError):
        return False


def ownership(root):
    path = Path(root) / '.freevideo' / 'downloaded-models.json'
    if not path.exists():
        return path, {}
    value = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(value, dict) or value.get('root') != str(Path(root).resolve()) or not isinstance(value.get('files'), dict):
        raise ValueError('Invalid model ownership record; models retained')
    return path, value['files']


def record_download(root, path, row, *, origin='download'):
    # Called only after a previously absent target was downloaded/copied and verified.
    if not conversion_source(row) or not managed_model(root, path):
        return
    ledger, files = ownership(root)
    files[str(Path(path).absolute())] = dict(fingerprint(path), sha256=row.get('sha256'),
                                           git_blob=row.get('git_blob'), source_file=row['file'], origin=origin)
    save(ledger, {'root': str(Path(root).resolve()), 'files': files})


def compact_sources(plan, rows, report):
    """Caller must first verify the complete FP8 cache and pass GPU setup probes.

    Deletion uses an allowlist of pinned conversion-only filenames and a matching
    download receipt. External directories, preexisting/shared/replaced files,
    decoder/encoder weights, configs, failed attempts and outputs are retained.
    """
    from .network import hash_file
    root, model_dir = Path(plan['root']).resolve(), Path(plan['model_dir']).resolve()
    result = {'mode': plan.get('storage', 'compact'), 'deleted': [], 'retained': [],
              'released_bytes': 0, 'started_epoch': time.time()}
    if result['mode'] != 'compact':
        save(report, result)
        return result
    try:
        ledger, files = ownership(root)
    except (OSError, ValueError, TypeError) as error:
        result['retained'].append({'reason': str(error)})
        save(report, result)
        return result
    # Record intent before modifying any file; update after each successful unlink.
    save(report, result)
    for row in rows:
        if not conversion_source(row):
            continue
        path = model_dir / row['file']
        if not path.exists():
            continue
        expected = files.get(str(path), {})
        current = fingerprint(path)
        if (not isinstance(expected, dict) or not managed_model(root, path)
                or 'change_time_ns' in current and current['change_time_ns'] is None
                or expected.get('source_file') != row['file'] or expected.get('bytes') != row['bytes']
                or expected.get('sha256') != row.get('sha256') or expected.get('git_blob') != row.get('git_blob')
                or any(expected.get(k) != v for k, v in current.items())):
            result['retained'].append({'file': str(path), 'reason': 'Not an unchanged, unshared file downloaded by this installation'})
            continue
        try:
            # Destructive cleanup cannot trust a metadata cache hit. Equal-size
            # edits may share a filesystem timestamp tick with the receipt.
            if row.get('sha256'):
                matches = hash_file(path, discard_cache=True) == row['sha256']
            elif row.get('git_blob'):
                matches = hash_file(path, 'sha1', git_blob=True, discard_cache=True) == row['git_blob']
            else:
                raise OSError('No pinned content hash; source retained')
            if not matches:
                raise OSError('Source content changed; retained without deletion')
            result['pending'] = {'file': str(path), 'bytes': row['bytes']}
            save(report, result)
            # Hashing can take time for model shards. Recheck both identity and
            # sharing immediately before unlink, after saving deletion intent.
            if fingerprint(path) != current or not managed_model(root, path):
                raise OSError('File changed or became shared during cleanup')
            path.unlink()
        except OSError as error:
            result['retained'].append({'file': str(path), 'reason': str(error)})
            result['pending'] = None
            save(report, result)
            continue
        result['pending'] = None
        result['deleted'].append({'file': str(path), 'bytes': row['bytes'], 'source_file': row['file']})
        result['released_bytes'] += row['bytes']
        files.pop(str(path), None)
        save(report, result)
        save(ledger, {'root': str(root), 'files': files})
    result['completed_epoch'] = time.time()
    save(report, result)
    return result
