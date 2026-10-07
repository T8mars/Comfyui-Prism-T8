"""Exclude setup, tests and GPU work; children retain their parent's lease.

GPU work is device-scoped, so it takes the machine lease shared and its own
device lease exclusively. Setup still takes the machine lease exclusively and
therefore still excludes every device. Without this split one measurement
occupied an entire multi-GPU host.
"""
from contextlib import contextmanager
import os
from pathlib import Path
import re
import sys


from .paths import data_root
from .system import windows
from .processes import inherited_fd
LOCK_PATH = Path(os.environ.get('FREEVIDEO_LOCK_PATH', data_root() / 'engine.lock'))
LOCK_ENV = 'FREEVIDEO_RUNTIME_LOCK_FD'


def lock_holders(paths):
    """Read-only Linux diagnostics, including orphans retaining inherited FDs."""
    if windows() or not Path('/proc/self/fd').is_dir():
        return []
    identities = set()
    for path in paths:
        try:
            item = Path(path).stat()
            identities.add((item.st_dev, item.st_ino))
        except OSError:
            pass
    holders = []
    for directory in Path('/proc').iterdir():
        if not directory.name.isdigit():
            continue
        try:
            for fd in (directory / 'fd').iterdir():
                try:
                    item = fd.stat()
                    if (item.st_dev, item.st_ino) in identities:
                        holders.append(int(directory.name))
                        break
                except OSError:
                    continue
        except OSError:
            continue
    return sorted(holders)


def device_lock_path(device, path=None):
    """The lease for one device, beside the machine lease it shares a root with."""
    base = Path(path or os.environ.get('FREEVIDEO_LOCK_PATH', data_root() / 'engine.lock'))
    token = re.sub(r'[^A-Za-z0-9-]', '_', str(device))[:64] or 'unknown'
    return base.with_name(base.stem + '-gpu-' + token + base.suffix)


@contextmanager
def runtime_lock(path=None, *, inherit=True, shared=False):
    path = Path(path or os.environ.get('FREEVIDEO_LOCK_PATH', data_root() / 'engine.lock'))
    path.parent.mkdir(parents=True, exist_ok=True)
    inherited = inherited_fd() if inherit else None
    if inherited is None:
        stream = path.open('a+b')
    else:
        descriptor = int(inherited)
        actual, expected = os.fstat(descriptor), path.stat()
        if (actual.st_dev, actual.st_ino) != (expected.st_dev, expected.st_ino):
            raise ValueError('Inherited runtime lock descriptor does not match the GPU-work lock')
        stream = os.fdopen(os.dup(descriptor), 'a+b')
    try:
        # An inherited descriptor shares the parent's open-file description.
        # Close our descriptor on exit; do not unlock the parent's live work.
        if windows():
            if inherited is None:
                import msvcrt
                if path.stat().st_size == 0:
                    stream.write(b'\0')
                    stream.flush()
                stream.seek(0)
                try:
                    msvcrt.locking(stream.fileno(),
                                   msvcrt.LK_NBRLCK if shared else msvcrt.LK_NBLCK, 1)
                except OSError as error:
                    raise BlockingIOError('Another process holds the installation lease') from error
            # The parent holds the byte-range lock; its Job Object owns this
            # worker tree. Closing a duplicated child handle must not unlock it.
        else:
            import fcntl
            fcntl.flock(stream, (fcntl.LOCK_SH if shared else fcntl.LOCK_EX) | fcntl.LOCK_NB)
        yield stream.fileno()
    finally:
        # Releasing the lease must not discard completed work. Taking it still
        # fails loudly, but once the request has run, a close that fails has
        # nothing left to protect: the kernel drops the descriptor when this
        # process exits, and the lease is advisory, so the next holder waits on
        # the file rather than on this handle. Observed once in 120 concurrent
        # eight-GPU measurements on a WekaFS share, as OSError EBADFD raised by
        # the close itself after a full video had been sampled, decoded and
        # written; the exit code threw that video away.
        try:
            stream.close()
        except OSError as error:
            print('Could not release the runtime lease at %s: %s. The request is '
                  'unaffected; the descriptor is dropped when this process exits.'
                  % (path, error), file=sys.stderr, flush=True)
