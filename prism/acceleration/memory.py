"""Windows streamed weights have reclaimable mapped pages in their RSS."""
import ctypes
import os

import psutil

GIB = 2**30
WINDOWS = os.name == 'nt'


def snapshot(process):
    memory = psutil.virtual_memory()
    rss = process.memory_info().rss
    row = dict(available_bytes=memory.available, total_bytes=memory.total,
               rss_bytes=rss, guard_bytes=rss, guard_metric='RSS', reclaimable_mapped_bytes=0)
    if WINDOWS:
        from .vendor.win32 import memory_status, process_memory
        try:
            system = memory_status()
            current = process_memory(process.pid)
            private = current['private_working_set_bytes']
            row.update(available_bytes=system['available_bytes'], total_bytes=system['total_bytes'],
                       rss_bytes=current['rss_bytes'], guard_bytes=private,
                       guard_metric='private working set',
                       reclaimable_mapped_bytes=max(0, current['rss_bytes'] - private))
        except OSError:
            # Keep the conservative RSS guard when the OS cannot report it.
            pass
    return row


def trim_current_working_set():
    """Release this process's pageable residency; never touch another process."""
    if not WINDOWS:
        return False
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    api = ctypes.WinDLL('psapi', use_last_error=True)
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    kernel.GetCurrentProcess.argtypes = []
    api.EmptyWorkingSet.restype = ctypes.c_int
    api.EmptyWorkingSet.argtypes = [ctypes.c_void_p]
    return bool(api.EmptyWorkingSet(kernel.GetCurrentProcess()))


def check(process, ram_gib, emit):
    row = snapshot(process)
    # Reclaim our own mapped model pages before declaring a memory emergency.
    # Locked staging/KV buffers remain locked; weights and RNG are unchanged.
    if WINDOWS and row['available_bytes'] < 8 * GIB and row['reclaimable_mapped_bytes'] > GIB:
        before = row
        trimmed = trim_current_working_set()
        row = snapshot(process)
        emit(event='working_set_reclaim', succeeded=trimmed, before=before, after=row)
    if row['available_bytes'] < 6 * GIB:
        emit(event='memory_stop', **row)
        raise RuntimeError('Prism stopped: less than 6 GiB system RAM available')
    if row['guard_bytes'] > (ram_gib + 6) * GIB:
        emit(event='memory_stop', **row)
        raise RuntimeError('Prism worker exceeded its host-memory budget')
    return row
