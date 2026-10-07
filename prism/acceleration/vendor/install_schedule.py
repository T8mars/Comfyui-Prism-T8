"""Run ready installation tasks with bounded workers and exclusive writers."""
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED, CancelledError


def run(tasks, cancel, *, workers=3, progress=None):
    names = set(tasks)
    for name, task in tasks.items():
        if not set(task.get('after', ())).issubset(names):
            raise ValueError('Unknown installation prerequisite: ' + name)
    pending = set(names)
    # Validate the whole graph before doing any installation writes.
    while pending:
        ready = {name for name in pending if not set(tasks[name].get('after', ())) & pending}
        if not ready:
            raise ValueError('Cyclic installation prerequisites')
        pending -= ready
    pending, active, results, held = dict(tasks), {}, {}, set()
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix='install') as pool:
        try:
            while pending or active:
                if cancel.is_set():
                    raise CancelledError('Installation cancelled; completed files retained')
                for name, task in list(pending.items()):
                    resources = set(task.get('writes', ()))
                    if (len(active) == workers or not set(task.get('after', ())).issubset(results)
                            or resources & held):
                        continue
                    held.update(resources)
                    del pending[name]
                    if progress:
                        progress(name, 'running', len(results), len(tasks))
                    active[pool.submit(task['run'])] = (name, resources)
                done, _ = wait(active, timeout=.1, return_when=FIRST_COMPLETED)
                for future in done:
                    name, resources = active.pop(future)
                    results[name] = future.result()
                    held -= resources
                    if progress:
                        progress(name, 'complete', len(results), len(tasks))
        except BaseException:
            cancel.set()
            for future in active:
                future.cancel()
            raise
    return results
