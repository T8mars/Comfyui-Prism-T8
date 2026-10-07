"""Windows allocator admission; no Torch import or device probing at import time."""
import math
import time
from .system import windows

_trial_allocator_owner = None


def owned_pool_capacity(torch):
    """Read the idle owner's pool capacity before policy subtracts its reserve.

    A fresh Windows CUDA context's free-memory reading cannot tell us how much
    of the resident worker's pool has already been included by the driver.
    Use the same owner/context and bounds as allocator admission instead.
    """
    total = torch.cuda.get_device_properties(0).total_memory
    free, _ = torch.cuda.mem_get_info()
    reserved = torch.cuda.memory_reserved()
    local, reader = None, None
    try:
        from .windows_gpu_memory import AdapterMemory
        reader = AdapterMemory()
        local = reader.sample().get('local')
    except (OSError, RuntimeError, AttributeError, ValueError):
        pass
    finally:
        if reader is not None:
            try:
                reader.close()
            except (OSError, RuntimeError):
                pass  # A failed optional reader close does not invalidate CUDA's own measurement.
    return allocator_ceiling(total, total, free, reserved, 0, local)


def allocator_ceiling(budget, total, free, reserved, reserve, local=None):
    """Bound the *whole process allocator*, crediting its own pool exactly once.

    The policy budget already excludes growth headroom. Live free memory and
    the WDDM process budget are independent bounds, each with that same reserve,
    not deductions from the already reduced policy budget. WDDM is not a PCIe
    traffic counter and allocations outside Torch remain outside this ceiling.
    """
    for value in (budget, total, free, reserved, reserve):
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError('GPU allocator admission requires finite nonnegative byte counts')
    if not budget or not total:
        raise ValueError('GPU allocator admission requires positive capacity and budget')
    bounds = dict(planning_budget_bytes=int(budget), physical_capacity_bytes=int(total),
                  live_pool_capacity_bytes=max(0, int(free + reserved - reserve)))
    native = None
    if isinstance(local, dict):
        driver_budget, usage = local.get('budget_bytes'), local.get('usage_bytes')
        if (type(driver_budget) is int and driver_budget > 0 and
                type(usage) is int and usage >= 0):
            native = max(0, usage - reserved)
            bounds['wddm_pool_capacity_bytes'] = max(0, int(driver_budget - native - reserve))
    return dict(allocator_limit_bytes=min(bounds.values()), bounds=bounds,
                observed_non_torch_local_bytes=native,
                owned_allocator_reserved_bytes=int(reserved), live_free_bytes=int(free),
                growth_reserve_bytes=int(reserve))


def configure(torch, budget_bytes, explicit_limit=None, *, reserve_bytes=0, system=None, capacity_trial=False):
    """Bound Windows, a low-memory trial, or an explicit benchmark allocation.

    Reclaimed pool blocks can still be reused by the caching allocator. Its own
    OOM is recoverable by the controller, unlike silently relying on WDDM to
    back excessive GPU allocation with system memory. This is not a whole-device
    limit and cannot prevent every instance of driver paging.
    """
    global _trial_allocator_owner
    from .policy import memory_fraction
    total = torch.cuda.get_device_properties(0).total_memory
    planned = memory_fraction(budget_bytes, total)
    if explicit_limit is not None and (type(explicit_limit) is not int or not 0 < explicit_limit <= total):
        raise ValueError('Benchmark allocator limit must be positive and no larger than physical VRAM')
    desktop = windows() if system is None else system == 'Windows'
    report = dict(device_total_bytes=total, budget_bytes=budget_bytes, planned_fraction=planned,
                  enforced=False, benchmark_allocator_limit_bytes=explicit_limit,
                  benchmark_allocator_limit_enforced=explicit_limit is not None,
                  capacity_trial_limit_enforced=False,
                  windows_allocator_limit_enforced=False,
                  scope='PyTorch caching allocator only; other CUDA allocations and whole-device usage are not capped.')
    limit = explicit_limit
    if desktop:
        # Initialize CUDA before asking the driver for a context-matched LUID.
        torch.cuda.init()
        free, _ = torch.cuda.mem_get_info()
        reserved = torch.cuda.memory_reserved()
        reader, local = None, None
        try:
            from .windows_gpu_memory import AdapterMemory
            reader = AdapterMemory()
            observed = reader.sample()
            local = observed.get('local')
            if not isinstance(local, dict) or type(local.get('budget_bytes')) is not int or local['budget_bytes'] <= 0:
                raise ValueError('WDDM returned no usable local GPU budget')
            report['windows_memory'] = dict(status='available', **observed)
        except (OSError, RuntimeError, AttributeError, ValueError) as error:
            report['windows_memory'] = dict(status='unavailable', reason=str(error))
        finally:
            if reader is not None:
                try:
                    reader.close()
                except Exception as error:
                    report['windows_memory_cleanup_error'] = str(error)
    elif capacity_trial:
        free, _ = torch.cuda.mem_get_info()
        reserved = torch.cuda.memory_reserved()
        local = None
    if desktop or capacity_trial:
        decision = allocator_ceiling(budget_bytes, total, free, reserved, reserve_bytes, local)
        report['admission'] = decision
        limit = min(decision['allocator_limit_bytes'], limit if limit is not None else total)
        if limit <= 0:
            raise torch.cuda.OutOfMemoryError('CUDA out of memory: no local GPU allocator capacity remains after live budgets')
        report.update(enforced=True, windows_allocator_limit_enforced=desktop,
                      capacity_trial_limit_enforced=bool(capacity_trial),
                      reason='Keep the CUDA allocator within the live local-memory budget; '
                             'recover with controlled weight offload instead of permitting allocator oversubscription.')
        # A resident process may already have a larger *unused* pool. A new
        # fraction affects future allocation, so release that pool once now.
        if reserved > limit:
            torch.cuda.empty_cache()
            report['released_cached_pool_before_request'] = True
    if limit is not None:
        torch.cuda.set_per_process_memory_fraction(limit / total)
        _trial_allocator_owner = torch if capacity_trial and not desktop else None
    elif _trial_allocator_owner is torch:
        # A later Linux request can regain ordinary capacity in the same
        # resident worker. Do not leave yesterday's temporary trial cap active.
        torch.cuda.set_per_process_memory_fraction(1.)
        _trial_allocator_owner = None
        report['released_capacity_trial_limit'] = True
    report['effective_allocator_limit_bytes'] = limit
    return report


class LiveGPUBudget:
    """Refresh a Windows allocator ceiling at idle CUDA stage/step boundaries.

    The 256 MiB reserve is growth headroom, in addition to observed allocations
    outside Torch. A larger driver budget must survive two observations before
    use. Pressure takes effect immediately, reclaiming optional weights first.
    No background thread changes CUDA limits or frees tensors in active kernels.
    """
    def __init__(self, torch, report, maximum_bytes, reserve_bytes, *, reader_factory=None):
        self.torch, self.report = torch, report
        self.total = report['device_total_bytes']
        self.limit = report['effective_allocator_limit_bytes']
        self.maximum = min(maximum_bytes, self.total,
                           report.get('benchmark_allocator_limit_bytes') or self.total)
        self.reserve = reserve_bytes
        self.pending = None
        self.started = time.monotonic()
        self.reader = None
        self.receipt = report['dynamic_budget'] = dict(enabled=True,
            base_reserve_bytes=reserve_bytes, maximum_allocator_bytes=self.maximum,
            minimum_limit_bytes=self.limit, maximum_limit_bytes=self.limit,
            observations=0, transitions=[], errors=[])
        try:
            if reader_factory is None:
                from .windows_gpu_memory import AdapterMemory
                reader_factory = AdapterMemory
            self.reader = reader_factory()
        except (OSError, RuntimeError, AttributeError, ValueError) as error:
            self._error(error)

    def _error(self, error):
        errors = self.receipt['errors']
        if len(errors) < 8 and str(error) not in errors:
            errors.append(str(error))

    def refresh(self, stage, reclaim=None):
        cuda = self.torch.cuda
        free, _ = cuda.mem_get_info()
        reserved = cuda.memory_reserved()
        local = None
        if self.reader is not None:
            try:
                local = self.reader.sample().get('local')
            except (OSError, RuntimeError, AttributeError, ValueError) as error:
                self._error(error)
        decision = allocator_ceiling(self.maximum, self.total, free, reserved, self.reserve, local)
        candidate = decision['allocator_limit_bytes']
        # An unavailable driver reading never grants extra residency. CUDA can
        # still report less live capacity and trigger reclamation.
        known = 'wddm_pool_capacity_bytes' in decision['bounds']
        if not known:
            candidate = min(candidate, self.limit)
        target, action = self.limit, 'hold'
        if candidate < self.limit:
            target, action, self.pending = candidate, 'shrink', None
        elif candidate >= self.limit + 64 * 2**20:
            if self.pending is not None:
                target, action = min(candidate, self.pending), 'grow'
                self.pending = None
            else:
                self.pending, action = candidate, 'pending'
        else:
            self.pending = None
        row = dict(stage=stage, elapsed_seconds=time.monotonic()-self.started,
                   previous_limit_bytes=self.limit, candidate_limit_bytes=candidate,
                   allocator_limit_bytes=target, action=action, driver_available=known,
                   live_free_bytes=free, reserved_bytes=reserved,
                   non_torch_local_bytes=decision['observed_non_torch_local_bytes'],
                   driver_budget_bytes=local.get('budget_bytes') if known else None)
        self.receipt['observations'] += 1
        self.receipt['last'] = row
        if action in ('shrink', 'grow'):
            if len(self.receipt['transitions']) < 64:
                self.receipt['transitions'].append(row)
            if action == 'shrink':
                if reclaim is not None:
                    reclaim(target)
                if cuda.memory_reserved() > target:
                    cuda.empty_cache()
            # The setter only governs future allocations. If active tensors
            # cannot fit after reclamation, use the existing controlled OOM
            # recovery rather than assume that setting a fraction evicts them.
            cuda.set_per_process_memory_fraction(max(0, target) / self.total)
            self.limit = target
            self.report['effective_allocator_limit_bytes'] = target
            self.receipt['minimum_limit_bytes'] = min(self.receipt['minimum_limit_bytes'], target)
            self.receipt['maximum_limit_bytes'] = max(self.receipt['maximum_limit_bytes'], target)
            if target <= 0 or cuda.memory_allocated() > target:
                raise cuda.OutOfMemoryError('CUDA out of memory: live local GPU budget decreased below active allocations')
        return self.limit

    def close(self):
        reader, self.reader = self.reader, None
        if reader is not None:
            try:
                reader.close()
            except (OSError, RuntimeError) as error:
                self._error(error)
