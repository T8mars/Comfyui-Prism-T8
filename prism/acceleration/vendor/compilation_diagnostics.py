"""Read existing PyTorch compiler counters; never import or profile torch."""
import math
import sys


TIMERS = {
    'frame_compile': '_compile.compile_inner',
    'torch_build_hash': 'inductor_codecache_torch_key',
    'backend_compile': 'compile_fx_inner',
    'graph_lowering': 'GraphLowering.run',
    'code_generation': 'GraphLowering.codegen',
    'module_load': 'PyCodeCache.load_by_key_path',
    'kernel_reload': 'reload_kernel_in_parent',
    'async_compile_wait': 'async_compile.wait',
    'autograd_cache_load': 'AOTAutogradCache.inductor_load',
    'autotune': 'CachingAutotuner.benchmark_all_configs',
}
CACHES = {
    'graph_hits': ('inductor', 'fxgraph_cache_hit'),
    'graph_misses': ('inductor', 'fxgraph_cache_miss'),
    'graph_bypasses': ('inductor', 'fxgraph_cache_bypass'),
    'autograd_hits': ('aot_autograd', 'autograd_cache_hit'),
    'autograd_misses': ('aot_autograd', 'autograd_cache_miss'),
    'unique_graphs': ('stats', 'unique_graphs'),
}
SCOPE = ('Existing PyTorch host timers, including tracing, cache lookup/loading and code generation. '
         'Timers nest and may overlap GPU work; do not add them together. '
         'Missing fields are unavailable, not evidence of zero compilation.')


def _number(value):
    return type(value) in (float, int) and math.isfinite(value) and value >= 0


def sanitize(value):
    """Keep only fixed numeric fields in diagnostic reports."""
    if not isinstance(value, dict):
        return {}
    result = {'available': value.get('available') is True}
    for group, allowed in (('time_seconds', TIMERS), ('cache_counts', CACHES)):
        values = value.get(group)
        if isinstance(values, dict):
            result[group] = {key: values[key] for key in allowed if _number(values.get(key))}
    return result


def snapshot():
    module = sys.modules.get('torch._dynamo.utils')
    timers = getattr(module, 'compilation_time_metrics', None)
    if not isinstance(timers, dict):
        return {'available': False}
    result = dict(available=True, time_seconds={}, cache_counts={})
    for label, key in TIMERS.items():
        values = timers.get(key)
        if isinstance(values, list) and all(_number(value) for value in values):
            result['time_seconds'][label] = sum(values)
    counters = getattr(module, 'counters', {})
    for label, (group, key) in CACHES.items():
        values = counters.get(group, {}) if isinstance(counters, dict) else {}
        value = values.get(key) if isinstance(values, dict) else None
        if _number(value):
            result['cache_counts'][label] = value
    return result


def difference(before, after):
    result = {'available': after.get('available') is True}
    for group in ('time_seconds', 'cache_counts'):
        previous = before.get(group, {})
        result[group] = {key: value - previous.get(key, 0)
                         for key, value in after.get(group, {}).items()
                         if value >= previous.get(key, 0)}
    return result
