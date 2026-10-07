"""Torch-free executable defaults for request identity, never policy selection.

The values mirror Engine.__init__ and decode_to_file, checked by CPU AST tests.
They describe what omitted options execute, not recommended settings. Callers
keep the user's original profile for reporting and execution. Future changes to
these defaults require reviewing the durable-ledger schema migration as well.
"""
from types import MappingProxyType


ENGINE_DEFAULTS = MappingProxyType({
    'attention': 'dense', 'prefetch': True, 'adaln_cache': False,
    'inference_kernels': False, 'query_chunk': 0, 'ff_chunk': 0, 'head_chunk': 0,
    'offload_refiner': False, 'attention_cpu_outputs': False, 'preload_host': False,
    'pin_host_weights': False, 'adaln_disk_cache': False, 'projection_chunk': 0,
    'grouped_attention_outputs': False, 'fp8_ff_recompute': False,
    'resident_blocks': 0, 'pin_host_gb': 0., 'cache_refined_text': False, 'steps': 8,
    'window_batch': 1, 'linear_compute': 'native-fp8', 'fp8_gemm': 'auto',
    'window_varlen': False, 'varlen_smooth_k': True, 'task': 't2va',
    'stream_weights': False, 'head_parallelism': 1, 'residual_offload': False,
})
DECODER_DEFAULTS = MappingProxyType({
    'offload': False, 'prefetch': True, 'tile_group': False,
    'preload': False, 'pin_weights': False,
    'stream_output': False, 'stream_weights': False,
    'resident_blocks': 0, 'linear_compute_cache': False,
})


def engine_options(options):
    if not isinstance(options, dict):
        raise ValueError('Engine options must be a mapping')
    return _resolved(ENGINE_DEFAULTS, options)


def decoder_options(options):
    if not isinstance(options, dict):
        raise ValueError('Decoder options must be a mapping')
    return _resolved(DECODER_DEFAULTS, options)


def _resolved(defaults, options):
    result = dict(defaults)
    for key, value in options.items():
        # JSON may spell the same default as 0/0.0 or True/1. Retain the
        # declared representation when equality means identical execution.
        result[key] = defaults[key] if key in defaults and value == defaults[key] else value
    return result
