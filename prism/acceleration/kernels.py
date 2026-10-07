"""Validated FreeVideo kernel switches, applied before the first model forward.

Adapted from FlashML-org/FreeVideo (Apache-2.0), commit
40525196a33bc7ff6a455ce9b18b74f3c827bbb1, prism_worker.py::kernel_setup.
The vendored kernels themselves remain unchanged.
"""
import os


def initialize(policy):
    os.environ['PRISM_BSA_HEAD_CHUNK'] = str(policy.get('head_chunk', 8))
    from .vendor.prism_model import sampling
    sampling.PARK_RESIDUAL = bool(policy.get('park_residual'))
    dynamic_shapes = None
    try:
        import torch._dynamo
        torch._dynamo.config.automatic_dynamic_shapes = False
        dynamic_shapes = torch._dynamo.config.automatic_dynamic_shapes
    except (ImportError, AttributeError):
        pass
    if policy.get('rope_chunk_elems'):
        from .vendor.prism_model import wan_video_dit
        wan_video_dit._ROPE_CHUNK_ELEMS = int(policy['rope_chunk_elems'])
    from .vendor.prism_model import sage_bsa, block_sparse_attention  # noqa: F401
    from .vendor.prism_model.block_sparse_attention import dynamic_block_attention as dba
    if policy.get('attention') == 'exact':
        os.environ['PRISM_SAGE_BSA'] = 'exact'
    else:
        sage_bsa.patch_prism(pv=policy.get('sage', 'auto'), head_chunk=policy.get('head_chunk', 8))
    if not getattr(dba._dyn_bsa_kernel, '_skip_pad', False):
        original = dba._dyn_bsa_kernel

        class SkipPad:
            _skip_pad = True
            _orig = original

            @staticmethod
            def apply(q, k, v, sm_scale, idx, lens, cq, ck, sp, valid):
                if valid is not None:
                    qv = valid.view(-1, cq).any(dim=1)
                    lens = lens * qv.view(1, 1, -1).to(lens.dtype)
                return original.apply(q, k, v, sm_scale, idx, lens, cq, ck, sp, valid)

        dba._dyn_bsa_kernel = SkipPad
    return dict(automatic_dynamic_shapes=dynamic_shapes,
                skip_fully_padded_queries=bool(getattr(dba._dyn_bsa_kernel, '_skip_pad', False)),
                sage_fallback_route=policy.get('attention') != 'exact',
                source='prism_worker.py::kernel_setup')
