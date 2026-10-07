import os

import torch

from prism.acceleration.kernels import initialize


def test_upstream_setup_installs_static_shapes_sage_and_padding_guard(monkeypatch):
    import torch._dynamo
    from prism.acceleration.vendor.prism_model import sampling, sage_bsa, wan_video_dit
    from prism.acceleration.vendor.prism_model.block_sparse_attention import dynamic_block_attention as dba
    calls = []
    monkeypatch.setattr(sage_bsa, 'patch_prism', lambda **kw: calls.append(kw))
    monkeypatch.setattr(torch._dynamo.config, 'automatic_dynamic_shapes', True)
    monkeypatch.setattr(sampling, 'PARK_RESIDUAL', False)
    monkeypatch.setattr(wan_video_dit, '_ROPE_CHUNK_ELEMS', 0)
    monkeypatch.setenv('PRISM_BSA_HEAD_CHUNK', '1')
    class Original:
        @staticmethod
        def apply(q, k, v, sm_scale, idx, lens, cq, ck, sp, valid):
            return lens
    monkeypatch.setattr(dba, '_dyn_bsa_kernel', Original)
    actual_setup = initialize(dict(head_chunk=4, sage='fp8', park_residual=True, rope_chunk_elems=4096))
    assert actual_setup == dict(automatic_dynamic_shapes=False, skip_fully_padded_queries=True,
                               sage_fallback_route=True, source='prism_worker.py::kernel_setup')
    assert not torch._dynamo.config.automatic_dynamic_shapes
    assert sampling.PARK_RESIDUAL and wan_video_dit._ROPE_CHUNK_ELEMS == 4096
    assert os.environ['PRISM_BSA_HEAD_CHUNK'] == '4'
    assert calls == [dict(pv='fp8', head_chunk=4)]
    lens = torch.tensor([[[7, 9]]], dtype=torch.int32)
    valid = torch.tensor([True, False, False, False])
    actual = dba._dyn_bsa_kernel.apply(None, None, None, None, None, lens, 2, 2, None, valid)
    assert torch.equal(actual, torch.tensor([[[7, 0]]], dtype=torch.int32))
    assert torch.equal(lens, torch.tensor([[[7, 9]]], dtype=torch.int32))
    installed = dba._dyn_bsa_kernel
    initialize(dict(head_chunk=4))
    assert dba._dyn_bsa_kernel is installed
    assert torch.equal(installed.apply(None, None, None, None, None, lens, 2, 2, None, None), lens)
