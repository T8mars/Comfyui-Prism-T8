"""Frozen inference offload restores original storage instead of copying CUDA back."""
import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from prism.format import TensorReader
from prism.offload import FrozenOffloadModule


class TiedCodec(nn.Module):
    def __init__(self, weight, scale):
        super().__init__()
        self.left, self.right = nn.Linear(16, 16, bias=False), nn.Linear(16, 16, bias=False)
        self.left.weight = self.right.weight = nn.Parameter(weight, requires_grad=False)
        self.left.register_buffer("scale", scale)
        self.right.register_buffer("scale", scale)
        self.config = {"native": True}
        self.tiling = False
        self.to_calls = []
        self.eval()

    @property
    def dtype(self):
        return self.left.weight.dtype

    def encode(self, value):
        return self.left(value) * self.left.scale

    def decode(self, value):
        return self.right(value) * self.right.scale

    def enable_tiling(self):
        self.tiling = True

    def forward(self, value):
        return self.decode(self.encode(value))

    def to(self, *args, **kwargs):
        self.to_calls.append((args, kwargs))
        return super().to(*args, **kwargs)


@pytest.fixture
def mapped_codec(tmp_path):
    path = tmp_path / "immutable.safetensors"
    save_file({"weight": torch.arange(256, dtype=torch.float32).reshape(16, 16) / 256,
               "scale": torch.linspace(0.5, 1.5, 16)}, path)
    with TensorReader(path) as reader:
        weight, scale = reader.get_tensor("weight"), reader.get_tensor("scale")
    return TiedCodec(weight, scale), path


def test_cpu_restore_reuses_mapping_and_preserves_native_api(mapped_codec):
    module, path = mapped_codec
    weight_pointer, scale_pointer = module.left.weight.data_ptr(), module.left.scale.data_ptr()
    wrapped = FrozenOffloadModule(module)
    value = torch.linspace(-1, 1, 16).unsqueeze(0)
    expected = module(value)
    wrapped.enable_tiling()
    assert module.tiling and wrapped.config is module.config and wrapped.dtype == module.dtype
    for _ in range(3):
        assert wrapped.to("cpu") is wrapped
        assert wrapped.left.weight is wrapped.right.weight
        assert wrapped.left.scale is wrapped.right.scale
        assert wrapped.left.weight.data_ptr() == weight_pointer
        assert wrapped.left.scale.data_ptr() == scale_pointer
        assert torch.equal(wrapped(value), expected)
        assert torch.equal(wrapped.decode(value), module.decode(value))
    assert not module.to_calls, "CPU restore must not enter Module.to's copy path"
    # Restore must not modify the read-only file mapping.
    with TensorReader(path, copy=True) as reader:
        assert torch.equal(reader.get_tensor("weight"), wrapped.left.weight)


def test_explicit_dtype_overloads_update_snapshot_and_keep_aliases(mapped_codec):
    module, _ = mapped_codec
    wrapped = FrozenOffloadModule(module)
    assert wrapped.to(torch.empty((), dtype=torch.float64)) is wrapped
    assert wrapped.dtype == torch.float64 and wrapped.left.scale.dtype == torch.float64
    pointer = wrapped.left.weight.data_ptr()
    assert wrapped.cpu().left.weight.data_ptr() == pointer
    assert wrapped.left.weight is wrapped.right.weight
    assert wrapped.left.scale is wrapped.right.scale
    assert wrapped.bfloat16().dtype == torch.bfloat16
    assert wrapped.float().dtype == torch.float32
    assert wrapped.half().dtype == torch.float16
    with pytest.raises(TypeError, match="floating point"):
        wrapped.to(dtype=torch.int8)
    assert wrapped.dtype == torch.float16


def test_explicit_memory_format_survives_cpu_restore():
    module = nn.Conv2d(4, 4, 3).eval().requires_grad_(False)
    wrapped = FrozenOffloadModule(module).to(memory_format=torch.channels_last)
    pointer = wrapped.weight.data_ptr()
    assert wrapped.weight.is_contiguous(memory_format=torch.channels_last)
    assert wrapped.cpu().weight.data_ptr() == pointer
    assert wrapped.weight.is_contiguous(memory_format=torch.channels_last)


def test_trainable_parameters_are_rejected():
    with pytest.raises(ValueError, match="frozen CPU"):
        FrozenOffloadModule(nn.Linear(16, 16))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA transfer needs a GPU")
def test_repeated_gpu_transfer_restores_same_mmap_and_aliases(mapped_codec):
    module, _ = mapped_codec
    pointers = module.left.weight.data_ptr(), module.left.scale.data_ptr()
    wrapped = FrozenOffloadModule(module)
    expected = module(torch.linspace(-1, 1, 16).unsqueeze(0))
    with torch.inference_mode():
        for _ in range(3):
            wrapped.cuda()
            assert wrapped.left.weight is wrapped.right.weight
            assert wrapped.left.scale is wrapped.right.scale
            assert wrapped.left.weight.is_cuda and wrapped.left.scale.is_cuda
            output = wrapped(torch.linspace(-1, 1, 16, device="cuda").unsqueeze(0))
            torch.testing.assert_close(output.cpu(), expected)
            call_count = len(module.to_calls)
            wrapped.cpu()
            assert len(module.to_calls) == call_count
            assert wrapped.left.weight.data_ptr() == pointers[0]
            assert wrapped.left.scale.data_ptr() == pointers[1]
            assert wrapped.left.weight is wrapped.right.weight
            assert wrapped.left.scale is wrapped.right.scale
            assert all(t.device.type == "cpu" for t in wrapped.parameters())
            assert all(t.device.type == "cpu" for t in wrapped.buffers())

