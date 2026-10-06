"""Frozen encoder/codec and fused-block CPU offload; diffusion math is unchanged."""
import inspect

import torch
from torch import nn


class _FrozenCPUState:
    """Immutable CPU storage owners plus original parameter/buffer alias groups."""
    def __init__(self, module):
        self.parameters = {}
        self.buffers = {}
        self.records = []
        for child in module.modules():
            parameters, buffers = {}, {}
            for name, value in child._parameters.items():
                parameters[name] = None if value is None else id(value)
                if value is not None:
                    if value.device.type != "cpu" or value.requires_grad:
                        raise ValueError("Frozen offload requires frozen CPU parameters")
                    self.parameters.setdefault(id(value), value.detach())
            for name, value in child._buffers.items():
                buffers[name] = None if value is None else id(value)
                if value is not None:
                    if value.device.type != "cpu":
                        raise ValueError("Frozen offload requires CPU buffers")
                    self.buffers.setdefault(id(value), value.detach())
            self.records.append((child, parameters, buffers))

    def restore(self):
        # Detached storage owners survive Parameter.data changes during to(cuda).
        parameters = {key: nn.Parameter(value, requires_grad=False) for key, value in self.parameters.items()}
        for child, names, buffers in self.records:
            child._parameters.update({name: parameters[key] if key is not None else None for name, key in names.items()})
            child._buffers.update({name: self.buffers[key] if key is not None else None for name, key in buffers.items()})

    def rebind_aliases(self):
        # Module._apply can convert repeated registrations independently, notably
        # buffers. Keep the original sharing groups after a device/dtype change.
        parameters, buffers = {}, {}
        for child, names, buffer_names in self.records:
            for name, key in names.items():
                if key is not None:
                    child._parameters[name] = parameters.setdefault(key, child._parameters[name])
            for name, key in buffer_names.items():
                if key is not None:
                    child._buffers[name] = buffers.setdefault(key, child._buffers[name])


class FrozenOffloadModule(nn.Module):
    """Keep a frozen encoder/codec's original CPU storage across GPU transfers.

    Native config, dtype, encode/decode, tokenizer forward and tiling methods are
    delegated to the original module. CPU restoration only rebinds storage;
    explicitly requested dtype/layout changes create a new CPU snapshot once.
    This wrapper is for immutable inference weights, including ConvRot buffers.
    """
    def __init__(self, module):
        super().__init__()
        self.module = module
        self.cpu_state = _FrozenCPUState(module)
        self.current_device = torch.device("cpu")
        self.train(module.training)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(super().__getattr__("module"), name)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def to(self, *args, **kwargs):
        # Use the same overload parser as nn.Module.to (device/dtype/tensor).
        device, dtype, non_blocking, memory_format = torch._C._nn._parse_to(*args, **kwargs)
        if dtype is not None and not (dtype.is_floating_point or dtype.is_complex):
            raise TypeError("nn.Module.to only accepts floating point or complex dtypes")
        device = torch.device(device) if device is not None else self.current_device
        if dtype is not None or memory_format is not None:
            self.cpu_state.restore()
            options = {"dtype": dtype, "non_blocking": non_blocking}
            if memory_format is not None:
                options["memory_format"] = memory_format
            self.module.to("cpu", **options)
            self.cpu_state.rebind_aliases()
            self.cpu_state = _FrozenCPUState(self.module)
        if device.type == "cpu":
            self.cpu_state.restore()
        else:
            self.module.to(device, non_blocking=non_blocking)
            self.cpu_state.rebind_aliases()
        self.current_device = device
        return self

    def cpu(self):
        return self.to("cpu")

    def cuda(self, device=None):
        return self.to(torch.device("cuda", device))

    def float(self):
        return self.to(dtype=torch.float32)

    def double(self):
        return self.to(dtype=torch.float64)

    def half(self):
        return self.to(dtype=torch.float16)

    def bfloat16(self):
        return self.to(dtype=torch.bfloat16)


class ManagedTransformer(nn.Module):
    def __init__(self, bridge, block_offload=False, interrupt=None, fixed_device=None):
        super().__init__()
        self.bridge = bridge
        self.block_offload = block_offload
        self.device = torch.device("cpu")
        self.fixed_device = fixed_device
        self.handles = []
        self.cpu_state = {}
        self.fused_signatures = {}
        if block_offload:
            self.fused_signatures = {id(block): inspect.signature(block.forward) for block in bridge.fusion_blocks}
            blocks = list(bridge.fusion_blocks) + list(bridge.remaining_video_blocks)
            if bridge.video_dit_2 is not None:
                blocks += list(bridge.video_dit_2.blocks)
            for block in blocks:
                # Keep original mmap-backed CPU tensors. In inference, weights
                # never change; restoring these avoids GPU->CPU copies and a
                # private 32GB RAM duplicate after one pass through the model.
                self.cpu_state[id(block)] = [(module, {name: value.detach() if value is not None else None
                                                      for name, value in module._parameters.items()}, dict(module._buffers))
                                              for module in block.modules()]
                self.handles.append(block.register_forward_pre_hook(self._pre, with_kwargs=True))
                self.handles.append(block.register_forward_hook(self._post, always_call=True))
        self.interrupt = interrupt

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(super().__getattr__("bridge"), name)

    def _pre(self, module, args, kwargs=None):
        if self.interrupt:
            self.interrupt()
        signature = self.fused_signatures.get(id(module))
        if signature is not None:
            arguments = signature.bind_partial(*args, **(kwargs or {})).arguments
            if arguments.get("override_video_block") is not None:
                # The low-noise override has its own hook. Moving the entire
                # fused block would also read and upload the idle primary expert.
                for child in (module.audio_block, module.a2v_conditioner, module.v2a_conditioner):
                    if child is not None:
                        child.to(self.device)
                return
        module.to(self.device)

    def _post(self, module, args, result):
        self._restore(module)

    def _restore(self, module):
        for child, parameters, buffers in self.cpu_state[id(module)]:
            # nn.Module.to() may mutate Parameter.data in place. Keep detached
            # CPU tensors in the snapshot and create wrappers on every restore.
            child._parameters.update({name: nn.Parameter(value, requires_grad=False) if value is not None else None
                                      for name, value in parameters.items()})
            child._buffers.update(buffers)

    def to(self, device, *args, **kwargs):
        device = torch.device(device)
        if self.fixed_device is not None:
            # FSDP owns transformer parameter residency. The pipeline can still
            # offload its frozen encoders without moving sharded parameters.
            self.device = torch.device(self.fixed_device)
            return self
        self.device = device
        if not self.block_offload or device.type == "cpu":
            self.bridge.to(device, *args, **kwargs)
        else:
            # DiT blocks remain on CPU. Permanent embeddings/heads/patch convolutions
            # are small; the secondary expert's blocks must also remain on CPU.
            for name, module in self.bridge.named_children():
                if name in ("fusion_blocks", "remaining_video_blocks"):
                    continue
                if name == "video_dit_2":
                    for child_name, child in module.named_children():
                        if child_name != "blocks":
                            child.to(device, *args, **kwargs)
                else:
                    module.to(device, *args, **kwargs)
        return self

    def forward(self, *args, **kwargs):
        return self.bridge(*args, **kwargs)

    def close(self):
        for block in list(self.bridge.fusion_blocks) + list(self.bridge.remaining_video_blocks):
            if id(block) in self.cpu_state:
                self._restore(block)
        if self.bridge.video_dit_2 is not None:
            for block in self.bridge.video_dit_2.blocks:
                if id(block) in self.cpu_state:
                    self._restore(block)
        self.to("cpu")
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.cpu_state.clear()
        self.fused_signatures.clear()
