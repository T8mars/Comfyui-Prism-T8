"""Bound H3 activations with head groups; optional locally validated overlap.

Reuses upstream's head-sliced linear branch (originally used by Ulysses).
No heads, windows, text states or scan frames are dropped. Sliced GEMMs can
change floating-point reduction order and must be reported as an ablation.
"""
from contextlib import nullcontext
import types
import torch
import torch.nn.functional as F


def install_head_chunks(transformer, policy, chunk, cpu_outputs=False, projection_chunk=1024, grouped_outputs=False,
                        parallelism=1):
    if chunk < 1 or projection_chunk < 1:
        raise ValueError('Head and attention-output chunks must be positive')
    if type(parallelism) is not int or parallelism not in (1, 2, 4):
        raise ValueError('Head parallelism must be 1, 2 or 4')
    if parallelism > 1 and (cpu_outputs or grouped_outputs):
        raise ValueError('Parallel head groups require GPU attention outputs')
    from src.models.hybrid_transform import iter_hybrids
    from src.models.softmax_attention.kernels import _qk_prep
    from src.models.ops.fp8_linear import Fp8Linear, quantize_activation
    from .fp8_ops import ColumnScale, project_with_scale, row_scale, sliced_projection

    def forward(attn, x, rotary):
        layout = attn.layout
        if layout is None or not attn.inference_mode:
            raise ValueError('Head chunks require an explicit packed layout and inference kernels')
        bounds = attn._bounds(layout)
        if all(lo <= 0 and hi >= layout.num_frames - 1 for lo, hi in bounds):
            raise ValueError('This profile is for the VDN hybrid window layout')
        orig, branch = attn.orig, attn.linear_attention
        projections = (orig.to_q, orig.to_k, orig.to_v)
        quantized = quantize_activation(x) if all(isinstance(p, Fp8Linear) for p in projections) else None
        video = slice(layout.video_start, layout.video_end)
        text = slice(*layout.text_range)
        frame_mean = x[video].view(layout.num_frames, layout.tokens_per_frame, -1).mean(1, dtype=torch.float32)
        gate_hidden = x[video] if branch.output_gate.down is None else branch.output_gate.down(x[video])
        soft_shape = (len(x), attn.num_heads * attn.head_dim)
        linear_shape = (layout.num_frames * layout.tokens_per_frame, attn.num_heads * branch.head_dim)
        soft_scale = ColumnScale(len(x), x.device) if cpu_outputs and isinstance(orig.to_out[0], Fp8Linear) else None
        linear_scale = ColumnScale(linear_shape[0], x.device) if cpu_outputs and isinstance(attn.to_out_linear, Fp8Linear) else None
        output_width = orig.to_out[0].weight_fp8.shape[0] if isinstance(orig.to_out[0], Fp8Linear) else orig.to_out[0].out_features
        if grouped_outputs:
            from .activation_staging import HeadReadouts
            widths = [min(chunk, attn.num_heads - start) * attn.head_dim for start in range(0, attn.num_heads, chunk)]
            stores = getattr(policy, 'group_readouts', None)
            if stores is None or any(not store.supports(rows, widths, x.dtype, projection_chunk)
                                     for store, rows in zip(stores, (len(x), linear_shape[0]))):
                policy.group_readouts = tuple(HeadReadouts(rows, widths, x.dtype, projection_chunk)
                                             for rows in (len(x), linear_shape[0]))
            soft, linear = policy.group_readouts
        elif cpu_outputs:
            key = (soft_shape, linear_shape, x.dtype)
            if getattr(policy, 'host_output_key', None) != key:
                policy.host_outputs = (torch.empty(soft_shape, dtype=x.dtype, pin_memory=True),
                                       torch.empty(linear_shape, dtype=x.dtype, pin_memory=True))
                policy.host_output_key = key
            soft, linear = policy.host_outputs
        else:
            soft, linear = x.new_empty(soft_shape), x.new_empty(linear_shape)
        def group(start):
            end = min(start + chunk, attn.num_heads)
            heads = slice(start, end)
            channels = slice(start * attn.head_dim, end * attn.head_dim)
            raw = tuple(sliced_projection(proj, x, channels, quantized)
                        .view(len(x), end - start, attn.head_dim) for proj in projections)
            q = _qk_prep(raw[0], orig.norm_q.weight, orig.norm_q.eps, *rotary)
            k = _qk_prep(raw[1], orig.norm_k.weight, orig.norm_k.eps, *rotary)
            attended = policy(q, k, raw[2], layout, bounds, attn.head_dim ** -.5, attn.anchor_frames)
            del q, k
            if attn.enable_softmax_gate:
                gate_module = attn.softmax_gate
                gate_x = x if gate_module.down is None else gate_module.down(x)
                gate = torch.sigmoid(F.linear(gate_x, gate_module.up.weight[heads], gate_module.up.bias[heads]))
                attended.mul_(gate[..., None])
            if soft_scale is not None:
                soft_scale.update(attended.flatten(1))
            if grouped_outputs:
                soft.store(start // chunk, attended.flatten(1))
            else:
                soft[:, channels].copy_(attended.flatten(1), non_blocking=cpu_outputs)
            del attended
            beta = torch.sigmoid(F.linear(x[video], branch.beta_proj.weight[heads]))
            text_beta = torch.sigmoid(F.linear(x[text], branch.beta_proj.weight[heads])) if attn.enable_text_state else None
            linear_channels = slice(start * branch.head_dim, end * branch.head_dim)
            gate = torch.sigmoid(F.linear(gate_hidden, branch.output_gate.up.weight[linear_channels],
                                         branch.output_gate.up.bias[linear_channels]))
            gate = gate.view(-1, end - start, branch.head_dim)
            readout = branch(None, layout.num_frames, layout.tokens_per_frame, bounds,
                             qkv_raw=tuple(t[video] for t in raw), frame_size=layout.frame_size,
                             skip_ends=attn.anchor_frames == 'both',
                             text_qkv_raw=tuple(t[text] for t in raw) if attn.enable_text_state else None,
                             inference=True, heads=heads, beta=beta, gate=gate,
                             frame_mean=frame_mean, text_beta=text_beta)
            if linear_scale is not None:
                linear_scale.update(readout)
            if grouped_outputs:
                linear.store(start // chunk, readout.contiguous())
            else:
                linear[:, linear_channels].copy_(readout, non_blocking=cpu_outputs)
            del raw, readout, gate, beta, text_beta

        # Keep the exact projection/head/window shapes. Merely widening the
        # head chunk changes GEMM rounding; independent streams can overlap
        # these original groups without changing their arithmetic partition.
        # First execute this shape on the calling stream: upstream attention
        # and scan caches lazily create shared device indices/compiled kernels.
        warm_key = (tuple(x.shape), x.dtype, str(x.device), chunk, layout, tuple(bounds),
                    attn.anchor_frames, attn.enable_text_state, policy.window_batch,
                    policy.backend, policy.window_varlen)
        concurrent = parallelism > 1 and getattr(attn, '_freevideo_parallel_warm', None) == warm_key
        parent = torch.cuda.current_stream(x.device) if concurrent else None
        streams = []
        if concurrent:
            policy.prepare(layout, bounds, x.device, attn.anchor_frames, batches=True)
            from src.models.linear_attention.scan import _gather_indices
            _gather_indices(bounds, layout.num_frames, x.device)
            stream_key = (str(x.device), parallelism)
            if getattr(policy, 'head_stream_key', None) != stream_key:
                policy.head_streams = [torch.cuda.Stream(device=x.device) for _ in range(parallelism)]
                policy.head_stream_key = stream_key
            streams = policy.head_streams
            # Inputs and disjoint output planes belong to the parent allocator.
            # record_stream also protects their lifetime if a group raises OOM.
            shared = [x, frame_mean, gate_hidden, soft, linear, *rotary]
            if quantized is not None:
                shared.extend(quantized)
            for stream in streams:
                stream.wait_stream(parent)
                for tensor in shared:
                    tensor.record_stream(stream)
        try:
            for index, start in enumerate(range(0, attn.num_heads, chunk)):
                with torch.cuda.stream(streams[index % len(streams)]) if streams else nullcontext():
                    group(start)
        finally:
            # Output projection, offload cleanup and the next block cannot
            # race unfinished groups, including after a recoverable OOM.
            for stream in streams:
                parent.wait_stream(stream)
        if parallelism > 1:
            attn._freevideo_parallel_warm = warm_key
            counter = 'parallel_head_calls' if concurrent else 'parallel_head_warmups'
            setattr(policy, counter, getattr(policy, counter, 0) + 1)
        del quantized
        soft_scale = soft_scale.finish() if soft_scale is not None else None
        linear_scale = linear_scale.finish() if linear_scale is not None else None
        if grouped_outputs:
            torch.cuda.current_stream(x.device).synchronize()
            out = x.new_empty((len(x), output_width))
            for rows, part in soft.project(lambda part: part, out, len(x)):
                scale = row_scale(soft_scale, rows) if soft_scale is not None else None
                out[rows] = orig.to_out[1](project_with_scale(orig.to_out[0], part, scale))
            for rows, part in linear.project(lambda part: part, out, linear_shape[0]):
                scale = row_scale(linear_scale, rows) if linear_scale is not None else None
                out[video][rows].add_(project_with_scale(attn.to_out_linear, part, scale))
            return out
        if cpu_outputs:
            # All D2H writes must finish before the pinned outputs are read by H2D.
            torch.cuda.current_stream(x.device).synchronize()
            # Project bounded row slices directly from the host readout planes.
            # Copying whole planes back would retain >2 GB of temporary buffers.
            out = x.new_empty((len(x), output_width))
            for row in range(0, len(x), projection_chunk):
                rows = slice(row, row + projection_chunk)
                part = soft[rows].to(x.device, non_blocking=True)
                scale = row_scale(soft_scale, rows) if soft_scale is not None else None
                out[rows] = orig.to_out[1](project_with_scale(orig.to_out[0], part, scale))
                del part
            for row in range(0, len(linear), projection_chunk):
                rows = slice(row, row + projection_chunk)
                part = linear[rows].to(x.device, non_blocking=True)
                scale = row_scale(linear_scale, rows) if linear_scale is not None else None
                out[video][rows].add_(project_with_scale(attn.to_out_linear, part, scale))
                del part
            return out
        out = orig.to_out[1](orig.to_out[0](soft))
        del soft
        out[video] += attn.to_out_linear(linear)
        return out

    for attn in iter_hybrids(transformer):
        if attn.head_dim != attn.linear_attention.head_dim:
            raise ValueError('The shared-QKV profile requires matching branch head dimensions')
        attn._hybrid_forward = types.MethodType(forward, attn)
