"""Release consumed activations and reuse branch outputs in fused H3 blocks."""
import types
import torch


def _post_into_branch(residual, gate, indices, branch):
    branch.copy_(residual + gate.index_select(0, indices) * branch)
    return branch


def install_bounded_blocks(model, residual_offload=False):
    from src.models.ops.fused_block import _compiled, _pre_ref
    pre = _compiled('pre', _pre_ref)
    # Autotuning a mutating kernel clones its full branch input. That extra
    # ~758 MiB copy defeats reuse under small memory limits. This elementwise
    # kernel needs no shape search or reduction; use the fixed launch policy.
    post = torch.compile(_post_into_branch, dynamic=False,
                         options={'triton.autotune_pointwise': False})

    def forward(self, hidden_states, temb, adaln_indices, rotary_emb, attention_mask=None):
        if torch.is_grad_enabled():
            raise RuntimeError('Buffer reuse requires inference without autograd')
        hidden, indices = hidden_states, adaln_indices
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = self.adaln_proj(temb)
        normalized = pre(hidden, self.norm1.weight, self.norm1.eps, scale_a, shift_a, indices)
        branch = self.attn(normalized, rotary_emb, attention_mask)
        del normalized
        hidden = post(hidden, gate_a, indices, branch)
        del branch
        normalized = pre(hidden, self.norm2.weight, self.norm2.eps, scale_f, shift_f, indices)
        branch = self.ff(normalized)
        del normalized
        return post(hidden, gate_f, indices, branch)

    if residual_offload:
        from .residual import residual_forward
        forward = residual_forward(pre, post)
    for block in model.transformer_blocks:
        block.forward = types.MethodType(forward, block)
