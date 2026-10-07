"""Exact signed column permutation from Comfy regular H to Sylvester H.

For H4 regular, H_regular = D H_sylvester P D.  P swaps the two
bits of each base-4 digit; D is -1 for each digit equal to 3.
Consequently (W H_regular) P D = W D H_sylvester. No requantization.
"""
from functools import lru_cache

import torch


@lru_cache(maxsize=3)
def tables(group):
    if group not in (16, 64, 256):
        raise ValueError("Expected a Comfy ConvRot group: 16, 64 or 256")
    permutation, signs = [], []
    for index in range(group):
        source, perm, sign, offset = index, 0, 1, 0
        while source or (1 << offset) < group:
            digit = source & 3
            perm |= ((digit & 1) << 1 | (digit >> 1)) << offset
            if digit == 3:
                sign = -sign
            source >>= 2
            offset += 2
        permutation.append(perm)
        signs.append(sign)
    return torch.tensor(permutation, dtype=torch.long), torch.tensor(signs, dtype=torch.int8)


def convert_weight(weight, group):
    """Return a bounded-tensor INT8 permutation, retaining the original scales."""
    if weight.dtype != torch.int8 or weight.ndim != 2 or weight.shape[1] % group:
        raise ValueError("Invalid ConvRot INT8 matrix")
    if bool((weight == -128).any()):
        raise ValueError("Signed permutation cannot preserve INT8 -128; expected symmetric -127..127")
    permutation, signs = tables(group)
    value = weight.reshape(-1, group).index_select(1, permutation.to(weight.device))
    value.mul_(signs.to(weight.device))
    return value.reshape(weight.shape).contiguous()


def activation_mult(features, group):
    if features % group:
        raise ValueError("ConvRot group does not divide input features")
    return tables(group)[1].float().repeat(features // group)
