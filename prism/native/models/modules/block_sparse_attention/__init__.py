# Vendored from Tencent-Hunyuan/Prism 883e90a5c90dc8b7044c65eba0bb64e9342cb46a. See NATIVE_CHANGES.md.
from .bsa_interface import (
    flash_attn_bsa_3d,
    flash_attn_bsa_cross,
    flash_attn_bsa_3d_audio_guided,
    flash_attn_bsa_3d_variance_guided,
    compute_audio_spatial_concentration_gate,
    compute_timestep_reliability_gate,
    compute_block_audio_saliency,
    audio_weighted_mean_pooling_compression,
    compute_channel_variance_density,
)

from .bias_rectification import (
    bsa_taylor_sparse_attn,
    bsa_rectified_sparse_attn,
)
