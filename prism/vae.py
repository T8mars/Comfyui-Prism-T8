"""Keep Wan VAE singleton spatial attention valid on CUDA SDPA backends."""
from diffusers.models.autoencoders.autoencoder_kl_wan import WanAttentionBlock


class PrismWanAttentionBlock(WanAttentionBlock):
    def forward(self, x):
        if x.shape[-2:] != (1, 1):
            return super().forward(x)
        # A one-key softmax is exactly one, so attention returns V. The upstream
        # singleton QKV permutation can retain strideM=1 even after contiguous(),
        # which CUDA SDPA rejects on edge tiles. Preserve norm, V and projection.
        batch, channels, time, _, _ = x.shape
        hidden = x.permute(0, 2, 1, 3, 4).reshape(batch * time, channels, 1, 1)
        value = self.to_qkv(self.norm(hidden)).chunk(3, dim=1)[2]
        output = self.proj(value).view(batch, time, channels, 1, 1)
        return output.permute(0, 2, 1, 3, 4) + x


def adapt_video_vae(model):
    # Retain the configured module instances and every checkpoint key/storage.
    # No global monkey patch or modification of the installed diffusers package.
    for child in model.modules():
        if type(child) is WanAttentionBlock:
            child.__class__ = PrismWanAttentionBlock
    return model
