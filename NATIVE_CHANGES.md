# Native source provenance

`prism/native/` is derived from Tencent-Hunyuan/Prism commit
`883e90a5c90dc8b7044c65eba0bb64e9342cb46a`. Its original license and third-party
notices are preserved in `LICENSE`. `scripts/vendor_native.py` reproduces this
subset from a checkout at `.research/Prism`.

Local changes:

- Split six semicolon-separated statements in `dynamic_block_shape.py` for the
  Comfy Registry E702 preflight. The Python AST is unchanged.
- Private relative imports, allowing other ComfyUI plugins to use their own `hymm`.
- Load BSA kernels when requested, so dense SDPA does not require Triton.
- Eager video modulation and cross-modal rotary execution; no mandatory Inductor
  compilation on first inference. FP16/BF16 pointwise intermediates are explicitly
  FP32, with the final output restored according to the original floating dtype
  promotion rules. This preserves the original compiled rounding: the former
  eager BF16 implementation differed from the compiled kernels by roughly
  0.23–0.30% relative L2 in a measured native-shape comparison. FP32/FP64 precision
  and broadcasting remain supported. CPU boundary cases and GPU comparisons
  against the original compiled formulas cover the change; it is not, by itself,
  evidence that a full sample's color or audio issue is resolved.
- Disable import-time downloading of FlashAttention 3 kernel code. Existing
  FlashAttention 2 can still be used. Incompatible optional CUDA extensions are
  treated as unavailable; dense attention remains available.
- A sampling callback for ComfyUI progress and interruption.
- Reset VAE tiling when it is disabled on a subsequent invocation.
- Explicit full-precision timestep modules and disabled autocast for that path,
  preserving the intended FP32 timestep math across PyTorch versions. This is
  a deliberate portability implementation, not a claim that upstream FP32 CUDA
  autocast always fails.
- Explicit DAC input dtype. The adapter loads DAC in FP32 for audio stability.
- Reject non-finite denoised latents and check decoded video in four-frame chunks
  before conversion to integer PIL images, which could otherwise hide NaNs.

The paired scheduler, normalization constants, expert boundary, CFG equations,
bridge interaction, reference conditioning, and sparse kernels remain upstream
implementations. We never use destructive `remove_video_dit=True`.

INT8 ConvRot uses Comfy-Org's wire schema and regular Hadamard convention, checked
against `comfy_kitchen.tensor.int8_utils`. The integration implementation is in
`prism/quantization.py`; the quantization recipe is deliberately limited to real
block Linear layers. Convolutional codecs, embeddings, final heads and timestep
projections retain BF16 file storage.

`prism/format.py` validates safetensors layout using the Rust reader, then uses
owned per-tensor reads during conversion and a read-only Python mmap during
inference. This bypasses a Windows access violation observed in PyTorch storage
slicing when safetensors opened the 60.83 GiB official preview. Inference tensors
retain their backing mapping after the reader exits and are never modified in
place. Dtype, byte-content and reader lifetime are covered by tests.

`prism/offload.py` additionally retains the original CPU storage of frozen UMT5,
Wan VAE and DAC modules. Returning them to CPU restores those tensors and their
shared parameter/buffer registrations, avoiding a new anonymous GPU-to-CPU weight
copy. Native config, dtype, encoding, decoding and tiling methods are delegated
unchanged. This applies to frozen encoders with either ordinary or FSDP-managed
transformers; the transformer sharding path is unchanged. Explicit dtype/layout
changes update the CPU snapshot. This is an inference storage optimization, not
evidence that full-model sampling performance or output quality has improved.

Block offload also follows the native fused block's selected video expert. Its
kwargs-aware prehook binds the actual forward signature, including positional
override calls. During a low-noise override it uploads the audio block and the
two conditioning bridges; the secondary video block uploads through its own
hook. The idle primary video block remains on CPU. The always-called posthook
still restores the complete fused block on success or interruption. High-noise
and ordinary video blocks keep their original transfer behavior.
