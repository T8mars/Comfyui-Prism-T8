# FreeVideo Prism acceleration

The `Prism FreeVideo Accelerated Sampler` uses the pinned FreeVideo Prism engine
with the existing seven standalone Prism components. The native sampler remains
available for the original pipeline and its full sparse-attention controls.

## Setup

Use a BF16-capable NVIDIA CUDA GPU, a current ComfyUI with `comfy-kitchen`, and
Triton matching its PyTorch environment. Windows uses `triton-windows`. Keep the
CUDA PyTorch already installed in ComfyUI. The tested environment uses Python
3.10, PyTorch 2.11/cu128 and triton-windows 3.6. The Triton 3.7 FP16-accumulation
attention extension is disabled; the baseline Sage kernels remain enabled.

Download the two **260412 rank-256** LightX2V Wan2.2 I2V LoRAs:

```bash
python scripts/download_acceleration.py
```

The helper writes two independent safetensors to this plugin's `models/loras/`,
which its loader discovers automatically. It verifies both published SHA256
checksums. Existing seven standalone model files are reused. No Diffusers
checkpoint directory or complete FreeVideo model download is required.

For a manual installation, use these files from
[Kijai/WanVideo_comfy](https://huggingface.co/Kijai/WanVideo_comfy/tree/8260d429d19fd7a72304cad059160b95d843913f/LoRAs/Wan22_Lightx2v)
in ComfyUI's `models/loras/`:

- `Wan_2_2_I2V_A14B_HIGH_lightx2v_4step_lora_260412_rank_256_fp16.safetensors`
- `Wan_2_2_I2V_A14B_LOW_lightx2v_4step_lora_260412_rank_256_fp16.safetensors`

## Canvas workflows

Import `examples/08_freevideo_high.json` and select the seven model files,
reference image, and both LoRAs. The example outputs frames with the canvas
embedded, 48 kHz FLAC, and an MP4 with audio and a canvas preview. The other
profiles have complete workflows `06` (Light), `07` (Standard), and `09` (Max).

New samplers and distributed workflows use the upstream reference geometry:
1280×720, 205 frames, 24 fps. Smaller and shorter clips are experimental.
Existing saved canvases keep their serialized values; edit their width, height
and frames explicitly when comparing.

Change the sampler's **quality** dropdown to switch profiles in the same canvas.
An example's node title is a label; the dropdown selects the actual recipe.
Light, Standard and High share the video recipe and differ in audio guidance.

| Profile | Video steps / CFG | Audio integration | Reference / LoRA |
| --- | --- | --- | --- |
| Light | 8; CFG 2 on first step | Student K/V cache; 4 audio substeps, CFG 5 | Required / required |
| Standard | 8; CFG 2 on first step | Partial base teacher, layers 10–29; 4 substeps | Required / required |
| High | 8; CFG 2 on first step | Full base teacher; 4 substeps | Required / required |
| Max | 20; CFG 5 on all steps | Joint video/audio CFG | Optional / unused |

Independent video and audio prompts, seed, resolution, frame count and FPS are
supported. An empty audio prompt uses the video prompt. Dimensions must be
multiples of 16; frames must satisfy `(frames - 1) % 4 == 0`. Leave the image
disconnected in Max for FreeVideo's text-only path. Light/Standard/High require
an image. First-block step caching is disabled in every profile.

The upstream comparison uses **1280×720, 205 frames, 24 fps**. Smaller or shorter
clips, including the 768×432 examples, are experimental and do not establish
audio parity. For a Mandarin comparison use a Chinese scene prompt with
`<speech>中文台词</speech>` and leave `audio_prompt` empty to reuse that exact
prompt for audio, as upstream does.

## Memory and first-run preparation

The sampler runs in a separate worker with configurable GPU and host-memory
budgets. Defaults are **18 GiB VRAM / 20 GiB RAM**. Placement uses live memory,
streams inactive blocks and releases the high expert before loading the low
expert. Cancellation terminates that worker. The worker stops if available
system RAM falls below 6 GiB; its private state is released when it exits.
On Windows, the RAM guard measures the private working set rather than mapped
weight pages in total RSS. Under memory pressure it first releases its own
pageable residency, while preserving the 6 GiB emergency stop.

The first run prepares a disposable streaming cache in this plugin's
`models/.prism-acceleration-cache/`. With alpha and the two LoRAs it occupies
**36.30 GiB**, in addition to the seven input files and approximately 4.65 GiB
of LoRAs. The cache has internal block files; users continue selecting the
independent input models. Preparation reads and transforms one unit at a time,
uses CPU only, and writes its completion manifest atomically. Subsequent runs
reuse it. Changing an input file creates a new cache. Allow extra temporary
disk space during preparation and remove obsolete cache directories when needed.

The ConvRot-to-FreeVideo adapter changes only INT8 column ordering and signs.
It preserves the reconstructed quantized weights exactly, without BF16
re-quantization. FreeVideo's dynamic activation quantization, sparse attention
and distilled sampling still change inference behavior and are not numerically
identical to the native sampler.

This adapter preserves the selected ConvRot model. It does not reproduce the
published FreeVideo bundle's randomized Had128/MSE weight quantization. Using
the same LoRAs and quality profile alone does not establish audio parity with
that bundle.

The published FreeVideo INT8 bundle uses **BF16 UMT5**, separately from its
INT8 diffusion weights. Select a converted standalone BF16 text encoder for
a source-precision comparison. The smaller standalone INT8 ConvRot encoder
is supported with floating-point activations (W8A16), but produces different
conditioning. The worker receipt records the actual selected precision.
The native sampler also keeps floating activations in audio and bridge layers;
its Kitchen selection applies to the video towers. The accelerated worker keeps
FreeVideo's W8A8 diffusion recipe. Both samplers invalidate cached ComfyUI results
when their inference implementation changes.

Kernel initialization follows the pinned upstream worker: automatic dynamic
shapes are disabled, the Sage route is patched, and fully padded query tiles are
skipped. The worker receipt records the actual initialization settings.

FreeVideo's advertised **14.6×** result was measured on H200 at 1280×720,
205 frames, against its 50-step Original recipe and excludes loading. It is
not a speed guarantee for this integration or a laptop GPU. See the
[pinned upstream documentation](https://github.com/FlashML-org/FreeVideo/blob/40525196a33bc7ff6a455ce9b18b74f3c827bbb1/docs/Prism.md).

## Validation and known limits

The integration passes 185 regression tests and validation of all nine canvas
workflows. Real INT8 Light and High samples were generated; native multi-step
samples and the updated High kernel initialization were also exercised through
the public runtime. Video frames and audio tracks were fully decoded. Standard
and Max do not yet have local end-to-end samples; beta and multi-GPU quality
coverage remain incomplete.

**Audio can degrade for some seeds, reference images and prompts.** The user and
FreeVideo author reproduced this input-dependent limitation. Native INT8 and
accelerated outputs both require listening checks. ASR, finite waveforms, no
clipping and automated quality scores do not establish undistorted speech or
lip sync. A controlled High run with all 223 verified published core/LoRA files
also did not establish acceptable audio. That private diagnostic used the same
standalone BF16 text encoder and codecs; the public loader still accepts the
standalone ConvRot components described above.

Representative timings on an RTX 5090 Laptop 24 GB, at 24 fps:

| Path | Settings | Inference time | Scope |
| --- | --- | --- | --- |
| Native INT8, BF16 text | 768×432, 49 frames, 32 steps | 1185.17 s | Current native audio/bridge activation routing |
| High INT8, BF16 text | 768×432, 121 frames, 8 steps | 311.80 s | Updated kernel initialization |
| High INT8, INT8 text | 1280×720, 205 frames, 8 steps | 2210.40 s | Earlier reference-size control |

Timings include encoding, sampling and decoding; downloads, initial cache
preparation and ComfyUI output saving are excluded. The reference-size control
used 18/20 GiB budgets, with sampled CUDA allocation/reservation peaks of
15.87/16.49 GiB and worker RSS of 23.92 GiB. RAM is a placement budget, not a
strict RSS ceiling. These cases do not guarantee other inputs' speed or quality.

## Provenance

Engine: [FlashML-org/FreeVideo `40525196`](https://github.com/FlashML-org/FreeVideo/tree/40525196a33bc7ff6a455ce9b18b74f3c827bbb1).
Vendored code and checksums are under `prism/acceleration/vendor/`; reproduce
them using `scripts/vendor_accel.py`. Local adaptations add independent audio
prompt embeddings, propagate FPS into student and teacher passes, and select
Sage attention by default in the accelerated namespace. They do not alter
`prism/native/`. Source hashes and rendered hashes are recorded separately.
The local initialization wrapper is adapted from the same commit's
`prism_worker.py::kernel_setup`.

FreeVideo changes are Apache-2.0. Retained upstream Prism/MOVA/Wan/DAC notices
and license terms are in `vendor/NOTICE` and `vendor/LICENSE.Apache-2.0`.
The upstream repository's third-party attribution is retained in
`vendor/THIRD_PARTY_NOTICES.md`.
LightX2V LoRAs retain their upstream Apache-2.0 terms.
