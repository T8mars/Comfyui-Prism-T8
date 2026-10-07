# Prism T8 · ComfyUI

[腾讯 Prism](https://github.com/Tencent-Hunyuan/Prism) 原生视频与音频联合生成节点。保留官方双塔模型、跨模态桥接和配对调度器，提供七个独立组件加载器、INT8 ConvRot 权重及完整 ComfyUI 画布工作流。

[模型下载](https://huggingface.co/t8star/Prism-Comfy/tree/main) · [画布工作流](https://github.com/T8mars/Comfyui-Prism-T8/blob/main/examples/Prism-canvas-workflows.zip) · [真实 480p 样片](https://github.com/T8mars/Comfyui-Prism-T8/blob/main/examples/sample-480p.mp4)

## FreeVideo 加速

新增独立 **FreeVideo Accelerated Sampler**，提供 Light、Standard、High、Max 四档。Light 使用 8 步蒸馏视频采样和每步 4 次音频补偿，保留独立视频／音频提示词。现有七个独立 INT8 ConvRot／VAE 文件继续使用，Light/Standard/High 另需两份独立的 260412 rank-256 高／低噪声 LoRA。

```bash
python scripts/download_acceleration.py
```

重启后导入 [08 · FreeVideo High](examples/08_freevideo_high.json)，选择参考图和模型即可。加速需要兼容的 CUDA、Triton 与 comfy-kitchen。默认预算为 18 GiB 显存／20 GiB 内存；首次自动创建约 36.30 GiB 私有流式缓存，后续复用。四档配方、安装和资源说明见 [ACCELERATION.md](ACCELERATION.md)。官方约 15 倍速度来自 H200 对照，本机速度以实测为准。

**原模型音频限制（非本节点问题）：** 用户与 FreeVideo 作者的联合测试已确认，Prism 原模型自身在特定提示词、种子及参考图组合下会出现音频劣化。这类已复现的退化属于原模型的音频生成限制，并非本 ComfyUI 节点引入。生成后请实际试听音轨。

主工作流采用参考规格 **1280×720、205 帧、24 fps**；低尺寸、短时长属于实验配置。FreeVideo 发布包使用 BF16 UMT5，严格对照时应选择独立 BF16 文本编码器；INT8 文本仍受支持，但条件编码不同。

## 安装

Comfy Registry：[t8star/prism-t8](https://registry.comfy.org/t8star/prism-t8)，版本可用性以平台审核状态为准。也可手动安装：

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/T8mars/Comfyui-Prism-T8.git
cd Comfyui-Prism-T8
```

使用运行 ComfyUI 的 Python 安装依赖，然后重启：

```bash
python -m pip install -r requirements.txt
```

需要 Python 3.10+、支持 BF16 的 NVIDIA CUDA 环境；保留 ComfyUI 已安装的 CUDA PyTorch。保存带音频的 MP4 需要 PATH 中的 **ffmpeg**。默认 SDPA 不需要 Triton；启用原生 BSA 时另需与 PyTorch 匹配的 Triton，Windows 使用 `triton-windows`。

## 模型

从 [t8star/Prism-Comfy](https://huggingface.co/t8star/Prism-Comfy) 下载以下 **全部七个文件**，放入对应的 ComfyUI 模型目录。完整 alpha 模型约 **38.13 GiB**。

| 文件 | 目录 |
| --- | --- |
| `prism_alpha_video_dit_int8_convrot.safetensors` | `models/diffusion_models/` |
| `prism_alpha_video_dit_2_int8_convrot.safetensors` | `models/diffusion_models/` |
| `prism_alpha_audio_dit_int8_convrot.safetensors` | `models/diffusion_models/` |
| `prism_alpha_dual_tower_bridge_int8_convrot.safetensors` | `models/diffusion_models/` |
| `prism_alpha_text_encoder_int8_convrot.safetensors` | `models/text_encoders/` |
| `prism_alpha_video_vae_bf16.safetensors` | `models/vae/` |
| `prism_alpha_audio_vae_bf16.safetensors` | `models/vae/` |

文本编码器为 **UMT5**，视频去噪器为双 **DiT**。配置与 tokenizer 已嵌入独立文件，使用本插件加载器，无需 Diffusers 权重目录。`diffusers` 库仅作为原生模型类的代码依赖。

ConvRot 量化用于 block Linear；embedding、norm、时间投影、输出头与 VAE 保留浮点精度。音频 VAE 以 BF16 保存、FP32 执行。七文件必须来自同一 bundle；加载器会检查组件、形状、完整性与量化标记。

## 工作流

下载 [完整工作流包](https://github.com/T8mars/Comfyui-Prism-T8/blob/main/examples/Prism-canvas-workflows.zip)，将 JSON 拖入 ComfyUI 画布。把附带的 `prism_official_case5.png` 放入 ComfyUI `input/`，或在 **Load Image** 上传自己的参考图，再选择七个模型文件并运行。

| 工作流 | 用途 |
| --- | --- |
| [01 · I2VA](examples/01_native_i2va.json) | portable INT8 + SDPA，1280×720、205 帧、50 步 |
| [02 · Kitchen + BSA](examples/02_native_i2va_kitchen_bsa.json) | 视频 W8A8、原生 video/v2a BSA 与 IVPQ |
| [03 · 白帧 T2VA](examples/03_native_t2va_white_reference.json) | 官方白色首帧条件实验模式 |
| [04 · 720p](examples/04_native_i2va_720p.json) | 1280×720、205 帧、VAE tiling 参数预设 |
| [05 · Kitchen 对照](examples/05_native_i2va_validation.json) | 848×480、49 帧接口验证；不作为音质基准 |
| [06 · FreeVideo Light](examples/06_freevideo_light.json) | 8 步蒸馏，学生 K/V 音频补偿 |
| [07 · FreeVideo Standard](examples/07_freevideo_standard.json) | 8 步蒸馏，部分音频教师 |
| [08 · FreeVideo High](examples/08_freevideo_high.json) | 8 步蒸馏，完整音频教师 |
| [09 · FreeVideo Max](examples/09_freevideo_max.json) | 20 步基础模型，联合 CFG；无需蒸馏 LoRA |

每份都是包含节点位置、分组、参数与连线的**画布格式**，输出 PNG 帧、48 kHz FLAC、H.264/AAC MP4 及画布视频预览。MP4 保留全部视频帧，较短音轨补静音、较长音轨裁到视频结尾。详细导入说明见 [examples/README.md](examples/README.md)。

支持独立视频／音频提示词、`<music>` / `<sfx>` / `<speech>` 标签、CFG、seed、视觉／音频 shift、分块或整组件 CPU 卸载，以及原生稀疏注意力参数。高级参数见 [sparse_options.json](examples/sparse_options.json)。分辨率为 16 的倍数；帧数至少 5，满足 `(frames-1)%4==0`。BSA 的三维块各轴为 2 的幂，K 块至少 16 tokens；v2a 音频块为不小于 64 的 2 的幂。

## 运行与验证

原生采样的 `int8_backend` 控制视频主干：`portable` 使用 W8A16，`kitchen` 使用动态 W8A8。原生音频主干、桥接层和文本编码均保留浮点激活；扩散组件的独立 INT8 ConvRot 权重不变。文本加载器支持独立 BF16 UMT5，也支持较小的 INT8 版本。FreeVideo 发布包使用 BF16 UMT5；严格对照应选择该精度，INT8 文本并非同精度复现。分块卸载的速度受 CPU 内存与 PCIe 影响；INT8 不减少高分辨率激活占用。

真实 alpha INT8 样片已完成 **848×480、49 帧、50 步**生成与全帧画面检查，并完整解码声画轨；音频尚未试听。RTX 5090 Laptop 24 GB、分块卸载配置耗时约 47 分钟，PyTorch 峰值分配显存约 7.76 GiB。存在轻微构图漂移与细纹理偏软，量化不保证无损。

已通过 185 项回归测试及九份画布工作流校验；真实生成、完整解码与台词识别仅验证执行和可识别性，不能证明音质正常。历史 480p 样片不对应现在的 720p 默认配置；beta、多卡及各模式的完整质量对照尚未完成。

## 自行转换

```bash
python scripts/download_models.py --output checkpoints/official --variant alpha
python scripts/convert_models.py --base checkpoints/official/pretrained_models/MOVA-360p --preview checkpoints/official/preview_alpha/diffusion_pytorch_model.safetensors --output models/standalone --variant alpha --device cuda:0
```

转换输出也会被插件自动发现。源权重约 72.35 GiB，转换需额外预留最终模型及一个最大组件的临时空间。支持 `--variant beta`、`--dry-run` 与 `--components`。保持源权重不变时，可加 `--resume` 校验并复用已完成文件；`scripts/prepare_models.py` 会验证现有组件、恢复缺失清单并接续转换。已有完整七组件时，准备脚本无需源权重或联网。默认不覆盖文件，切换配方请选新输出目录。量化配方与文件校验见 Hugging Face 模型仓库。

## 来源与许可

原生源码固定于腾讯 Prism [`883e90a5`](https://github.com/Tencent-Hunyuan/Prism/tree/883e90a5c90dc8b7044c65eba0bb64e9342cb46a)，改动见 [NATIVE_CHANGES.md](NATIVE_CHANGES.md)。保留原始 [LICENSE](LICENSE) 与第三方归属声明：Prism 使用 MIT，第三方组件遵循各自许可。本项目是社区 ComfyUI 集成。

## T8star

[B站](https://space.bilibili.com/385085361) · [YouTube](https://www.youtube.com/@T8star-Aix/) · [API](https://api.seedance.nz/sign-up?aff=5f4w) · [免费画廊](https://www.openzhenzhen.com) · [在线 AI 应用](https://www.runninghub.ai/zh-cn/user-center/1907375370302308353/userPost?inviteCode=rh-v1121) · [ComfyUI 整合包](https://pan.quark.cn/s/264edb7e36bd) · [Hugging Face](https://huggingface.co/t8star)
