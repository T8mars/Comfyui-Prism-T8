# Prism T8 · ComfyUI

[腾讯 Prism](https://github.com/Tencent-Hunyuan/Prism) 原生视频与音频联合生成节点。保留官方双塔模型、跨模态桥接和配对调度器，提供七个独立组件加载器、INT8 ConvRot 权重及完整 ComfyUI 画布工作流。

[模型下载](https://huggingface.co/t8star/Prism-Comfy/tree/main) · [画布工作流](https://github.com/T8mars/Comfyui-Prism-T8/blob/main/examples/Prism-canvas-workflows.zip) · [真实 480p 样片](https://github.com/T8mars/Comfyui-Prism-T8/blob/main/examples/sample-480p.mp4)

## 安装

已提交 Comfy Registry（`t8star/prism-t8`），版本仍需平台审查。可先手动安装：

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
| [01 · I2VA](examples/01_native_i2va.json) | 推荐起点：portable INT8 + SDPA，848×480、49 帧、50 步 |
| [02 · Kitchen + BSA](examples/02_native_i2va_kitchen_bsa.json) | W8A8、原生 video/v2a BSA 与 IVPQ |
| [03 · 白帧 T2VA](examples/03_native_t2va_white_reference.json) | 官方白色首帧条件实验模式 |
| [04 · 720p](examples/04_native_i2va_720p.json) | 1280×720、205 帧、VAE tiling 参数预设 |
| [05 · Kitchen 对照](examples/05_native_i2va_validation.json) | Kitchen INT8 + dense SDPA |

每份都是包含节点位置、分组、参数与连线的**画布格式**，输出 PNG 帧、48 kHz FLAC、H.264/AAC MP4 及画布视频预览。MP4 保留全部视频帧，较短音轨补静音、较长音轨裁到视频结尾。详细导入说明见 [examples/README.md](examples/README.md)。

支持独立视频／音频提示词、`<music>` / `<sfx>` / `<speech>` 标签、CFG、seed、视觉／音频 shift、分块或整组件 CPU 卸载，以及原生稀疏注意力参数。高级参数见 [sparse_options.json](examples/sparse_options.json)。分辨率为 16 的倍数；帧数至少 5，满足 `(frames-1)%4==0`。BSA 的三维块各轴为 2 的幂，K 块至少 16 tokens；v2a 音频块为不小于 64 的 2 的幂。

## 运行与验证

默认 `portable` 是 W8A16 旋转与临时反量化路径；`kitchen` 使用 `comfy_kitchen.int8_linear` 执行动态 W8A8。分块卸载可以降低显存占用，速度受 CPU 内存与 PCIe 影响；INT8 不减少高分辨率激活占用。

真实 alpha INT8 样片已完成 **848×480、49 帧、50 步**生成与全帧画面检查，并完整解码声画轨；音频尚未试听。RTX 5090 Laptop 24 GB、分块卸载配置耗时约 47 分钟，PyTorch 峰值分配显存约 7.76 GiB。存在轻微构图漂移与细纹理偏软，量化不保证无损。

五份画布已实际导入和保存，回归测试全部通过。02/03/05 的完整 480p 样片、720p 长视频、beta 与多卡尚未完成实样验收；320×192 样片画质不佳，建议先使用 01 的默认设置。

## 自行转换

```bash
python scripts/download_models.py --output checkpoints/official --variant alpha
python scripts/convert_models.py --base checkpoints/official/pretrained_models/MOVA-360p --preview checkpoints/official/preview_alpha/diffusion_pytorch_model.safetensors --output models/standalone --variant alpha --device cuda:0
```

转换输出也会被插件自动发现。源权重约 72.35 GiB，转换需额外预留最终模型及一个最大组件的临时空间。支持 `--variant beta`、`--dry-run` 与 `--components`。保持源权重不变时，可加 `--resume` 校验并复用已完成文件；`scripts/prepare_models.py` 会验证现有组件、恢复缺失清单并接续转换。默认不覆盖文件，切换配方请选新输出目录。量化配方与文件校验见 Hugging Face 模型仓库。

## 来源与许可

原生源码固定于腾讯 Prism [`883e90a5`](https://github.com/Tencent-Hunyuan/Prism/tree/883e90a5c90dc8b7044c65eba0bb64e9342cb46a)，改动见 [NATIVE_CHANGES.md](NATIVE_CHANGES.md)。保留原始 [LICENSE](LICENSE) 与第三方归属声明：Prism 使用 MIT，第三方组件遵循各自许可。本项目是社区 ComfyUI 集成。

## T8star

[B站](https://space.bilibili.com/385085361) · [YouTube](https://www.youtube.com/@T8star-Aix/) · [API](https://api.seedance.nz/sign-up?aff=5f4w) · [免费画廊](https://www.openzhenzhen.com) · [在线 AI 应用](https://www.runninghub.ai/zh-cn/user-center/1907375370302308353/userPost?inviteCode=rh-v1121) · [ComfyUI 整合包](https://pan.quark.cn/s/264edb7e36bd) · [Hugging Face](https://huggingface.co/t8star)
