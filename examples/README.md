# Prism 完整画布工作流

这些文件是 ComfyUI `version: 0.4` 画布 JSON，包含节点位置、连线和参数，可拖进画布或从工作流菜单打开。

| 文件 | 用途 |
| --- | --- |
| `01_native_i2va.json` | 720p / 205 帧参考图联合生成，INT8 portable、dense SDPA |
| `02_native_i2va_kitchen_bsa.json` | Kitchen INT8，原生 video/v2a BSA、IVPQ 动态分块 |
| `03_native_t2va_white_reference.json` | 官方白图条件的文本生成路径，首帧为白图 |
| `04_native_i2va_720p.json` | 1280×720、205 帧、50 步、VAE 切片预设；本机尚未验收该规格 |
| `05_native_i2va_validation.json` | Kitchen INT8、dense SDPA，480p 短片接口对照，不作为音质基准 |
| `06_freevideo_light.json` | FreeVideo Light，8 步蒸馏和学生 K/V 音频补偿 |
| `07_freevideo_standard.json` | FreeVideo Standard，8 步蒸馏和部分音频教师 |
| `08_freevideo_high.json` | FreeVideo High，8 步蒸馏和完整音频教师 |
| `09_freevideo_max.json` | FreeVideo Max，20 步基础模型，无需蒸馏 LoRA |

1. 按项目根目录 README 安装插件和依赖；保持七个独立模型文件来自同一转换 bundle。
2. 将 `prism_official_case5.png` 复制到 ComfyUI 的 `input/`，或在 `Load Image` 中上传自己的参考图。附带图片原样来自固定版本的腾讯 Prism 官方示例。
3. 导入所需 JSON，检查七个模型下拉框，然后点击“运行”。01/02/03/04 使用 1280×720、205 帧、24 fps、50 步，种子固定为 42；05 保留 848×480、49 帧的实验接口对照。
4. 完整输出包括逐帧 PNG（内嵌画布工作流）、48 kHz FLAC、H.264/AAC MP4 及画布视频预览。输出位于 ComfyUI `output/Prism/<工作流名>/`。

负面提示词使用官方单图推理入口的默认中文设置；可以在采样节点中修改。
FreeVideo 的 06–09 使用上游参考规格 1280×720、205 帧、24 fps，默认资源预算 18 GiB 显存／20 GiB 内存。06–08 连接两份 260412 rank-256 LoRA；09 使用基础模型，可断开图片输入启用文本生成。首次会准备私有缓存。完整说明见 [ACCELERATION.md](../ACCELERATION.md)。
在加速采样节点的 `quality` 下拉框切换 Light / Standard / High / Max，实际模式以该参数为准，节点标题只是标签。前三档使用相同视频配方，区别是音频引导。旧版小分辨率、短时长预设属于实验规格，不能代表对白复现已通过。中文对白对照使用中文场景提示词，将台词置于 `<speech>` 标签内，并留空 `audio_prompt` 以复用完整场景提示词。
`sparse_options.json` 和 `validation_sparse_options.json` 是命令行参数文件，不是画布工作流。
部分种子、图片和提示词组合会出现音频退化，原生 INT8 与加速输出均需实际试听。解码、识别或画布校验通过不能证明音质正常。严格对照 FreeVideo 发布包时，文本加载器应选择独立 BF16 UMT5；这些画布默认保留 INT8 文本选项，需要手动更换。原生 Kitchen 设置控制视频主干，音频和桥接层保留浮点激活。历史 480p 样片不对应当前默认配置；已有旧画布保留旧参数，需要自行修改尺寸和帧数。
