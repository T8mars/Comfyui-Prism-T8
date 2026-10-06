# Prism 完整画布工作流

这些文件是 ComfyUI `version: 0.4` 画布 JSON，包含节点位置、连线和参数，可拖进画布或从工作流菜单打开。

| 文件 | 用途 |
| --- | --- |
| `01_native_i2va.json` | 参考图联合生成视频和音频，INT8 portable、dense SDPA；848×480 实际样片画面已复核 |
| `02_native_i2va_kitchen_bsa.json` | Kitchen INT8，原生 video/v2a BSA、IVPQ 动态分块 |
| `03_native_t2va_white_reference.json` | 官方白图条件的文本生成路径，首帧为白图 |
| `04_native_i2va_720p.json` | 1280×720、205 帧、50 步、VAE 切片预设；本机尚未验收该规格 |
| `05_native_i2va_validation.json` | Kitchen INT8、dense SDPA，同种子对照 |

1. 按项目根目录 README 安装插件和依赖；保持七个独立模型文件来自同一转换 bundle。
2. 将 `prism_official_case5.png` 复制到 ComfyUI 的 `input/`，或在 `Load Image` 中上传自己的参考图。附带图片原样来自固定版本的腾讯 Prism 官方示例。
3. 导入所需 JSON，检查七个模型下拉框，然后点击“运行”。01/02/03/05 默认 848×480、49 帧、24 fps、50 步，种子固定为 42。
4. 完整输出包括逐帧 PNG（内嵌画布工作流）、48 kHz FLAC、H.264/AAC MP4 及画布视频预览。输出位于 ComfyUI `output/Prism/<工作流名>/`。

负面提示词使用官方单图推理入口的默认中文设置；可以在采样节点中修改。
`sparse_options.json` 和 `validation_sparse_options.json` 是命令行参数文件，不是画布工作流。
01 相同参数的真实 480p 样片已完成全部 49 帧画面复核；仍有轻微构图漂移和细纹理偏软。音轨已完整解码，尚未试听。
02/03/05 的完整 480p 样片及 04 的 720p 长视频尚未验收。建议先使用 01 的默认设置。
