"""Review real completed ComfyUI jobs by fully decoding their saved MP4s."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import urllib.request

import numpy as np
from PIL import Image, ImageDraw


def decode(path, expected, target):
    probe = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-show_streams",
        "-show_format", "-of", "json", str(path)]))
    video = next(s for s in probe["streams"] if s["codec_type"] == "video")
    audio = next(s for s in probe["streams"] if s["codec_type"] == "audio")
    width, height = video["width"], video["height"]
    raw = subprocess.check_output(["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:v:0",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"])
    frames = np.frombuffer(raw, dtype=np.uint8).reshape(-1, height, width, 3)
    wav = subprocess.check_output(["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:a:0",
        "-acodec", "pcm_f32le", "-f", "f32le", "pipe:1"])
    waveform = np.frombuffer(wav, dtype="<f4")
    assert len(frames) == expected["num_frames"]
    assert (width, height) == (expected["width"], expected["height"])
    assert int(audio["sample_rate"]) == 48000 and np.isfinite(waveform).all()
    expected_duration = len(frames) / expected["fps"]
    assert abs(float(video["duration"]) - expected_duration) < 1 / expected["fps"]
    assert abs(float(audio["duration"]) - expected_duration) < 0.06
    target.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(path), "-map", "0:a:0",
                    "-acodec", "pcm_s16le", str(target / "audio.wav")], check=True)
    indices = np.linspace(0, len(frames) - 1, 9, dtype=int).tolist()
    sheet = Image.new("RGB", (width * 3, (height + 24) * 3), "#202020")
    draw = ImageDraw.Draw(sheet)
    for index, frame_id in enumerate(indices):
        x, y = index % 3 * width, index // 3 * (height + 24)
        im = Image.fromarray(frames[frame_id])
        im.save(target / f"frame_{frame_id:04d}.png")
        sheet.paste(im, (x, y))
        draw.text((x + 8, y + height + 3), f"frame {frame_id} / {frame_id / expected['fps']:.3f}s", fill="white")
    sheet.save(target / "contact-sheet.jpg", quality=95)
    values = waveform.astype(np.float64)
    return {"file": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "full_video_decode": True, "full_audio_decode": True, "frames": len(frames),
        "width": width, "height": height, "video_codec": video["codec_name"],
        "audio_codec": audio["codec_name"], "sample_rate": int(audio["sample_rate"]),
        "duration": float(video["duration"]), "audio_duration": float(audio["duration"]),
        "audio_rms": float(np.sqrt(np.mean(values ** 2))), "audio_peak": float(np.abs(values).max()),
        "audio_abs_ge_0_99_fraction": float(np.mean(np.abs(values) >= .99)),
        "frame_std": float(frames.astype(np.float32).std()),
        "mean_frame_difference": float(np.abs(np.diff(frames.astype(np.float32), axis=0)).mean()),
        "contact_sheet": str((target / "contact-sheet.jpg").resolve()),
        "audio_wav": str((target / "audio.wav").resolve()),
        "semantic_quality": "requires separate visual/audio inspection; decode metrics alone are insufficient"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", default="http://127.0.0.1:8198")
    parser.add_argument("--output-root", default="outputs/comfyui-review/output")
    parser.add_argument("--report-root", default="outputs/comfyui-review/reports")
    args = parser.parse_args()
    with urllib.request.urlopen(args.server.rstrip("/") + "/history") as response:
        fresh_history = json.load(response)
    root = Path(args.report_root)
    root.mkdir(parents=True, exist_ok=True)
    history_file = root / "history.json"
    history = json.loads(history_file.read_text(encoding="utf-8")) if history_file.exists() else {}
    history.update(fresh_history)
    (root / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    jobs = []
    for prompt_id, job in history.items():
        samplers = {key: n for key, n in job["prompt"][2].items() if n["class_type"] == "PrismNativeSampler"}
        if not samplers:
            continue
        inputs = next(iter(samplers.values()))["inputs"]
        result = {"prompt_id": prompt_id, "status": job["status"], "sampler": inputs,
                  "workflow_embedded": "workflow" in job["prompt"][3].get("extra_pnginfo", {})}
        if job["status"]["status_str"] == "success":
            videos = [entry for output in job["outputs"].values() for entry in output.get("images", [])
                      if entry["filename"].endswith(".mp4")]
            assert len(videos) == 1
            entry = videos[0]
            path = Path(args.output_root) / entry["subfolder"] / entry["filename"]
            result["media"] = decode(path, inputs, root / prompt_id)
        jobs.append(result)
    report = {"scope": "real frontend queued jobs; full MP4 decode, frame count, duration and audio statistics",
              "jobs": jobs}
    (root / "sample-review.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
