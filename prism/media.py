"""Atomic MP4 output with joint audio; mux failures always surface."""
from pathlib import Path
import math
import shutil
import subprocess
import tempfile
import wave

import torch


def save_video(frames, audio, fps, path, interrupt=None):
    executable = shutil.which("ffmpeg")
    if not executable:
        raise RuntimeError("ffmpeg must be installed and available on PATH to save Prism video/audio")
    if frames.ndim != 4 or frames.shape[-1] != 3 or frames.shape[0] < 1:
        raise ValueError("Expected RGB IMAGE frames [T,H,W,3]")
    if type(fps) not in (float, int) or not math.isfinite(fps) or fps <= 0 or not torch.isfinite(frames).all():
        raise ValueError("Invalid frame rate or non-finite frames")
    waveform = audio["waveform"]
    rate = audio["sample_rate"]
    if waveform.ndim != 3 or waveform.shape[0] != 1 or waveform.shape[1] not in (1, 2):
        raise ValueError("Expected one mono/stereo AUDIO batch")
    if type(rate) is not int or rate <= 0 or waveform.shape[-1] < 1 or not torch.isfinite(waveform).all():
        raise ValueError("Invalid audio")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    count, height, width, _ = frames.shape
    with tempfile.TemporaryDirectory(prefix=".prism_mux_", dir=path.parent) as folder:
        folder = Path(folder)
        pcm = (waveform[0].detach().cpu().clamp(-1, 1).transpose(0, 1).numpy() * 32767).astype("int16")
        with wave.open(str(folder / "audio.wav"), "wb") as stream:
            stream.setnchannels(pcm.shape[1])
            stream.setsampwidth(2)
            stream.setframerate(rate)
            stream.writeframes(pcm.tobytes())
        temporary = folder / "video.mp4"
        command = [executable, "-hide_banner", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                   "-s", f"{width}x{height}", "-r", str(fps), "-i", "pipe:0", "-i", str(folder / "audio.wav"),
                   "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
                   "-movflags", "+faststart", "-shortest", str(temporary)]
        with open(folder / "ffmpeg.log", "wb") as log:
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=log)
            try:
                for frame in frames:
                    if interrupt:
                        interrupt()
                    pixels = (frame.detach().cpu().clamp(0, 1).numpy() * 255).round().astype("uint8")
                    process.stdin.write(pixels.tobytes())
                process.stdin.close()
                while True:
                    if interrupt:
                        interrupt()
                    try:
                        code = process.wait(timeout=0.1)
                        break
                    except subprocess.TimeoutExpired:
                        pass
                if code:
                    raise RuntimeError(f"ffmpeg failed ({code}): {(folder / 'ffmpeg.log').read_text(errors='replace')[-2000:]}")
            except BaseException:
                if process.poll() is None:
                    process.kill()
                    process.wait()
                raise
        if interrupt:
            interrupt()
        import os
        os.link(temporary, path)
    return str(path)
