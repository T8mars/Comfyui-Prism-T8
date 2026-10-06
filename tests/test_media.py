import json
import shutil
import subprocess

import pytest
import torch

from prism.media import save_video


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="ffmpeg/ffprobe missing")
def test_mp4_has_both_tracks_and_no_overwrite(tmp_path):
    path = tmp_path / "joint.mp4"
    frames = torch.linspace(0., 1., 5).view(5, 1, 1, 1).expand(5, 16, 16, 3)
    waveform = torch.sin(torch.arange(10000).float() / 19).reshape(1, 1, -1) * 0.1
    save_video(frames, {"waveform": waveform, "sample_rate": 48000}, 24., path)
    data = json.loads(subprocess.check_output([shutil.which("ffprobe"), "-v", "error", "-show_streams", "-of", "json", str(path)]))
    assert {stream["codec_type"] for stream in data["streams"]} == {"video", "audio"}
    video = next(stream for stream in data["streams"] if stream["codec_type"] == "video")
    assert video["nb_frames"] == "5" and video["width"] == 16 and video["height"] == 16
    with pytest.raises(FileExistsError):
        save_video(frames, {"waveform": waveform, "sample_rate": 48000}, 24., path)


def test_bad_audio_never_writes_video(tmp_path):
    with pytest.raises(ValueError):
        save_video(torch.zeros(5, 16, 16, 3), {"waveform": torch.full((1, 1, 20), float("nan")), "sample_rate": 48000}, 24., tmp_path / "bad.mp4")
    assert not list(tmp_path.iterdir())


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="ffmpeg/ffprobe missing")
@pytest.mark.parametrize("audio_samples,fps", [(480, 24.), (480000, 24.), (480, 23.976)])
def test_audio_duration_does_not_truncate_video(tmp_path, audio_samples, fps):
    path = tmp_path / "aligned.mp4"
    frames = torch.linspace(0., 1., 49).view(49, 1, 1, 1).expand(49, 16, 16, 3)
    save_video(frames, {"waveform": torch.zeros(1, 1, audio_samples), "sample_rate": 48000}, fps, path)
    data = json.loads(subprocess.check_output([
        shutil.which("ffprobe"), "-v", "error", "-count_frames", "-show_streams", "-of", "json", str(path)
    ]))
    video = next(stream for stream in data["streams"] if stream["codec_type"] == "video")
    audio = next(stream for stream in data["streams"] if stream["codec_type"] == "audio")
    assert int(video["nb_read_frames"]) == len(frames)
    assert float(video["duration"]) == pytest.approx(len(frames) / fps, abs=1 / fps)
    # AAC ends on a codec packet boundary, within one packet of the video.
    assert float(audio["duration"]) == pytest.approx(float(video["duration"]), abs=1024 / 48000)
