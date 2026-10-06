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
