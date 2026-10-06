"""Distributed argument and setup failures must work without a CUDA launch."""
import pytest
import torch
import torch.distributed as dist

from prism.distributed import initialize
from prism.native.utils import parallel_states
from prism.settings import OFFICIAL_NEGATIVE_PROMPT
from scripts.sample_distributed import parse_args


def test_distributed_cli_negative_default_and_explicit_override():
    required = ["--models", "unused", "--reference", "unused"]
    assert parse_args(required).negative_prompt == OFFICIAL_NEGATIVE_PROMPT
    assert parse_args(required + ["--negative-prompt", "custom"]).negative_prompt == "custom"
    assert parse_args(required + ["--negative-prompt", ""]).negative_prompt == ""


def test_distributed_cli_rejects_zero_sp_size():
    with pytest.raises(SystemExit) as error:
        parse_args(["--models", "unused", "--reference", "unused", "--sp-size", "0"])
    assert error.value.code == 2


@pytest.mark.parametrize("sp_size", [0, -1, True, 1.5])
def test_invalid_sp_size_rejected_before_cuda_or_process_group(monkeypatch, sp_size):
    def forbidden(*args, **kwargs):
        pytest.fail("Invalid SP size must fail before CUDA or process-group setup")
    monkeypatch.setattr(torch.cuda, "set_device", forbidden)
    monkeypatch.setattr(dist, "init_process_group", forbidden)
    with pytest.raises(ValueError, match="positive integer"):
        initialize(sp_size)


def mock_setup(monkeypatch, world_size):
    events = []
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setattr(torch.cuda, "set_device", lambda rank: events.append(("device", rank)))
    monkeypatch.setattr(dist, "init_process_group", lambda backend: events.append(("init", backend)))
    monkeypatch.setattr(dist, "get_world_size", lambda: world_size)
    monkeypatch.setattr(dist, "destroy_process_group", lambda: events.append(("destroy",)))
    return events


def test_indivisible_world_size_cleans_initialized_group(monkeypatch):
    events = mock_setup(monkeypatch, 3)
    with pytest.raises(ValueError, match="divisible"):
        initialize(2)
    assert events == [("device", 0), ("init", "nccl"), ("destroy",)]


def test_native_parallel_setup_failure_cleans_group(monkeypatch):
    events = mock_setup(monkeypatch, 4)
    def failed(**kwargs):
        raise RuntimeError("native parallel setup failed")
    monkeypatch.setattr(parallel_states, "initialize_parallel_state", failed)
    with pytest.raises(RuntimeError, match="native parallel setup failed"):
        initialize(2)
    assert events[-1] == ("destroy",)


def test_successful_setup_retains_group_for_worker(monkeypatch):
    events = mock_setup(monkeypatch, 4)
    state = object()
    def successful(**kwargs):
        assert kwargs == {"sp": 2, "dp_replicate": 2}
        return state
    monkeypatch.setattr(parallel_states, "initialize_parallel_state", successful)
    device, result = initialize(2)
    assert device == torch.device("cuda", 0) and result is state
    assert events == [("device", 0), ("init", "nccl")]
