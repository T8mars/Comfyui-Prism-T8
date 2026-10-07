from types import SimpleNamespace

import pytest

from prism.acceleration import memory


def observation(available, rss=24, private=4):
    return dict(available_bytes=int(available * memory.GIB), total_bytes=64 * memory.GIB,
                rss_bytes=rss * memory.GIB, guard_bytes=private * memory.GIB,
                guard_metric='private working set', reclaimable_mapped_bytes=(rss - private) * memory.GIB)


def test_windows_reclaims_mapped_pages_before_emergency(monkeypatch):
    values = iter([observation(5.65), observation(28, rss=4)])
    monkeypatch.setattr(memory, 'WINDOWS', True)
    monkeypatch.setattr(memory, 'snapshot', lambda _: next(values))
    trims = []
    monkeypatch.setattr(memory, 'trim_current_working_set', lambda: trims.append(True) or True)
    events = []
    actual = memory.check(object(), 20, lambda **row: events.append(row))
    assert trims == [True] and actual['available_bytes'] == 28 * memory.GIB
    assert events[0]['event'] == 'working_set_reclaim'


def test_failed_reclaim_keeps_emergency_floor(monkeypatch):
    monkeypatch.setattr(memory, 'WINDOWS', True)
    monkeypatch.setattr(memory, 'snapshot', lambda _: observation(5))
    monkeypatch.setattr(memory, 'trim_current_working_set', lambda: False)
    with pytest.raises(RuntimeError, match='less than 6 GiB'):
        memory.check(object(), 20, lambda **_: None)


def test_mapped_rss_is_not_private_ram_budget(monkeypatch):
    monkeypatch.setattr(memory, 'snapshot', lambda _: observation(20, rss=40))
    assert memory.check(object(), 20, lambda **_: None)['guard_bytes'] == 4 * memory.GIB
    monkeypatch.setattr(memory, 'snapshot', lambda _: observation(20, rss=40, private=27))
    with pytest.raises(RuntimeError, match='host-memory budget'):
        memory.check(object(), 20, lambda **_: None)


def test_nonwindows_keeps_rss_guard_and_does_not_trim(monkeypatch):
    monkeypatch.setattr(memory, 'WINDOWS', False)
    monkeypatch.setattr(memory.psutil, 'virtual_memory', lambda: SimpleNamespace(available=5 * memory.GIB, total=64 * memory.GIB))
    process = SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=3 * memory.GIB))
    row = memory.snapshot(process)
    assert row['guard_bytes'] == row['rss_bytes'] == 3 * memory.GIB
    assert row['guard_metric'] == 'RSS' and row['reclaimable_mapped_bytes'] == 0
    with pytest.raises(RuntimeError, match='less than 6 GiB'):
        memory.check(process, 20, lambda **_: None)
