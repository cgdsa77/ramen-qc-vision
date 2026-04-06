import pytest

from src.webapp import state


def test_prune_job_store_trims_oldest(monkeypatch):
    monkeypatch.setattr(state, "_MAX_BG_JOBS", 2)
    small = {"a": 1, "b": 2, "c": 3}
    state.prune_job_store(small)
    assert len(small) == 2
    assert "a" not in small


def test_resolve_async_mode(monkeypatch):
    monkeypatch.delenv("RAMEN_DEFAULT_ASYNC_DETECT", raising=False)
    assert state.resolve_async_mode(None) is False
    monkeypatch.setenv("RAMEN_DEFAULT_ASYNC_DETECT", "1")
    assert state.resolve_async_mode(None) is True
    assert state.resolve_async_mode(False) is False
