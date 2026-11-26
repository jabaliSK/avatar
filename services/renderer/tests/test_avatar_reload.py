"""Runtime.get_avatar keeps one avatar in memory and loads it again once it has been onboarded again on disk."""
import json
import os

import pytest

from app import server
from app.avatar_store import avatar_stamp


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    monkeypatch.setattr(server.Config, "avatars", str(tmp_path))
    loads = []

    def fake_load(root, name, pipe):
        loads.append(name)
        return type("FakeAvatar", (), {"name": name, "stamp": avatar_stamp(root, name)})()

    monkeypatch.setattr(server, "load_avatar", fake_load)
    return server.Runtime(), tmp_path, loads


def _write_meta(root, name, mtime=None):
    os.makedirs(root / name, exist_ok=True)
    path = root / name / "meta.json"
    path.write_text(json.dumps({"name": name}))
    if mtime is not None:
        os.utime(path, ns=(mtime, mtime))


def test_loaded_avatar_is_reused_until_meta_changes(runtime):
    rt, root, loads = runtime
    _write_meta(root, "a", mtime=1_000_000_000)
    first = rt.get_avatar("a")
    assert rt.get_avatar("a") is first and loads == ["a"]

    _write_meta(root, "a", mtime=2_000_000_000)  # onboarded again
    assert rt.get_avatar("a") is not first and loads == ["a", "a"]


def test_loading_another_avatar_replaces_the_one_in_memory(runtime):
    rt, root, loads = runtime
    _write_meta(root, "a")
    _write_meta(root, "b")
    rt.get_avatar("a")
    rt.get_avatar("b")
    assert list(rt.loaded) == ["b"]


def test_unknown_avatar_is_a_key_error(runtime):
    rt, _, loads = runtime
    with pytest.raises(KeyError):
        rt.get_avatar("nobody")
    assert loads == []
