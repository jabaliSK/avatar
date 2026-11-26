"""Per-avatar settings, the frames played while the avatar is quiet, and the blink schedule of those frames."""
import numpy as np
import pytest

from app.avatar_store import (BLINK_FRAMES, DEFAULT_SETTINGS, SETTING_RANGES, BaseLoop, blink_gaps, preload_names, read_settings,
                              write_settings)


def test_settings_default_when_nothing_is_stored(tmp_path):
    assert read_settings(tmp_path) == DEFAULT_SETTINGS


def test_settings_are_clamped_and_unknown_keys_are_dropped(tmp_path):
    saved = write_settings(tmp_path, {"expression": 9, "head_motion": -3, "voice": "x"})
    assert saved == {"expression": SETTING_RANGES["expression"][1], "head_motion": SETTING_RANGES["head_motion"][0]}
    assert read_settings(tmp_path) == saved


def test_a_partial_update_keeps_the_other_setting(tmp_path):
    write_settings(tmp_path, {"expression": 0.9})
    assert write_settings(tmp_path, {"head_motion": 0.2}) == {"expression": 0.9, "head_motion": 0.2}


def test_a_damaged_settings_file_falls_back_to_the_defaults(tmp_path):
    (tmp_path / "settings.json").write_text('{"expression": "loud", "head_motion": ')
    assert read_settings(tmp_path) == DEFAULT_SETTINGS
    (tmp_path / "settings.json").write_text('{"expression": "loud", "head_motion": 0.3}')
    assert read_settings(tmp_path) == {**DEFAULT_SETTINGS, "head_motion": 0.3}


def test_preload_skips_a_name_that_is_gone(tmp_path, caplog):
    (tmp_path / "kept").mkdir()
    (tmp_path / "kept" / "meta.json").write_text("{}")
    (tmp_path / "unfinished").mkdir()  # no meta.json yet: not an avatar
    with caplog.at_level("WARNING", logger="kka.avatars"):
        assert preload_names(["gone", "kept", "unfinished"], tmp_path) == ["kept"]
    assert "'gone'" in caplog.text and "'unfinished'" in caplog.text
    assert preload_names(["all"], tmp_path) == ["kept"]
    assert preload_names(["kept"], tmp_path / "missing-folder") == []


def test_base_loop_wraps_when_cyclic_and_plays_ping_pong_otherwise():
    frames = np.arange(5, dtype=np.uint8).reshape(5, 1, 1, 1)
    cyclic, ping_pong = BaseLoop(frames, cyclic=True), BaseLoop(frames, cyclic=False)
    assert [int(cyclic.at(i)[0, 0, 0]) for i in range(7)] == [0, 1, 2, 3, 4, 0, 1]
    assert [int(ping_pong.at(i)[0, 0, 0]) for i in range(7)] == [0, 1, 2, 3, 4, 4, 3]


@pytest.mark.parametrize("cycle", [60, 162, 300, 1054])
def test_blinks_fall_whole_inside_the_cycle(cycle):
    gaps = blink_gaps(cycle)
    assert all(g > 0 for g in gaps)
    start = gaps[0]
    for open_frames in gaps[1:-1]:
        start += BLINK_FRAMES + open_frames  # the next blink starts after the previous one and the open frames in between
    assert start + BLINK_FRAMES <= cycle  # the last blink ends before the loop wraps
    assert gaps[-1] >= cycle  # and no blink follows it
