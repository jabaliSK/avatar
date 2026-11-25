import json

import numpy as np
import pytest

from photostage import autopack, builder, library


def test_kind_by_face_height():
    assert autopack.kind({"face": [0, 100, 50, 400], "size": [1000, 1000]}) == "closeup"
    assert autopack.kind({"face": [0, 100, 50, 300], "size": [1000, 1000]}) == "wide"
    assert builder.kind({"face": [0, 100, 50, 400], "size": [1000, 1000]}) == "closeup"


def test_resample_picks_nearest_frames():
    frames = np.arange(26)
    assert list(autopack.resample(frames, 25, 16)) == [round(i * 25 / 16) for i in range(17)]


def test_close_loop_drops_the_repeat_and_lands_on_the_first_frame():
    frames = np.zeros((40, 2, 2, 3), np.uint8)
    frames[-1] = 12
    closed, gap = autopack.close_loop(frames, count=8)
    assert len(closed) == 39 and gap == 12
    # The correction grows towards the end, so the last kept frame leads smoothly back into the first.
    assert closed[30].max() == 0 and closed[-1].max() < closed[0].max() + 12


def test_livetalk_loop_leaves_out_the_photo_and_closes_on_its_own_start():
    take = np.full((30, 2, 2, 3), 50, np.uint8)
    take[0] = 200  # the photo, whose texture differs from the generated frames
    take[-1] = 60  # the take comes back near, not onto, its start
    frames, info = autopack.livetalk_loop(take)
    assert len(frames) == 28 and frames.max() < 200
    assert info["end_gap"] == 10 and info["closed"]


def test_rest_layout_puts_blinks_in_their_slots_only():
    ids = autopack.rest_layout(64, [8, 24])
    assert len(ids) == 128 and ids[:64] == list(range(64))
    assert ids[64 + 8:64 + 12] == [64, 65, 66, 67] and ids[64 + 24:64 + 28] == [68, 69, 70, 71]
    assert ids[64 + 12] == 12


def test_silence_rows_cover_the_loop_with_padding():
    rows = builder.silence_rows(4, pad=2)
    assert [row[1] for row in rows] == [2, 3, 0, 1, 2, 3, 0, 1]


def test_filler_box_bounds_only_the_changed_mouth():
    rest = [np.full((100, 100, 3), 50, np.uint8) for _ in range(2)]
    changed = rest[1].copy()
    changed[60:70, 40:52] = 200
    changed[5:10, 5:10] = 200  # outside the mouth region: ignored
    box = builder.filler_box([[(0, rest[0]), (1, changed)]], rest, [20, 40, 80, 95])
    x, y, w, h = box
    assert x <= 40 and y <= 60 and x + w >= 52 and y + h >= 70 and x % 2 == 0 and y % 2 == 0
    assert x >= 20 and y >= 40
    with pytest.raises(builder.BuildError):
        builder.filler_box([[(0, rest[0])]], rest, [20, 40, 80, 95])


def test_build_state(tmp_path):
    assert library.build_state(tmp_path) == {"state": "ready"}
    (tmp_path / "build.json").write_text(json.dumps({"state": "building", "job": "wide", "step": 1, "steps": 5,
                                                     "percent": 3}))
    assert library.build_state(tmp_path)["state"] == "building"
    (tmp_path / "build.json").write_text(json.dumps({"state": "building", "job": "fillers"}))
    assert library.build_state(tmp_path) == {"state": "ready"}


def _jpeg(value, mouth=None):
    import cv2
    picture = np.full((100, 100, 3), value, np.uint8)
    if mouth is not None:
        picture[60:70, 40:52] = mouth
    return cv2.imencode(".jpg", picture)[1].tobytes()


def test_nearest_loop_frames_finds_the_matching_pose():
    quiet = [_jpeg(v) for v in (20, 90, 160, 230)]
    rest = [_jpeg(v) for v in (150, 95, 225)]
    assert builder.nearest_loop_frames(rest, quiet, phases=(0, 1, 2)) == {0: 2, 1: 1, 2: 3}


def test_pack_fillers_writes_takes_in_the_speaking_loop(tmp_path, monkeypatch):
    import base64
    monkeypatch.setattr(builder, "FILLERS", ["Let me check that."])
    quiet = [_jpeg(50) for _ in range(4)]
    take = lambda start: [base64.b64encode(_jpeg(50, 200)).decode() for _ in range(3)]
    (tmp_path / "filler-loop-0.json").write_text(json.dumps({
        "text": "Let me check that.", "rate": 24000, "pcm": base64.b64encode(b"\0\0" * 10).decode(),
        "frames": {"0": take(2), "8": take(1)}, "loop": {"0": [2, 3, 0], "8": [1, 2, 3]}}))
    box = builder.pack_fillers(tmp_path, quiet, [30, 30, 70, 75])
    info = json.loads((tmp_path / "fillers.json").read_text())
    assert info["version"] == builder.FILLER_VERSION and info["loopFps"] == info["restFps"] == builder.FPS
    assert info["box"] == box and info["fade"] == builder.FADE
    (filler,) = info["fillers"]
    assert [(t["phase"], t["loop"], [r[0] for r in t["rows"]]) for t in filler["takes"]] == [(0, 2, [2, 3, 0]),
                                                                                              (8, 1, [1, 2, 3])]
    data = (tmp_path / "fillers.bin").read_bytes()
    assert filler["audio"] == [0, 20] and all(r[1] + r[2] <= len(data) for t in filler["takes"] for r in t["rows"])
