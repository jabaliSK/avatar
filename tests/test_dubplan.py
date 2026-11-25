import random

import numpy as np
import pytest

from photostage import dubplan
from photostage.dubplan import Clip

IDLE = Clip("idle", "idle", "idle", 32, 16.0, True)
THINK = Clip("think", "think", "think", 32, 16.0, True)
EXPLAIN = Clip("explain", "talk", "explain", 81, 16.0, False)  # 5 s
GREET = Clip("greet", "talk", "greet", 33, 16.0, False)  # 2 s
PACK = {clip.name: clip for clip in (IDLE, THINK, EXPLAIN, GREET)}
STILL = Clip("photo", "idle", "idle", 1, 25, True)


def rows_of(plan):
    return [dict(zip(("clip", "frame", "next", "next_frame", "mix", "dub"), row)) for row in plan.frames]


def check(plan, speech, clips=PACK):
    """What the worker relies on: valid clips and frames, weights in range, lip-sync for all of the speech, and an end
    where the lip-sync has faded out."""
    rows = rows_of(plan)
    assert plan.seconds >= speech + dubplan.DUB_FADE
    for i, row in enumerate(rows):
        clip = clips.get(plan.clips[row["clip"]], STILL)
        assert 0 <= row["frame"] < clip.frames
        if row["next"] >= 0:
            assert 0 <= row["next_frame"] < clips.get(plan.clips[row["next"]], STILL).frames
            assert 0 < row["mix"] < 1
        else:
            assert row["mix"] == 0
        if i / dubplan.FPS <= speech:
            assert row["dub"] == 1
        else:
            assert 0 <= row["dub"] <= rows[i - 1]["dub"]
    assert rows[-1]["dub"] == 0 and rows[-1]["next"] == -1
    return rows


def test_sentences_and_moods_follow_the_browser():
    said = dubplan.sentences("Hello there! Could you tell me more? Thank you. Because it matters, we test it.")
    assert [s["text"] for s in said] == ["Hello there!", "Could you tell me more?", "Thank you.",
                                         "Because it matters, we test it."]
    assert said[0]["from"] == 0 and said[-1]["to"] == 1
    assert [dubplan.mood(s, i) for i, s in enumerate(said)] == ["greet", "ask", "calm", "explain"]
    assert dubplan.mood({"text": "Absolutely wonderful work on all of this today, team", "words": 9}, 1) == "emphasis"
    assert dubplan.sentences("...") == []


def test_a_long_reply_gestures_with_rests_and_crossfades_between_them():
    speech = 11.5
    plan = dubplan.plan(PACK, "Hello! Let me explain how this works, because it matters to you and to everyone here.",
                        speech, rng=random.Random(1))
    rows = check(plan, speech)
    assert plan.gestures[0][0] == "greet" and plan.gestures[0][1] == 0
    assert len(plan.gestures) >= 2
    assert {name for name, _, _ in plan.gestures} <= {"explain", "greet"}
    # Between two gestures the body rests in a quiet clip, never a still frame.
    names = [plan.clips[r["clip"]] for r in rows]
    assert "idle" in names or "think" in names
    # Each switch of clip crossfades over CROSSFADE seconds, from the frame the previous clip stopped at.
    fading = [r for r in rows if r["next"] >= 0]
    assert fading and all(r["mix"] < 1 for r in fading)
    assert max(sum(1 for _ in group) for group in _runs(rows)) >= 3


def _runs(rows):
    run = []
    for row in rows:
        if row["next"] >= 0:
            run.append(row)
        elif run:
            yield run
            run = []


def test_a_gesture_that_ends_with_the_speech_leaves_time_for_the_lip_sync_to_fade():
    for speech in (1.9, 2.0, 2.1):
        plan = dubplan.plan({"idle": IDLE, "greet": GREET}, "Hello!", speech, rng=random.Random(0))
        rows = check(plan, speech, {"idle": IDLE, "greet": GREET})
        assert plan.gestures == [["greet", 0.0, 2.0]]
        assert plan.clips[rows[-1]["clip"]] == "idle"


def test_lip_sync_holds_until_a_gesture_that_outlasts_the_speech_has_ended():
    speech = 4.0
    plan = dubplan.plan({"idle": IDLE, "explain": EXPLAIN}, "Let me explain.", speech, rng=random.Random(0))
    rows = check(plan, speech, {"idle": IDLE, "explain": EXPLAIN})
    assert plan.gestures == [["explain", 0.0, EXPLAIN.seconds]]
    fading = [i / dubplan.FPS for i, row in enumerate(rows) if 0 < row["dub"] < 1]
    assert fading and min(fading) >= EXPLAIN.seconds - dubplan.DUB_FADE - 2 / dubplan.FPS
    assert plan.clips[rows[-1]["clip"]] == "explain"


def test_short_replies_and_no_motion_only_rest():
    assert dubplan.plan(PACK, "Hi.", 0.3, rng=random.Random(0)).gestures == []
    plan = dubplan.plan(PACK, "Hello! Let me explain this to you.", 6, pace=0, rng=random.Random(0))
    assert plan.gestures == [] and set(plan.clips) <= {"idle", "think"}
    check(plan, 6)


def test_without_a_quiet_clip_the_photo_stands_in():
    plan = dubplan.plan({}, "Hello there.", 1.0, still=STILL)
    assert plan.clips == ["photo"]
    rows = check(plan, 1.0, {})
    assert all(r["clip"] == 0 and r["frame"] == 0 for r in rows)
    with pytest.raises(ValueError):
        dubplan.plan({"greet": GREET}, "Hello there.", 1.0)
    with pytest.raises(ValueError):
        dubplan.plan(PACK, "Hello there.", 0)


def test_the_plan_is_read_from_the_clip_pack():
    manifest = {"clips": {
        "idle-s8": {"role": "idle", "mood": "idle", "duration": 2.0, "fps": 16, "loop": True},
        "explain-long-s6": {"role": "talk", "mood": "explain", "duration": 5.0625, "fps": 16},
        "bad": {"role": "dance", "duration": 1, "fps": 16},
        "empty": {"role": "talk", "duration": 0.05, "fps": 16},
    }}
    clips = dubplan.clips_of(manifest)
    assert sorted(clips) == ["explain-long-s6", "idle-s8"]
    assert clips["explain-long-s6"].frames == 81 and clips["explain-long-s6"].seconds == 5.0
    assert clips["idle-s8"].frame(2.0) == 0 and clips["explain-long-s6"].frame(9) == 80
    message = dubplan.plan(clips, "Let me explain.", 4.0, rng=random.Random(0)).message()
    assert message["fps"] == 25 and message["clips"] == sorted(set(message["clips"]))


def _poses(values, frames):
    found = np.full((frames, 1), 50.0, np.float32)
    for index, value in values.items():
        found[index] = value
    return found


def test_rests_begin_where_both_their_ends_meet_what_is_around_them():
    # Every idle frame looks different; the greet ends like idle frame 7, explain begins like 15 and ends like 3, and the
    # photo looks like idle frame 19.
    poses = {"idle": np.arange(32, dtype=np.float32)[:, None], "greet": _poses({32: 7}, 33),
             "explain": _poses({0: 15, 80: 3}, 81)}
    pieces = [[GREET, 0.0, 2.0, 0.0], [IDLE, 2.0, 0.5, 0.0], [EXPLAIN, 2.5, 5.0, 0.0], [IDLE, 7.5, 1.0, 0.0]]
    dubplan.begin_rests(pieces, poses, np.array([19.0], np.float32))
    assert [piece[3] for piece in pieces] == [0.0, 7 / 16, 0.0, 3 / 16]
    # A rest the poses do not cover begins at its start; the last one ends like the photo (idle frame 4: 28 + 8 frames).
    pieces = [[THINK, 0.0, 1.0, 0.0], [IDLE, 1.0, 0.5, 0.0]]
    dubplan.begin_rests(pieces, {"idle": poses["idle"]}, np.array([4.0], np.float32))
    assert pieces[0][3] == 0.0 and pieces[1][3] == 28 / 16


def test_poses_move_where_rests_begin_but_not_the_timeline():
    text = "Hello! Let me explain how this works, because it matters to you and to everyone here."
    poses = {name: np.random.default_rng(i).normal(size=(clip.frames, 4)).astype(np.float32)
             for i, (name, clip) in enumerate(PACK.items())}
    plain = dubplan.plan(PACK, text, 11.5, rng=random.Random(1))
    posed = dubplan.plan(PACK, text, 11.5, rng=random.Random(1), poses=poses, anchor=poses["idle"][0])
    assert posed.gestures == plain.gestures and len(posed.frames) == len(plain.frames)
    rows = check(posed, 11.5)
    resting = [(r, p) for r, p in zip(rows, rows_of(plain)) if posed.clips[r["clip"]] in ("idle", "think")]
    assert any(r["frame"] != p["frame"] for r, p in resting)


def test_continuous_motion_keeps_phase_speed_and_one_face_through_the_tail():
    plan = dubplan.continuous(IDLE, 5.17, 29)
    assert plan.clips == ["idle"] and not plan.gestures
    assert [row[1] for row in plan.frames] == [(29 + i * 16 // 25) % 32 for i in range(len(plan.frames))]
    assert all(row[2:] == [-1, 0, 0.0, 1.0] for row in plan.frames)
    assert plan.quiet_from / 25 >= 5.17 + 0.4
    assert plan.seconds - plan.quiet_from / 25 >= 0.2
    assert plan.message()["quiet_from"] == plan.quiet_from
    for frame in (-1, 32, True, 0.5):
        with pytest.raises(ValueError, match="start frame"):
            dubplan.continuous(IDLE, 1, frame)
    with pytest.raises(ValueError, match="idle loop"):
        dubplan.continuous(GREET, 1, 0)
