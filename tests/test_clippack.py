"""The clip pack builder: what it makes of the clips a video model left behind. The clips here are synthetic: a textured
picture carried through light, a "hand" and a turn of the neck by known amounts, so that what comes out can be checked
against the truth, and the manifest against the player's own code."""

import contextlib
import io
import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from photostage import clippack, clips
from test_clips import moved, texture

ROOT = Path(__file__).resolve().parents[1]
MODEL = "wan-i2v-a14b-lightning"
SIZE, FACE, OTHER_FACE = [192, 344], [72, 30, 120, 84], [72, 40, 120, 94]
AREA, FPS, COUNT = 30000, 16, 33
NECK = {"turn": 1.5, "shift": (2.0, -1.5)}  # how far the explain clip moves the neck: degrees, work pixels


def scene(shape, seed):
    """A picture of the kind a camera makes: smooth, mostly light and dark with a little colour, away from the ends of the range."""
    base = cv2.GaussianBlur(texture(shape, seed), (0, 0), 1.5).astype(np.float32)
    base = 128 + (base - 128) * 2.2  # smoothing took the contrast out of it
    return np.clip(128 + 0.5 * (0.85 * base.mean(2, keepdims=True) + 0.15 * base - 128), 0, 255).astype(np.uint8)


def film(first, count, hand=0.0, bumps=1, dim=0.0, turn=0.0, shift=(0.0, 0.0)):
    """`first` carried through `count` frames: a "hand" (another texture over a part of the lower picture) that comes and
    goes `bumps` times, the light dimmed by `dim` at the most, the neck turned by `turn` degrees and moved by `shift`
    pixels at the most (the upper part of the picture moves with it; the camera and the rest of the picture are still).
    Each starts from nothing and comes back to nothing."""
    height, width = first.shape[:2]
    mask = np.zeros((height, width), np.float32)
    mask[int(height * 0.55):int(height * 0.8), int(width * 0.15):int(width * 0.6)] = 1
    mask = cv2.GaussianBlur(mask, (0, 0), 2)[..., None]
    upper = np.clip((0.52 - np.arange(height) / height) / 0.1, 0, 1).astype(np.float32)[:, None, None]
    other = texture((height, width), seed=77).astype(np.float32)
    frames = []
    for i in range(count):
        along = i / (count - 1)
        s = np.sin(np.pi * along) ** 2
        turned, _ = moved(first, turn * s, (shift[0] * s, shift[1] * s))
        picture = first * (1 - upper) + turned * upper
        alpha = hand * np.sin(np.pi * bumps * along) ** 2 * mask
        frames.append(np.clip((picture * (1 - alpha) + other * alpha) * (1 - dim * s) + 0.5, 0, 255).astype(np.uint8))
    return np.stack(frames)


def make_run(root, photo, gesture, frames_of, face=FACE, seed=1):
    """A run as the clip maker leaves it: a folder with clip.mp4, first.png and info.json."""
    plan = clips.plan_region(SIZE, face, 16, AREA)
    first = clips.crop(photo, plan)
    folder = root / f"{MODEL}-{gesture}-s{seed}"
    folder.mkdir(parents=True)
    clips.write_video(folder / "clip.mp4", frames_of(first), FPS)
    cv2.imwrite(str(folder / "first.png"), cv2.cvtColor(first, cv2.COLOR_RGB2BGR))
    info = {"avatar": "avatar", "model": MODEL, "gesture": gesture, "seed": seed, "frames": COUNT, "fps": FPS, **plan,
            "band": list(clips.band_rows(plan, face)), "face_h": (face[3] - face[1]) * clips.work_scale(plan)}
    (folder / "info.json").write_text(json.dumps(info))
    return plan, first


def write_photo(folder, photo):
    cv2.imwrite(str(folder / "photo.png"), cv2.cvtColor(photo, cv2.COLOR_RGB2BGR))


@pytest.fixture(scope="module")
def made(tmp_path_factory):
    """An avatar (photo and rig) and the runs of four clips of it: a thinking clip whose light drifts, an explaining one
    that moves the neck, one that does not move, and one made for another part of the photo."""
    base = tmp_path_factory.mktemp("made")
    avatar, runs = base / "avatar", base / "runs"
    avatar.mkdir()
    photo = scene((SIZE[1], SIZE[0]), seed=11)
    write_photo(avatar, photo)
    (avatar / "rig.json").write_text(json.dumps({"size": SIZE, "face": FACE}))
    plan, first = make_run(runs, photo, "thinking", lambda f: film(f, COUNT, hand=0.5, bumps=2, dim=0.10))
    make_run(runs, photo, "explain", lambda f: film(f, COUNT, hand=1.0, **NECK))
    make_run(runs, photo, "calm", lambda f: np.stack([f] * COUNT))
    make_run(runs, photo, "ask", lambda f: film(f, COUNT, hand=1.0), face=OTHER_FACE)
    return SimpleNamespace(avatar=avatar, runs=runs, plan=plan, first=first)


def build(made, where, *argv, photo=None):
    """clippack.main on a copy of the avatar in `where`: (return code, what it printed, the avatar's folder)."""
    folder = where / "avatar"
    shutil.copytree(made.avatar, folder, dirs_exist_ok=True)
    if photo is not None:
        write_photo(folder, photo)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = clippack.main([str(folder), "--clips", str(made.runs), *argv])
    return code, out.getvalue(), folder


@pytest.fixture(scope="module")
def pack(made, tmp_path_factory):
    code, out, folder = build(made, tmp_path_factory.mktemp("pack"))
    return SimpleNamespace(code=code, out=out, folder=folder, manifest=json.loads((folder / "clips.json").read_text()))


def test_levels_are_fitted_through_a_patch_that_really_differs():
    source = scene((240, 320), seed=2).astype(np.float32)
    gain, bias = np.array([0.9, 0.85, 0.95]), np.array([6.0, -4.0, 10.0])
    target = source * gain + bias
    target[60:110, 100:180] = scene((50, 80), seed=9)  # a hand
    got_gain, got_bias = clippack.fit_levels(source, target)
    assert got_gain == pytest.approx(gain, abs=0.03) and got_bias == pytest.approx(bias, abs=2.5)


def test_full_avatar_geometry_tracks_the_face_not_the_static_background():
    plan = clips.plan_avatar(SIZE, FACE, 16, AREA)
    box = clips.face_box(plan, FACE)
    info = {**plan, "band": list(clips.band_rows(plan, FACE)), "framing": "avatar", "track_box": box}
    spec = clippack.geometry({"size": SIZE, "face": FACE}, info)
    assert spec["mode"] == "avatar" and spec["box"] == [0, 0, *SIZE]
    assert spec["pivot"] == pytest.approx([(FACE[0] + FACE[2]) / 2, (FACE[1] + FACE[3]) / 2], abs=2)
    with pytest.raises(SystemExit, match="track_box"):
        clippack.geometry({"size": SIZE, "face": FACE}, {**info, "track_box": [0, 0, 0, 10]})

def test_full_head_tracks_the_collar_instead_of_the_independently_moving_video_face():
    first = scene((344, 192), seed=23)
    info = {"model": MODEL, "gesture": "explain", "seed": 1, "fps": FPS,
            "region": [0, 0, 192, 344], "work": [192, 344], "band": [84, 138],
            "face_h": 54, "framing": "avatar", "track_box": [72, 30, 120, 84]}
    frames = []
    for amount in np.sin(np.linspace(0, np.pi, COUNT)) ** 2:
        body, _ = moved(first, 0, (4 * amount, 0))
        face, _ = moved(first, 0, (-8 * amount, 0))
        body[:84] = face[:84]
        frames.append(body)
    prepared = clippack.prepare("explain", info, np.stack(frames), first, clippack.Gates(), face_mode="head")
    spec = clippack.geometry({"size": SIZE, "face": FACE}, info, "head")
    assert spec["pivot"] == [96, 100]
    assert spec["mode"] == "avatar" and spec["box"] == info["region"]
    assert "faceTrack" not in prepared.entry
    track = np.array(prepared.entry["track"])
    assert track[:, 2].max() == pytest.approx(4, abs=0.6)
    assert np.abs(track[:, 3]).max() < 0.4
    with pytest.raises(SystemExit, match="visible collar"):
        clippack.tracking_box({**info, "track_box": [72, 290, 120, 341]}, "head")


def test_a_full_avatar_clip_narrower_than_the_photo_is_widened_with_the_photos_own_edge():
    photo = scene((SIZE[1], SIZE[0]), seed=31)
    info = {"model": MODEL, "gesture": "explain", "seed": 1, "fps": FPS, "region": [9, 0, 191, 344], "work": [182, 344],
            "band": [84, 138], "face_h": 54, "framing": "avatar", "track_box": [63, 30, 111, 84]}
    first = clips.crop(photo, info)
    frames = film(first, COUNT, hand=1.0)
    seated, pad = clippack.full_seat(info, SIZE)
    assert pad == [9, 0, 1, 0] and seated["region"] == [0, 0, *SIZE] and seated["work"] == SIZE
    assert seated["track_box"] == FACE and seated["band"] == info["band"]
    assert clippack.geometry({"size": SIZE, "face": FACE}, seated, "head")["box"] == [0, 0, *SIZE]
    assert clippack.full_seat(seated, SIZE)[1] is None
    prepared = clippack.prepare("explain", info, frames, first, clippack.Gates(), face_mode="head", photo=photo)
    mid, feather = COUNT // 2, clippack.SEAT_FEATHER
    assert prepared.frames.shape[1:3] == (SIZE[1], SIZE[0])
    assert np.array_equal(prepared.frames[mid, :, :9], photo[:, :9]) and np.array_equal(prepared.frames[mid, :, -1:], photo[:, -1:])
    shown = prepared.frames[mid, :, 9 + feather:191 - feather].astype(np.float32)
    assert np.abs(shown - frames[mid, :, feather:182 - feather]).mean() < 2
    hand = (slice(200, 265), slice(45, 105))  # where film() puts the hand, moved by the 9 pixels added on the left
    assert np.abs(prepared.frames[mid][hand].astype(np.float32) - photo[hand]).mean() > 20


def test_face_registration_uses_the_renderers_reference_and_recovers_nonuniform_pose():
    reference = np.random.default_rng(12).uniform(0.2, 0.8, (478, 2)).astype(np.float32)
    expected = np.array([[0.96, 0.06, 5 / 512], [-0.025, 1.02, -5 / 512]])
    moved = np.column_stack([reference, np.ones(len(reference))]) @ expected.T
    moved[[13, 14, 17, 152], 1] += 0.08  # lips and jaw move independently of the face's pose
    pose = clippack.face_pose(reference, moved).reshape(2, 3)
    points = np.array([[0.3, 0.3, 1], [0.7, 0.3, 1], [0.5, 0.65, 1]])
    assert np.linalg.norm(points @ (pose - expected).T, axis=1).max() * 512 < 1.5
    assert clippack.face_pose(reference, None) is None


def test_face_registration_rejects_a_bad_stable_landmark():
    reference = np.random.default_rng(22).uniform(0.2, 0.8, (478, 2)).astype(np.float32)
    expected = np.array([[0.96, 0.06, 5 / 512], [-0.025, 1.02, -5 / 512]])
    moved = np.column_stack([reference, np.ones(len(reference))]) @ expected.T
    moved[33] += [0.10, -0.08]

    pose = clippack.face_pose(reference, moved).reshape(2, 3)

    assert np.linalg.norm(np.array([[0.3, 0.3, 1], [0.7, 0.3, 1]]) @ (pose - expected).T, axis=1).max() * 512 < 1.0


def test_light_that_drifts_is_put_right_and_a_hand_is_kept():
    reference = scene((120, 160), seed=4)
    hand = texture((40, 50), seed=9).astype(np.float32)
    frames, truth = [], []
    for i in range(COUNT):
        s = np.sin(np.pi * i / (COUNT - 1)) ** 2
        real = reference.astype(np.float32)
        real[70:110, 30:80] = real[70:110, 30:80] * (1 - s) + hand * s  # a hand comes into the picture
        truth.append(real)
        frames.append(np.clip(real * (1 - 0.12 * s) + 8 * s, 0, 255).astype(np.uint8))
    out, report = clippack.correct_tone(np.stack(frames), reference)
    assert np.abs(out.astype(np.float32) - np.stack(truth)).mean() < 1.0  # a clip dimmed by an eighth is as bright as it began
    assert not report["refused"] and report["drift"]["gain"] == pytest.approx(0.13, abs=0.03)
    assert np.abs(out[COUNT // 2][70:110, 30:80].astype(np.float32) - reference[70:110, 30:80]).mean() > 20


def test_the_first_frame_is_brought_onto_the_picture_the_model_was_given():
    reference = scene((120, 160), seed=4)
    frames = np.stack([np.clip(reference * 0.93 + 5, 0, 255).astype(np.uint8)] * 5)
    out, report = clippack.correct_tone(frames, reference)
    assert np.abs(out.astype(np.float32) - reference).mean() < 1.0
    assert report["first"]["gain"] == pytest.approx(1 / 0.93 - 1, abs=0.02)


def test_a_frame_that_no_fit_can_put_right_is_left_as_it_is():
    reference = scene((120, 160), seed=4)
    frames = np.stack([reference] * 5)
    frames[2] = 255 - reference
    out, report = clippack.correct_tone(frames, reference)
    assert report["refused"] and np.abs(out[2].astype(int) - frames[2]).max() <= 1


def test_a_loop_is_closed_over_its_last_second_and_left_alone_before_it():
    frames = np.stack([np.full((8, 8, 3), 100 + 5 * i, np.uint8) for i in range(17)])
    out = clippack.close_loop(frames, 8)  # a second is 8 frames
    assert np.array_equal(out[-1], out[0]) and np.array_equal(out[:9], frames[:9])
    assert np.abs(np.diff(out[:, 0, 0, 0].astype(int))).max() <= 10  # the whole difference (80) is spread over the second, not a jump
    assert out[12, 0, 0, 0] == 120


def dips(count, where, depth, peak=9.0, width=2.5):
    """A distance-from-rest curve that is `peak` but for dips to `depth` at the frames `where`."""
    i = np.arange(count)
    out = np.full(count, peak)
    for at, low in zip(where, depth):
        out = np.minimum(out, low + (peak - low) * (1 - np.exp(-(((i - at) / width) ** 2))))
    return out


def test_exits_are_calm_moments_not_too_near_the_start_or_each_other():
    # dips at 0.25 s (too soon), 1.25 s and 1.5 s (too close: the calmer is kept), 2.5 s and 2.25 s (too close to 1.5 s)
    rest = dips(49, [4, 20, 24, 36, 40], [1.0, 1.0, 0.2, 1.5, 1.0])
    assert clippack.find_exits(rest, 16) == [1.5, 2.5]
    assert clippack.find_exits(np.full(49, 9.0), 16) == []  # nothing calm in it
    assert clippack.find_exits(dips(49, [24], [12.0], peak=50.0), 16) == []  # a dip that is not calm beside a peak of 50


def test_the_neck_is_followed_in_radians_and_photo_pixels():
    picture = scene((208, 144), seed=6)
    rows, scale = (20, 61), 0.75
    centre = np.array([72.0, (rows[0] + rows[1]) / 2])
    frames, truth = [], []
    for i in range(COUNT):
        s = np.sin(np.pi * i / (COUNT - 1)) ** 2
        frame, matrix = moved(picture, 2.0 * s, (3.0 * s, -2.0 * s))
        frames.append(frame)
        truth.append([np.arctan2(matrix[1, 0], matrix[0, 0]), *(matrix[:, :2] @ centre + matrix[:, 2] - centre) / scale])
    track = clippack.neck_track(np.stack(frames), rows, scale, loop=False)
    truth = np.array(truth)
    assert track.shape == (COUNT, 4) and track[:, 0] == pytest.approx(1.0, abs=0.01)
    for column, tolerance in ((1, 0.004), (2, 0.4), (3, 0.4)):
        assert track[:, column] == pytest.approx(clippack.smoothed(truth[:, column - 1], clippack.TRACK_SIGMA), abs=tolerance)
    assert np.abs(track[:, 3]).max() > 2  # the neck did move, so the numbers above are not zeros agreeing


def test_a_picture_with_nothing_to_follow_gives_a_track_that_stays_at_rest():
    track = clippack.neck_track(np.full((6, 208, 144, 3), 128, np.uint8), (20, 61), 0.75, loop=True)
    assert np.array_equal(track, np.tile([1.0, 0.0, 0.0, 0.0], (6, 1)))


GOOD = {"fidelity": 40.0, "band_lost": 0.0, "tone_drift": 2.0, "softening": 0.9, "flicker": 0.8, "motion_peak": 6.0}


def test_a_clip_is_judged_on_what_it_is_for():
    gates = clippack.Gates()

    def problems(role="talk", gap=1.0, exits=(), duration=4.0, **numbers):
        return clippack.judge({**GOOD, **numbers}, role, gap, list(exits), duration, gates)

    assert problems() == [] and problems("think", exits=[2.0], motion_peak=5.0) == []
    assert any("hardly moves" in p for p in problems(motion_peak=1.0))
    assert any("too much" in p for p in problems("think", exits=[2.0], motion_peak=12.0))
    assert any("no calm moment" in p for p in problems("think", motion_peak=5.0))  # four seconds with nowhere to let go
    assert any("come back" in p for p in problems(gap=9.0))
    for field, value, word in (("fidelity", 20.0, "photo"), ("band_lost", 0.5, "neck"), ("tone_drift", 25.0, "colours"),
                               ("softening", 0.2, "blurs"), ("flicker", 9.0, "shimmers")):
        found = problems(**{field: value})
        assert len(found) == 1 and word in found[0]

@pytest.mark.parametrize("gesture,role", [("idle", "idle"), ("listening", "listen")])
def test_non_speaking_states_get_closed_loops_and_quiet_quality_gates(made, gesture, role):
    info = json.loads((made.runs / f"{MODEL}-thinking-s1" / "info.json").read_text())
    info["gesture"] = gesture
    frames = film(made.first, COUNT, hand=0.5, bumps=2, dim=0.1)
    clip = clippack.prepare("state", info, frames, made.first, clippack.Gates())
    assert clip.role == role and clip.entry["loop"]
    assert len(clip.frames) == COUNT - 1
    assert len(clip.entry["track"]) == COUNT - 1
    assert clip.problems == []
    assert any("too much" in p for p in clippack.judge(
        {**GOOD, "motion_peak": 12}, role, 1, [1], 2, clippack.Gates()))
    assert any("no calm moment" in p for p in clippack.judge(
        GOOD, role, 1, [], 4, clippack.Gates()))


def test_a_gate_is_moved_with_name_equals_number():
    assert clippack.parse(["a", "--gate", "softening=0.4", "--gate", "gap=9"]).gates == clippack.Gates(softening=0.4, gap=9.0)
    for bad in ("nonsense=1", "softening=lots", "softening"):
        with pytest.raises(SystemExit):
            clippack.parse(["a", "--gate", bad])


def test_a_clip_is_named_for_its_gesture_and_seed():
    assert clippack.clip_name(Path(f"x/{MODEL}-explain-s3"), {"model": MODEL}) == "explain-s3"
    assert clippack.clip_name(Path("x/my-own"), {"model": MODEL}) == "my-own"


def test_runs_of_one_model_are_found_and_models_are_not_mixed(tmp_path):
    stub = {"gesture": "explain", "seed": 3, "frames": 81, "fps": 16, "region": [0, 0, 1, 1], "work": [16, 16], "band": [0, 1], "face_h": 1.0}
    for model in ("wan-ti2v-5b", "wan-i2v-a14b"):
        (tmp_path / f"{model}-explain-s3").mkdir()
        (tmp_path / f"{model}-explain-s3" / "info.json").write_text(json.dumps({**stub, "model": model}))
    (tmp_path / "broken").mkdir()
    (tmp_path / "broken" / "info.json").write_text("{")
    (tmp_path / "plan").mkdir()
    (tmp_path / "plan" / "info.json").write_text(json.dumps({"model": "wan-i2v-a14b"}))  # a dry run's: no clip in it
    with pytest.raises(SystemExit, match="several models"):
        clippack.find_runs(tmp_path, None)
    assert [run.name for run, _ in clippack.find_runs(tmp_path, "wan-i2v-a14b")] == ["wan-i2v-a14b-explain-s3"]
    shutil.rmtree(tmp_path / "wan-ti2v-5b-explain-s3")
    assert [run.name for run, _ in clippack.find_runs(tmp_path, None)] == ["wan-i2v-a14b-explain-s3"]


def test_a_clip_that_fails_is_left_out_with_the_reason_and_the_numbers_are_printed(pack):
    assert pack.code == 0 and set(pack.manifest["clips"]) == {"thinking-s1", "explain-s1"}
    assert "calm-s1" in pack.out and "left out: it hardly moves" in pack.out
    assert "ask-s1: left out, it was made for another part of the photo than most of the clips." in pack.out
    assert "Installed 2 clips" in pack.out


def test_installing_one_replacement_preserves_unchanged_clip_cache_versions(tmp_path):
    folder, stage = tmp_path / "avatar", tmp_path / "stage"
    folder.mkdir()
    stage.mkdir()
    same, changed = "clip-explain.mp4", "clip-idle.mp4"
    (folder / same).write_bytes(b"gesture")
    (folder / changed).write_bytes(b"old idle")
    before = (folder / same).stat()
    (stage / same).write_bytes(b"gesture")
    (stage / changed).write_bytes(b"new idle")
    clippack.install(folder, {"version": 1}, {name: stage / name for name in (same, changed)})
    after = (folder / same).stat()
    assert (after.st_ino, after.st_mtime_ns) == (before.st_ino, before.st_mtime_ns)
    assert (folder / changed).read_bytes() == b"new idle"
    assert not list(stage.iterdir())


def test_the_manifest_says_how_the_clips_sit_on_the_photo(made, pack):
    manifest, plan = pack.manifest, made.plan
    rows, scale = clips.band_rows(plan, FACE), clips.work_scale(plan)
    assert manifest["version"] == 1 and manifest["size"] == SIZE and manifest["box"] == plan["region"]
    assert manifest["pivot"] == pytest.approx([plan["region"][0] + plan["work"][0] / 2 / scale,
                                               plan["region"][1] + (rows[0] + rows[1]) / 2 / scale], abs=0.01)
    assert manifest["follow"] == [FACE[1], FACE[3]] and manifest["types"] == clippack.TYPES


def test_each_clip_is_told_what_it_is_for_and_when_it_can_end(pack):
    think, talk = pack.manifest["clips"]["thinking-s1"], pack.manifest["clips"]["explain-s1"]
    assert (think["role"], think["mood"], think["loop"]) == ("think", "think", True)
    assert (talk["role"], talk["mood"], talk["loop"]) == ("talk", "explain", False)
    # a loop is closed on its first frame and does not show it twice; a clip that is played once shows all of its own
    assert think["fps"] == talk["fps"] == FPS
    assert think["duration"] == pytest.approx((COUNT - 1) / FPS) and len(think["track"]) == COUNT - 1
    assert talk["duration"] == pytest.approx(COUNT / FPS) and len(talk["track"]) == COUNT
    assert len(think["exits"]) == 1 and think["exits"][0] == pytest.approx(1.0, abs=0.25)  # its hand goes in and out twice
    assert talk["exits"] == []
    assert think["files"] == {"mp4": "clip-thinking-s1.mp4", "webm": "clip-thinking-s1.webm"}
    assert set(think["stats"]) >= {"fidelity", "softening", "flicker", "motion_peak", "tone_drift", "model", "gesture", "seed"}


def test_the_neck_of_a_clip_is_tracked_for_the_head_to_follow(made, pack):
    first, plan = made.first, made.plan
    rows, scale = clips.band_rows(plan, FACE), clips.work_scale(plan)
    centre = np.array([first.shape[1] / 2, (rows[0] + rows[1]) / 2])
    truth = []
    for i in range(COUNT):
        s = np.sin(np.pi * i / (COUNT - 1)) ** 2
        _, matrix = moved(first, NECK["turn"] * s, (NECK["shift"][0] * s, NECK["shift"][1] * s))
        truth.append([np.arctan2(matrix[1, 0], matrix[0, 0]), *(matrix[:, :2] @ centre + matrix[:, 2] - centre) / scale])
    truth, track = np.array(truth), np.array(pack.manifest["clips"]["explain-s1"]["track"])
    assert track.shape == (COUNT, 4) and track[:, 0] == pytest.approx(1.0, abs=0.01)
    for column, tolerance in ((1, 0.006), (2, 0.5), (3, 0.5)):
        assert track[:, column] == pytest.approx(clippack.smoothed(truth[:, column - 1], clippack.TRACK_SIGMA), abs=tolerance)
    still = np.array(pack.manifest["clips"]["thinking-s1"]["track"])
    assert np.abs(still[:, 1:]).max() < 0.5 and np.abs(still[:, 0] - 1).max() < 0.01  # the thinking clip leaves the neck be


def test_the_pack_is_two_formats_of_each_clip_and_nothing_else_is_left_behind(pack):
    assert sorted(p.name for p in pack.folder.iterdir()) == [
        "clip-explain-s1.mp4", "clip-explain-s1.webm", "clip-thinking-s1.mp4", "clip-thinking-s1.webm",
        "clips.json", "photo.png", "rig.json"]


@pytest.mark.parametrize("kind", ["mp4", "webm"])
def test_the_installed_clips_keep_the_light_of_the_photo(made, pack, kind):
    source, _ = clips.read_video(made.runs / f"{MODEL}-thinking-s1" / "clip.mp4")
    wandered = np.abs(source.reshape(len(source), -1, 3).mean(1) - source[0].reshape(-1, 3).mean(0)).max()
    assert wandered > 8  # the clip as the model made it was dimmed by a tenth in the middle
    frames, fps = clips.read_video(pack.folder / f"clip-thinking-s1.{kind}")
    assert fps == FPS and frames.shape == (COUNT - 1, *made.first.shape)
    assert np.abs(frames[0].astype(np.float32) - made.first).mean() < 4
    assert np.abs(frames.reshape(len(frames), -1, 3).mean(1) - frames[0].reshape(-1, 3).mean(0)).max() < 3


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_the_player_accepts_the_manifest(pack):
    script = (
        'const fs = require("fs"), vm = require("vm"); const context = vm.createContext({ console });'
        ' vm.runInContext(fs.readFileSync(process.argv[1], "utf8"), context);'
        ' const spec = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));'
        ' const pack = vm.runInContext("parseClipManifest", context)(spec, [Number(process.argv[3]), Number(process.argv[4])]);'
        ' console.log(JSON.stringify(pack && Object.keys(pack.clips).map((name) => { const c = pack.clips[name];'
        ' return [name, c.duration, c.last, c.exits, Boolean(c.track), c.loop]; })));'
    )
    run = subprocess.run(["node", "-e", script, str(ROOT / "photostage/web/clips.js"), str(pack.folder / "clips.json"), *map(str, SIZE)],
                         capture_output=True, text=True, timeout=60)
    assert run.returncode == 0, run.stderr
    seen = {name: rest for name, *rest in json.loads(run.stdout)}
    assert set(seen) == {"thinking-s1", "explain-s1"}
    for name, (duration, last, exits, tracked, loop) in seen.items():
        assert tracked and exits[-1] == last and last == pytest.approx(duration if loop else duration - 1 / FPS)
    assert seen["thinking-s1"][3] is True and seen["explain-s1"][3] is True


def test_a_dry_run_prints_the_numbers_and_installs_nothing(made, tmp_path):
    code, out, folder = build(made, tmp_path, "--dry-run")
    assert code == 0 and "2 of 3 clips would be installed" in out and "fidelity" in out
    assert not (folder / "clips.json").exists() and not list(folder.glob("clip-*"))


def test_only_the_clips_asked_for_are_used_and_a_pack_without_gestures_is_warned_of(made, tmp_path):
    code, out, folder = build(made, tmp_path, "--only", "thinking-s1")
    assert code == 0 and list(json.loads((folder / "clips.json").read_text())["clips"]) == ["thinking-s1"]
    assert "no gesture clip" in out and "no thinking clip" not in out


def test_when_no_clip_is_fit_nothing_is_installed_and_the_pack_there_is_kept(made, tmp_path):
    _, _, folder = build(made, tmp_path, "--only", "thinking-s1")
    before = (folder / "clips.json").read_bytes()
    code, out, _ = build(made, tmp_path, "--only", "calm-s1")  # it does not move
    assert code == 1 and "nothing was installed" in out
    assert (folder / "clips.json").read_bytes() == before and (folder / "clip-thinking-s1.webm").exists()


def test_a_failing_clip_is_used_if_it_is_forced_or_its_gate_is_moved(made, tmp_path):
    code, out, folder = build(made, tmp_path / "forced", "--only", "calm-s1", "--force", "calm-s1")
    assert code == 0 and "FORCED: it hardly moves" in out and (folder / "clip-calm-s1.webm").exists()
    code, out, folder = build(made, tmp_path / "moved", "--only", "calm-s1", "--gate", "motion=0")
    assert code == 0 and "left out" not in out and (folder / "clip-calm-s1.mp4").exists()


def test_a_new_pack_replaces_the_old_one_and_a_pack_can_be_removed(made, tmp_path):
    _, _, folder = build(made, tmp_path, "--only", "calm-s1", "--force", "calm-s1")
    assert (folder / "clip-calm-s1.mp4").exists()
    build(made, tmp_path, "--only", "thinking-s1")
    assert sorted(p.name for p in folder.glob("clip-*")) == ["clip-thinking-s1.mp4", "clip-thinking-s1.webm"]
    assert list(json.loads((folder / "clips.json").read_text())["clips"]) == ["thinking-s1"]
    code, out, _ = build(made, tmp_path, "--remove")
    assert code == 0 and "Removed 3 files" in out and not (folder / "clips.json").exists() and not list(folder.glob("clip-*"))
    assert "has no clip pack" in build(made, tmp_path, "--remove")[1]


def test_clips_of_another_photo_are_refused(made, tmp_path):
    code, out, folder = build(made, tmp_path, photo=scene((SIZE[1], SIZE[0]), seed=12))
    assert code == 1 and out.count("it was not made from this avatar's photo") == 3
    assert not (folder / "clips.json").exists()
