"""Body clips: the region an image-to-video model is shown, and the numbers that judge what it made. Clips here are
synthetic, made by moving a textured picture by known amounts, so that what `measure` reports can be checked against the truth."""

import json

import cv2
import numpy as np
import pytest

from photostage import clipbridge, clipgen, clipgen_liveavatar, clips

FACE = [561.4, 421.6, 1004.9, 865.2]  # the face box of a real portrait, 1536 x 2752
SIZE = [1536, 2752]


def texture(shape=(240, 320), seed=1):
    """A picture with plenty to follow: smoothed noise in all three channels."""
    noise = np.random.default_rng(seed).normal(0, 1, (*shape, 3)).astype(np.float32)
    smooth = cv2.GaussianBlur(noise, (0, 0), 1.6)
    return np.clip(128 + 90 * smooth / smooth.std(), 0, 255).astype(np.uint8)


def moved(picture, turn, shift):
    """The picture turned about its centre (degrees) and shifted; the matrix is returned with it."""
    height, width = picture.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), turn, 1.0)
    matrix[:, 2] += shift
    return cv2.warpAffine(picture, matrix, (width, height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT), matrix


@pytest.mark.parametrize("stride", [16, 32])
def test_plan_of_a_portrait_covers_chin_to_lower_edge_in_the_models_proportions(stride):
    plan = clips.plan_region(SIZE, FACE, stride, 900_000)
    left, top, right, bottom = plan["region"]
    work_w, work_h = plan["work"]
    assert work_w % stride == 0 and work_h % stride == 0
    assert (left, right) == (0, 1536) and bottom <= 2752 and 0 < top < FACE[3]
    # half a face above the chin
    assert top == pytest.approx(FACE[3] - clips.ABOVE_CHIN * (FACE[3] - FACE[1]), abs=1)
    # not stretched, and about the area asked for
    assert (bottom - top) / (right - left) == pytest.approx(work_h / work_w, abs=0.002)
    assert work_w * work_h == pytest.approx(900_000, rel=0.1)


def test_plan_of_a_wide_photo_keeps_to_the_reach_of_the_person():
    face = [3000, 300, 3400, 700]  # 400 px tall, near the right edge of a 4000 px wide photo
    plan = clips.plan_region([4000, 3000], face, 16, 900_000)
    left, _, right, _ = plan["region"]
    assert right - left == pytest.approx(clips.MAX_WIDTH * 400, abs=1)
    assert right == 4000  # moved in from the edge rather than cut off or padded
    centred = clips.plan_region([4000, 3000], [1800, 300, 2200, 700], 16, 900_000)["region"]
    assert (centred[0] + centred[2]) / 2 == pytest.approx(2000, abs=1)


def test_plan_of_a_tall_photo_is_no_taller_than_the_limit():
    face = [300, 200, 700, 600]
    plan = clips.plan_region([1000, 6000], face, 16, 900_000)
    left, top, right, bottom = plan["region"]
    assert bottom - top <= clips.MAX_ASPECT * (right - left) + 1


def test_plan_does_not_blow_up_a_small_photo():
    plan = clips.plan_region([400, 700], [150, 60, 250, 160], 16, 900_000)
    left, top, right, bottom = plan["region"]
    assert plan["work"][0] <= (right - left) + 16 and plan["work"][1] <= (bottom - top) + 16


def test_plan_refuses_a_photo_with_no_body_to_move():
    with pytest.raises(clips.Unsuitable):
        clips.plan_region([1000, 1000], [300, 100, 700, 500], 16, 900_000)


@pytest.mark.parametrize("size", [[1536, 2752], [1200, 5000], [4000, 1800]])
def test_video_avatar_plan_preserves_all_four_image_edges(size):
    plan = clips.plan_avatar(size, [200, 200, 400, 500], 16, 900_000)
    assert plan["region"] == [0, 0, *size]
    assert all(side % 16 == 0 for side in plan["work"])
    assert plan["work"][0] / plan["work"][1] == pytest.approx(size[0] / size[1], rel=0.03)


def test_face_tracking_ignores_a_more_textured_stationary_background():
    picture = texture()
    face = texture((100, 100), seed=8)
    frames = []
    for dx in (0, 2, 4):
        frame = picture.copy()
        frame[60:160, 100+dx:200+dx] = face
        frames.append(frame)
    result = clips.band_motion(np.stack(frames), (70, 150), (110, 190))
    assert result[-1, 2] == pytest.approx(4, abs=0.3)
    assert result[-1, 3] == pytest.approx(0, abs=0.3)


def test_the_head_band_runs_from_the_chin_down_a_face_height():
    plan = clips.plan_region(SIZE, FACE, 16, 900_000)
    first, last = clips.band_rows(plan, FACE)
    face_work = (FACE[3] - FACE[1]) * clips.work_scale(plan)
    assert last - first == pytest.approx(face_work, abs=2)
    assert first == pytest.approx((FACE[3] - plan["region"][1]) * clips.work_scale(plan), abs=1)


def test_a_clip_that_does_not_move_measures_as_still():
    picture = texture()
    frames = np.stack([picture] * 9)
    numbers = clips.measure(frames, picture, rows=(40, 160), face_h=100)
    assert numbers["motion_peak"] == 0 and numbers["loop_gap"] == 0 and numbers["flicker"] == 0
    assert numbers["tone_drift"] == 0 and numbers["softening"] == 1 and numbers["fidelity"] == 99
    assert numbers["band_shift"] == pytest.approx(0, abs=0.001) and numbers["band_turn"] == pytest.approx(0, abs=0.01)
    assert numbers["band_lost"] == 0


def test_the_band_follows_how_the_body_moves_and_a_hand_does_not_drag_it():
    picture = texture()
    rows, face_h = (40, 160), 100.0
    centre = np.array([160.0, 100.0])  # of the band
    steps = np.linspace(0, 1, 9)
    frames, expected = [], []
    for s in steps:
        frame, matrix = moved(picture, 2.0 * s, (5.0 * s, -3.0 * s))
        # a hand: a different texture passing across the band, over about a seventh of it
        x = int(40 + 200 * s)
        frame[70:130, x:x + 50] = texture((60, 50), seed=7)
        frames.append(frame)
        expected.append(np.hypot(*(matrix[:, :2] @ centre + matrix[:, 2] - centre)))
    numbers = clips.measure(np.stack(frames), rows=rows, face_h=face_h)
    assert numbers["band_shift"] == pytest.approx(max(expected) / face_h, abs=0.012)
    assert numbers["band_turn"] == pytest.approx(2.0, abs=0.3)
    assert numbers["band_lost"] == 0


def test_measure_catches_blur_flicker_light_drift_and_a_clip_that_is_not_a_loop():
    picture = texture()
    blurred = np.stack([cv2.GaussianBlur(picture, (0, 0), 0.1 + 2.0 * s) for s in np.linspace(0, 1, 9)])
    assert clips.measure(blurred)["softening"] < 0.3

    rng = np.random.default_rng(3)
    noisy = np.stack([np.clip(picture + rng.normal(0, 6, picture.shape), 0, 255).astype(np.uint8) for _ in range(9)])
    assert clips.measure(noisy)["flicker"] > 1  # noise of 6 levels, seen at a quarter of the size, is about 1.9
    assert clips.measure(noisy, picture)["fidelity"] == pytest.approx(20 * np.log10(255 / 6), abs=1.5)

    darkening = np.stack([np.clip(picture.astype(np.float32) - 4 * i, 0, 255).astype(np.uint8) for i in range(9)])
    numbers = clips.measure(darkening)
    assert numbers["tone_drift"] > 20 and numbers["loop_gap"] > 20
    assert numbers["flicker"] < 0.5  # a steady change of light is not shimmer


def test_a_picture_with_nothing_to_follow_is_reported_as_lost_not_as_still():
    flat = np.full((240, 320, 3), 128, np.uint8)
    numbers = clips.measure(np.stack([flat] * 4), rows=(40, 160), face_h=100)
    assert numbers["band_lost"] == 1 and "band_shift" not in numbers


def test_contact_sheet_is_one_row_of_numbered_frames():
    frames = np.stack([texture((120, 80), seed=i) for i in range(10)])
    sheet = clips.contact_sheet(frames, count=4, height=60)
    assert sheet.shape == (60, 4 * 40, 3)


def test_a_clip_written_to_video_reads_back_in_the_colours_it_was_written_in(tmp_path):
    colours = [(200, 40, 40), (40, 200, 40), (40, 40, 200), (230, 200, 30), (30, 30, 30), (225, 225, 225)]
    frames = np.stack([np.concatenate([np.full((48, 16, 3), c, np.uint8) for c in colours[i:] + colours[:i]], axis=1) for i in range(6)])
    clips.write_video(tmp_path / "clip.mp4", frames, 16)
    back, fps = clips.read_video(tmp_path / "clip.mp4")
    assert fps == 16 and back.shape == frames.shape
    assert np.abs(back.astype(int) - frames).mean() < 2  # a wrong matrix or range would be off by 10 or more


def test_every_prompt_has_a_role_and_a_mood_the_player_knows():
    assert set(clips.ROLES) == set(clips.GESTURES)
    assert {role for role, _ in clips.ROLES.values()} == {"idle", "listen", "think", "talk"}
    assert {mood for _, mood in clips.ROLES.values()} <= {"idle", "listen", "think", "calm", "explain", "emphasis", "greet", "ask"}
    assert [name for name, (role, _) in clips.ROLES.items() if role == "think"] == ["thinking"]


def test_dry_run_writes_the_picture_the_model_would_be_given(tmp_path):
    avatar = tmp_path / "avatar"
    avatar.mkdir()
    photo = cv2.resize(texture((344, 192), seed=5), (384, 688))
    cv2.imwrite(str(avatar / "photo.png"), photo)
    (avatar / "rig.json").write_text(json.dumps({"size": [384, 688], "face": [140, 60, 250, 170]}))
    assert clipgen.main([str(avatar), "--model", "wan-i2v-a14b", "--dry-run", "--out", str(tmp_path / "out")]) == 0
    target = tmp_path / "out" / "avatar" / "wan-i2v-a14b-plan"
    plan = json.loads((target / "plan.json").read_text())
    shown = cv2.imread(str(target / "first.png"))
    assert list(shown.shape[1::-1]) == plan["work"]
    assert plan["region"][1] < 170 < plan["region"][3]  # starts above the chin, reaches below it


def test_frames_that_a_model_cannot_make_are_refused():
    with pytest.raises(SystemExit):
        clipgen.parse(["avatar", "--frames", "80"])


@pytest.mark.parametrize("size, face", [(SIZE, FACE), ([2752, 1536], [1200, 200, 1600, 700]), ([1000, 1000], [400, 300, 600, 500])])
def test_live_avatar_is_shown_an_unstretched_crop_at_a_size_it_takes(size, face):
    plan = clipgen_liveavatar.plan(size, face)
    given, region = plan["given"], plan["region"]
    assert all(side % clipgen_liveavatar.STRIDE == 0 for side in given) and given[0] * given[1] <= clipgen_liveavatar.AREA
    assert plan["work"] == [side * clipgen_liveavatar.UPSCALE for side in given]
    assert 0 <= region[0] < region[2] <= size[0] and 0 <= region[1] < region[3] <= size[1]
    assert (region[2] - region[0]) / (region[3] - region[1]) == pytest.approx(given[0] / given[1], rel=0.002)
    assert region[0] <= face[0] and face[2] <= region[2] and region[1] <= face[1] and face[3] <= region[3]
    if size == SIZE:
        # The trained 704x384, losing only a sliver at the sides, where Live Avatar itself would pad top and bottom.
        assert given == [384, 704] and region[1:4:2] == [0, 2752] and region[2] - region[0] > 0.97 * size[0]


@pytest.mark.parametrize("speech_seconds", [0.0, 0.4, 2.6, 2.63, 4.0])
def test_a_drive_track_is_just_short_of_whole_live_avatar_pieces(speech_seconds):
    rate = 22050
    speech = np.full(int(speech_seconds * rate), 0.5, np.float32)
    track, blocks = clipgen_liveavatar.drive(speech, rate)
    seconds = len(track) / rate
    # How many pieces Live Avatar makes of a track (get_audio_embed_bucket_fps): exactly the ones planned for.
    assert int(seconds * clipgen_liveavatar.FPS / clipgen_liveavatar.BLOCK) + 1 == blocks
    lead = round(clipgen_liveavatar.LEAD_IN * rate) if len(speech) else 0
    assert np.array_equal(track[lead:lead + len(speech)], speech)
    assert not track[:lead].any() and not track[lead + len(speech):].any()
    if speech_seconds:
        assert seconds - clipgen_liveavatar.LEAD_IN - speech_seconds >= clipgen_liveavatar.TAIL
    else:
        assert blocks == clipgen_liveavatar.QUIET_BLOCKS


def test_a_live_avatar_clip_ends_where_it_looks_most_like_its_first_frame():
    rest = texture((200, 120), seed=2)
    frames = np.stack([np.roll(rest, shift, axis=1) for shift in [0, 9, 18, 12, 5, 1, 7, 3]])
    assert clipgen_liveavatar.settle(frames, 3) == 5
    assert clipgen_liveavatar.settle(frames, 6) == 7  # nothing before `earliest` is chosen


def test_a_live_avatar_gesture_ends_once_the_hands_are_still():
    rest = texture((200, 120), seed=2)
    shifts = [3 * min(index, 10) for index in range(21)]  # moving until frame 10, then still
    frames = np.stack([np.roll(rest, shift, axis=1) for shift in shifts])
    assert 11 <= clipgen_liveavatar.calm(frames, 4) <= 13
    assert clipgen_liveavatar.calm(frames, 15) == 15  # nothing before `earliest` is chosen


@pytest.mark.parametrize("offset, closed", [(4, True), (20, False)])
def test_a_bridge_meets_the_gesture_and_ends_on_its_first_frame(offset, closed):
    rest = texture((64, 48), seed=3)
    gesture = np.stack([np.roll(rest, shift, axis=1) for shift in [0, 4, 8, 8]])
    path = np.linspace(0, 1, 9)[:, None, None, None]
    # What Wan makes: the way from the gesture's last frame to its first, a little too bright throughout.
    bridge = np.clip(gesture[-1] * (1 - path) + gesture[0] * path + offset, 0, 255).astype(np.uint8)
    frames, numbers = clipbridge.join(gesture, bridge)
    assert len(frames) == len(gesture) + len(bridge) - 1 and np.array_equal(frames[:len(gesture)], gesture)
    own_step = np.abs(bridge[1].astype(np.float32) - bridge[2]).mean()
    assert np.abs(frames[len(gesture)].astype(np.float32) - gesture[-1]).mean() <= own_step + 0.5
    assert numbers["closed"] == closed and abs(numbers["end_gap"] - offset) < 0.15 * offset  # less where 255 clips
    last = np.abs(frames[-1].astype(np.float32) - gesture[0]).mean()
    assert last < 0.01 if closed else last > 0.85 * offset
