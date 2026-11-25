"""The body rig's hands, on pictures made for the purpose: a hand is a skin-coloured ellipse on dark cloth with a weave. MediaPipe
is stood in for (it only finds the hand); everything that cuts a hand out, makes what is behind it and judges it is the real
code."""

import cv2
import numpy as np
import pytest

from photostage import body

SKIN, CLOTH = (120, 150, 205), (62, 62, 66)  # BGR
FACE = 100.0  # the face's height in pixels: every limit of a hand is measured in it
AXES = (44, 26)  # a hand's half length and half width
PASSES = {"covered": 1.0, "solidity": 0.95, "cut": 0.0, "leftover": 0.0, "ghost": 0.0, "halo": 5.0, "flat": 0.0, "distortion": 1.0}


def weave(shape, grain=8.0, seed=1):
    """Dark cloth: a slow change of tone across it, and a fine grain."""
    rows, columns = shape
    raw = np.random.default_rng(seed).normal(0, grain, (rows, columns)).astype(np.float32)
    noise = cv2.GaussianBlur(raw, (0, 0), 0.6)[..., None]
    tone = np.linspace(-10, 10, columns, dtype=np.float32)[None, :, None]
    return np.clip(np.array(CLOTH, np.float32) + tone + noise, 0, 255).astype(np.uint8)


class Scene:
    """A picture with hands painted into it, and what MediaPipe would make of it for body.build_hands."""

    def __init__(self, monkeypatch, shape=(500, 800), grain=8.0):
        self.work = weave(shape, grain)
        self.hands = []
        monkeypatch.setattr(body, "find_hands", lambda work, pose, face: [np.full((21, 2), index, np.float32) for index in range(len(self.hands))])
        monkeypatch.setattr(body, "hand_matte", lambda work, points: self.matte(int(points[0, 0])))

    def add(self, centre, axes=AXES):
        cv2.ellipse(self.work, centre, axes, 0, 0, 360, SKIN, -1)
        self.work[:] = cv2.GaussianBlur(self.work, (0, 0), 0.8)
        self.hands.append((centre, axes))
        return len(self.hands) - 1

    def matte(self, index):
        """The hand as body.hand_matte describes it: its window, its cut-out and how skin-like each pixel is."""
        (x, y), (half_length, half_width) = self.hands[index]
        size = 2.0 * half_length
        pad = int(round(0.6 * size))
        left, top, right, bottom = x - half_length - pad, y - half_width - pad, x + half_length + pad + 1, y + half_width + pad + 1
        window = self.work[top:bottom, left:right]
        binary = np.zeros(window.shape[:2], np.uint8)
        cv2.ellipse(binary, (x - left, y - top), (half_length, half_width), 0, 0, 360, 1, -1)
        lab = cv2.cvtColor(window, cv2.COLOR_BGR2LAB).astype(np.float32)
        cloth, skin = np.median(lab[binary == 0][:, 1:], axis=0), np.median(lab[binary > 0][:, 1:], axis=0)
        # The wrist is at the hand's left end, and the knuckles are towards its right.
        local = np.tile(np.array([x - half_length - left, y - top], np.float32), (21, 1))
        local[[5, 9, 13, 17]] += [1.2 * half_length, 0]
        return {"window": (left, top, right, bottom), "binary": binary, "score": body.skin_score(lab, cloth, skin)[0], "size": size,
                "solidity": 0.95, "covered": 1.0, "limit": np.zeros(binary.shape, bool), "cloth": cloth, "skin": skin, "local": local}

    def parts(self, index=0):
        """What make_plate and check_hand are given for a hand: its window, matte, rig, layer and plate."""
        matte = self.matte(index)
        left, top, right, bottom = matte["window"]
        window = self.work[top:bottom, left:right]
        alpha = body.layer_alpha(matte["binary"], matte["score"], matte["size"])
        others = np.zeros(alpha.shape, bool)
        plate, bare = body.make_plate(window, alpha, matte["size"], matte["score"], others)
        return window, matte, body.hand_rig(matte, FACE), alpha, plate, bare, others

    def build(self):
        return body.build_hands(self.work, FACE, (0, 0, 10, 10), {})


def fine(image):
    """What is left of a picture when its smooth tone is taken away: the grain."""
    gray = image.astype(np.float32).mean(2)
    return gray - cv2.GaussianBlur(gray, (0, 0), 3.0)


def test_a_layer_is_the_shaved_matte_with_a_soft_edge_in_the_steps_it_is_stored_in():
    binary = np.zeros((120, 120), np.uint8)
    binary[30:90, 30:90] = 1
    alpha = body.layer_alpha(binary, np.zeros((120, 120), np.float32), 60.0)
    assert alpha[60, 60] == 1.0 and alpha[5, 5] == 0.0
    assert 0.0 < alpha[60, 30] < 1.0  # the matte's own rim is shaved: it is a blend of skin and cloth
    assert alpha[60, 15] == 0.0 or alpha[60, 15] < alpha[60, 30]
    assert np.array_equal(alpha, np.round(alpha * 255) / 255)


def test_skin_beside_the_matte_that_the_colour_test_passed_over_goes_in_the_layer():
    binary = np.zeros((120, 120), np.uint8)
    binary[30:90, 30:90] = 1
    score = np.zeros((120, 120), np.float32)
    score[30:90, 90:94] = 1.0  # a finger's shaded underside, left out of the matte
    alpha = body.layer_alpha(binary, score, 60.0)
    assert alpha[60, 91] > 0.5
    assert body.layer_alpha(binary, np.zeros((120, 120), np.float32), 60.0)[60, 91] < 0.1


def test_push_pull_spreads_what_is_known_over_the_hole_and_keeps_it_where_it_is_known():
    image = np.zeros((96, 96, 3), np.float32)
    image[:, :30] = 100.0
    image[:, 66:] = 200.0
    weight = np.zeros((96, 96), np.float32)
    weight[:, :30] = weight[:, 66:] = 1.0
    filled = body.push_pull(image, weight)
    assert np.allclose(filled[:, :30], 100.0, atol=1.0) and np.allclose(filled[:, 66:], 200.0, atol=1.0)
    middle = filled[48, 30:66, 0]
    assert middle.min() >= 99.0 and middle.max() <= 201.0 and np.all(np.diff(middle) >= -0.5)


def test_a_plate_carries_the_cloths_tone_and_grain_over_the_hand(monkeypatch):
    scene = Scene(monkeypatch)
    clean = scene.work.copy()
    scene.add((400, 250))
    window, matte, _, alpha, plate, bare = scene.parts()[:6]
    left, top, right, bottom = matte["window"]
    under = alpha > 0.9
    tone = lambda image: cv2.GaussianBlur(image.astype(np.float32), (0, 0), 6.0)
    # The tone under the hand is the cloth's own there ...
    assert np.abs(tone(plate) - tone(clean[top:bottom, left:right]))[under].mean() < 3.0
    # ... and so is its fineness. A smooth fill has a thousandth of the cloth's; the copied grain has a good part of it (a
    # copy is only made from plain cloth, so some of the weave is left out).
    assert fine(plate)[cv2.erode(under.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0].std() > 0.25 * fine(clean).std()
    assert bare.sum() < 0.02 * (alpha > 0.001).sum()  # hardly anything is left without grain


def test_no_plate_is_made_where_there_is_no_cloth_to_read():
    window = np.full((120, 120, 3), SKIN, np.uint8)
    full = np.ones((120, 120), np.float32)
    assert body.make_plate(window, full, 100.0, full, np.zeros((120, 120), bool)) is None


def test_cloth_without_a_grain_leaves_nothing_to_copy_and_nothing_missing(monkeypatch):
    scene = Scene(monkeypatch, grain=0.0)
    scene.add((400, 250))
    bare = scene.parts()[5]
    assert not bare.any()


def test_a_patch_of_plate_without_grain_stops_a_hand_but_specks_do_not(monkeypatch):
    scene = Scene(monkeypatch)
    scene.add((400, 250))
    window, matte, rig, alpha, plate, _, others = scene.parts()
    y, x = (np.array(matte["binary"].shape) // 2).tolist()
    # Sliding the hand up uncovers the cloth at its lower rim, which is where the missing grain is put.
    rim = slice(y + 14, y + 25), slice(x - 6, x + 6)
    patch, specks = np.zeros(alpha.shape, bool), np.zeros(alpha.shape, bool)
    patch[rim] = True
    specks[rim][::4, ::4] = True
    specks[rim][1::4, 2::4] = True
    measured = {name: body.check_hand(window.astype(np.float32), alpha, plate, bare, matte, rig, others, 1.0)
                for name, bare in (("patch", patch), ("specks", specks))}
    assert measured["patch"]["flat"] > body.LIMITS["flat"] and "flat" in body.failures(measured["patch"])
    assert measured["specks"]["flat"] == 0.0


def test_a_hand_on_plain_cloth_moves_the_whole_range(monkeypatch):
    scene = Scene(monkeypatch)
    scene.add((400, 250))
    layers, report = scene.build()
    rig = body.hand_rig(scene.matte(0), FACE)
    assert report[0]["kept"] and report[0]["amount"] == 1.0, report
    assert layers[0]["turn"] == pytest.approx(rig["turn"]) and layers[0]["slide"] == pytest.approx(rig["slide"])
    assert layers[0]["alpha"].shape == layers[0]["plate"].shape[:2]
    # The texture is clamped at its border, so nothing of the hand may reach it.
    alpha = layers[0]["alpha"]
    assert not (alpha[0].any() or alpha[-1].any() or alpha[:, 0].any() or alpha[:, -1].any())


@pytest.mark.parametrize("fault, amount", [({"flat": 0.025}, 0.35), ({"ghost": 0.009}, 0.5), ({"distortion": 3.0}, 0.5)])
def test_a_hand_that_shows_a_fault_at_full_range_gets_the_largest_range_that_does_not(monkeypatch, fault, amount):
    scene = Scene(monkeypatch)
    scene.add((400, 250))
    tried = []

    def check(photo, alpha, plate, bare, matte, rig, others, share):
        tried.append(share)
        return {**PASSES, **{name: 1.0 + (value - 1.0) * share if name == "distortion" else value * share for name, value in fault.items()}}

    monkeypatch.setattr(body, "check_hand", check)
    layers, report = scene.build()
    rig = body.hand_rig(scene.matte(0), FACE)
    assert report[0]["kept"] and report[0]["amount"] == amount and tried == [a for a in body.AMOUNTS if a >= amount]
    assert layers[0]["turn"] == pytest.approx(rig["turn"] * amount)
    assert layers[0]["slide"] == pytest.approx(rig["slide"] * amount)


def test_a_hand_stays_still_when_even_the_smallest_range_shows_a_fault(monkeypatch):
    scene = Scene(monkeypatch)
    scene.add((400, 250))
    monkeypatch.setattr(body, "check_hand", lambda *args: {**PASSES, "flat": 0.05 * args[-1]})
    layers, report = scene.build()
    assert layers == [] and not report[0]["kept"] and "flat" in report[0]["reason"]


def test_a_fault_that_a_smaller_range_does_not_cure_is_not_tried_again(monkeypatch):
    scene = Scene(monkeypatch)
    scene.add((400, 250))
    tried = []
    monkeypatch.setattr(body, "check_hand", lambda *args: tried.append(args[-1]) or {**PASSES, "leftover": 0.5})
    layers, report = scene.build()
    assert layers == [] and tried == [1.0] and "leftover" in report[0]["reason"]


def test_a_hand_slides_at_most_a_tenth_of_its_ramp_and_a_large_hand_no_more_than_the_cap(monkeypatch):
    rigs = []
    for axes in (AXES, (110, 60)):
        scene = Scene(monkeypatch)
        scene.add((400, 250), axes)
        rigs.append(body.hand_rig(scene.matte(0), FACE))
    small, large = rigs
    assert small["slide"] == pytest.approx(body.SLIDE_RAMP * small["ramp"]) and small["slide"] < body.SLIDE * FACE
    assert large["slide"] == pytest.approx(body.SLIDE * FACE) and large["slide"] < body.SLIDE_RAMP * large["ramp"]


def test_a_pose_stretches_a_hand_unevenly_only_where_the_wrists_ramp_blends_it_in(monkeypatch):
    scene = Scene(monkeypatch)
    scene.add((400, 250))
    _, _, rig, alpha, *_ = scene.parts()

    def worst(slide):
        return max(body.distortion_of(alpha.shape, rig, turn * rig["turn"], (x * slide, y * slide), alpha) for turn, x, y in body.POSES)

    assert body.distortion_of(alpha.shape, rig, 0.0, (0.0, 0.0), alpha) == pytest.approx(1.0)
    # With no ramp a hand moves as one piece, and nothing of it is stretched.
    assert body.distortion_of(alpha.shape, dict(rig, ramp=0.01), 0.3, (3.0, -2.0), alpha) == pytest.approx(1.0, abs=0.01)
    # Only what shows counts: beyond the ramp the hand moves as one piece, within it the hand is stretched.
    weight = body.ramp_weight(alpha.shape, rig["pivot"], rig["axis"], rig["ramp"])
    pose = (0.3, (3.0, -2.0))
    assert body.distortion_of(alpha.shape, rig, *pose, alpha * (weight >= 1.0)) == pytest.approx(1.0, abs=0.01)
    assert body.distortion_of(alpha.shape, rig, *pose, alpha * ((weight > 0.0) & (weight < 1.0))) > 1.3
    assert body.distortion_of(alpha.shape, rig, *pose, np.zeros_like(alpha)) == 1.0
    # The hand's own range is within the limit at its worst corner; a slide of a third of the ramp (as the range once was) is not.
    assert worst(rig["slide"]) <= body.LIMITS["distortion"] < worst(rig["ramp"] / 3.0)


def test_a_hand_whose_slide_would_stretch_it_gets_a_smaller_range(monkeypatch):
    scene = Scene(monkeypatch)
    scene.add((400, 250))
    real = body.hand_rig
    monkeypatch.setattr(body, "hand_rig", lambda matte, face: dict(real(matte, face), slide=0.3 * real(matte, face)["ramp"]))
    layers, report = scene.build()
    assert report[0]["kept"] and 0 < report[0]["amount"] < 1.0, report
    assert report[0]["metrics"]["distortion"] <= body.LIMITS["distortion"]


def test_two_hands_apart_both_move_and_two_that_could_meet_do_not(monkeypatch):
    # The gap between the hands' edges is 36 px in the second case, which a hand's range (a 4 px slide and a turn) does not
    # cross, and 8 px in the third, which it does.
    for position, kept in (((650, 250), 2), ((316, 250), 2), ((288, 250), 1)):
        scene = Scene(monkeypatch)
        scene.add((200, 250))
        scene.add(position, (36, 22))
        monkeypatch.setattr(body, "check_hand", lambda *args: dict(PASSES))
        layers, report = scene.build()
        assert len(layers) == kept, report
        if kept == 1:
            # The larger hand keeps its place, and the one that it could be drawn over keeps still.
            assert report[0]["kept"] and not report[1]["kept"] and "another hand" in report[1]["reason"]
