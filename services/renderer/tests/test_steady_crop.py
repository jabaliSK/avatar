"""SteadyCrop: the crop of a talking face follows the eyes and not the jaw, so a wide-open mouth does not swell the head."""
import numpy as np
import pytest

from core.utils.crop import SteadyCrop, crop_image

CROP = dict(dsize=512, scale=2.3, vy_ratio=-0.125, flag_do_rot=True, pt_crop_flag=False)  # what Source2Info asks for
IMAGE = np.zeros((400, 400, 3), np.uint8)
HOME = np.array((200.0, 150.0))


def face(jaw=0.0, size=1.0, shift=HOME, roll=0.0):
    """A 203-point face (landmark203 layout) with the eyes 56 px apart. `jaw` px lowers the chin, the lower lip and, a little, the mouth corners."""
    p = np.zeros((203, 2))
    angle = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    for first, x in ((0, -28.0), (24, 28.0)):  # eyes: corners are points 0 and 12 (left), 24 and 36 (right)
        p[first:first + 24] = np.stack([x + 8 * np.cos(angle), 4 * np.sin(angle)], axis=1)
    arc = np.linspace(0, np.pi, 39)
    p[106:145] = np.stack([-55 * np.cos(arc), 5 + 80 * np.sin(arc) + jaw * np.sin(arc) ** 4], axis=1)  # contour: ear, jaw, chin, jaw, ear
    p[145:185] = np.stack([np.tile(np.linspace(-45, 45, 20), 2), -25 + 3 * np.sin(np.arange(40))], axis=1)  # brows
    p[185:203] = np.stack([np.linspace(-6, 6, 18), np.linspace(0, 32, 18)], axis=1)  # nose
    lips = np.stack([np.linspace(-16, 16, 58), np.full(58, 52.0) + jaw * 0.5], axis=1)
    lips[0] = (-16, 52 + 0.2 * jaw)   # point 48, a mouth corner
    lips[18] = (16, 52 + 0.2 * jaw)   # point 66, the other
    p[48:106] = lips
    rot = np.array([[np.cos(roll), -np.sin(roll)], [np.sin(roll), np.cos(roll)]])
    return ((p @ rot.T) * size + np.asarray(shift)).astype(np.float32)


def crop(pts, steady=None):
    return crop_image(IMAGE, pts, steady=steady, **CROP)["M_c2o"]  # crop -> frame


def zoom(m):
    return float(np.hypot(m[0, 0], m[1, 0]))


def turn(m):
    return float(np.arctan2(m[1, 0], m[0, 0]))


def centre(m):
    return (m @ np.array([256.0, 256.0, 1.0]))[:2]


def calibrate(pts):
    return SteadyCrop(pts, scale=2.3, vx_ratio=0, vy_ratio=-0.125, use_lip=True)


def test_the_calibration_frame_gets_exactly_the_regular_crop():
    pts = face(jaw=6, size=1.3, roll=0.12, shift=(180.0, 170.0))
    assert np.allclose(crop(pts, calibrate(pts)), crop(pts), atol=1e-3)


@pytest.mark.parametrize("jaw", [10, 25, 40])
def test_an_open_mouth_swells_the_regular_crop_but_not_the_steady_one(jaw):
    closed, open_ = face(), face(jaw)
    steady = calibrate(closed)
    assert zoom(crop(open_)) > 1.08 * zoom(crop(closed))  # the problem: the crop grows with the jaw (10-17 % in a real talking clip)
    assert zoom(crop(open_, steady)) == pytest.approx(zoom(crop(closed, steady)), rel=1e-4)
    assert centre(crop(open_, steady)) == pytest.approx(centre(crop(closed, steady)), abs=1e-2)
    assert turn(crop(open_, steady)) == pytest.approx(turn(crop(closed, steady)), abs=1e-4)


def test_the_steady_crop_still_follows_the_head():
    closed = face()
    steady = calibrate(closed)
    before = crop(closed, steady)
    moved = crop(face(jaw=20, size=1.2, roll=0.2, shift=(230.0, 130.0)), steady)  # closer to the camera, turned, moved, mouth open
    rot = np.array([[np.cos(0.2), -np.sin(0.2)], [np.sin(0.2), np.cos(0.2)]])
    assert zoom(moved) == pytest.approx(1.2 * zoom(before), rel=1e-3)
    assert turn(moved) - turn(before) == pytest.approx(0.2, abs=1e-3)
    assert centre(moved) == pytest.approx(rot @ (centre(before) - HOME) * 1.2 + (230.0, 130.0), abs=0.1)


def test_a_collapsed_eye_distance_cannot_shrink_the_crop_to_nothing():
    closed = face()
    steady = calibrate(closed)
    lost = closed.copy()
    lost[24:48] = lost[0:24]  # both eyes on the same spot: a lost track
    m = crop(lost, steady)
    assert np.isfinite(m).all()
    assert zoom(m) == pytest.approx(0.5 * zoom(crop(closed, steady)), rel=1e-3)
