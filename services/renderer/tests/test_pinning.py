"""pin_static: a generated clip keeps the photo's pixels where nothing moves and its own where something does."""
import numpy as np
import pytest

from app.pinning import pin_static

H, W, N = 96, 64, 12
CAST = np.array([-1, -3, 2])


def _photo():
    rng = np.random.default_rng(0)
    return rng.integers(60, 190, (H, W, 3)).astype(np.uint8)


def _clip(photo, square_from=4):
    """The photo with a generator-style colour cast and shimmer; from frame `square_from` a bright square slides right."""
    rng = np.random.default_rng(1)
    frames = []
    for t in range(N):
        f = photo.astype(np.float32) + CAST + rng.normal(0, 1.2, photo.shape)
        if t >= square_from:
            x = 10 + 3 * (t - square_from)
            f[40:60, x:x + 14] = 250
        frames.append(np.clip(f, 0, 255).astype(np.uint8))
    return np.stack(frames)


def test_still_pixels_are_the_photos():
    photo = _photo()
    out = pin_static(_clip(photo), photo)
    assert out.shape == (N, H, W, 3) and out.dtype == np.uint8
    assert (out[0] == photo).all()  # nothing moves around frame 0
    far = (slice(0, 30), slice(None))  # rows the square never reaches
    for t in range(N):
        assert (out[t][far] == photo[far]).all()


def test_moving_pixels_come_from_the_clip_without_its_colour_cast():
    photo = _photo()
    clip = _clip(photo)
    out = pin_static(clip, photo)
    t = N - 1
    x = 10 + 3 * (t - 4)
    inside = out[t, 45:55, x + 3:x + 11].astype(int)
    assert abs(inside.mean() - (250 - CAST.mean())) < 4  # the square, with the generator's cast removed; not the photo's texture
    uncovered = out[t, 45:55, 12:15].astype(int)  # the square left this spot: it shows the clip's background again, not frozen paint
    assert np.abs(uncovered - photo[45:55, 12:15]).mean() < 4


def test_size_mismatch_is_rejected():
    photo = _photo()
    with pytest.raises(ValueError):
        pin_static(_clip(photo), photo[:-2])
