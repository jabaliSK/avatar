"""The end-of-reply walk (tail_path), the blink check (has_blinks) and the rest-face options: the body goes back to its rest pose the
way a person does, without a cut at the last word and without whipping a hand back."""
import numpy as np
import pytest

from app.head_engine import TAIL_CRUISE, TAIL_REST_SPEED, has_blinks, rest_options, tail_caps, tail_path

N = 81


@pytest.mark.parametrize("last,direction", [(80, 1), (80, -1), (40, 1), (40, -1), (3, 1), (1, -1)])
def test_walk_ends_at_rest_and_stays_inside_the_template(last, direction):
    path = tail_path(last, direction, N)
    assert path[-1] == 0
    assert all(0 <= i <= N - 1 for i in path)


def test_walk_is_empty_when_the_body_is_already_at_rest():
    assert tail_path(0, -1, N) == []


def test_walk_carries_on_before_it_turns_around():
    # moving away from the rest pose it keeps going for a moment instead of reversing on the spot
    assert tail_path(40, 1, N)[0] > 40


def test_walk_is_never_faster_than_cruise_and_brakes_into_the_rest():
    steps = np.abs(np.diff([40] + tail_path(40, -1, N)))
    assert steps.max() <= np.ceil(TAIL_CRUISE)
    assert steps[-1] <= 1


def test_caps_ramp_up_from_the_rest_pose():
    caps = tail_caps(N)
    assert caps[0] == TAIL_REST_SPEED and caps.max() == TAIL_CRUISE
    assert (np.diff(caps[:7]) > 0).all()


def test_a_fast_stretch_of_the_template_is_walked_slowly_and_announced_early():
    motion = np.ones(N)
    motion[29] = 4.0  # the picture changes four times as much between frames 29 and 30
    caps = tail_caps(N, motion)
    assert caps[30] == 1.0  # leaving frame 30 plays that change: never faster than the template itself
    assert caps[30] < caps[31] < caps[32] < caps[33]  # the walk brakes on the way there
    assert len(tail_path(60, -1, N, motion)) > len(tail_path(60, -1, N))


def test_has_blinks_sees_eyes_closing():
    eyes = np.full((N, 2), 0.3)
    assert not has_blinks({"eye_open_lst": eyes})
    eyes[30:33] = 0.1
    assert has_blinks({"eye_open_lst": eyes})


def test_only_a_loop_placed_on_the_real_face_gets_the_aligned_rest_options():
    assert rest_options({"aligned": True}) == {"scale": "rest", "exp": "nonlip"}
    assert rest_options({"aligned": False}) == rest_options({}) == {"scale": "frame", "exp": "eyes"}  # caches before format 4 have no flag
    assert rest_options({"aligned": True}, {"exp": "eyes"}) == {"scale": "rest", "exp": "eyes"}
