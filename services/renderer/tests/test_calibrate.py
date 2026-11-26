"""Placing the live head on the template's real face (app/calibrate.py, Loop.align), without a GPU or MediaPipe."""
import queue
from types import SimpleNamespace

import numpy as np
import pytest

from app import avatar, calibrate
from app.calibrate import ANCHORS, CROP, CalibrationError, calibrate_loop, calibration_matrices, fit_similarity

RENDER, REAL = 10, 100  # what the first pixel of a silent render / a real crop says, see _Landmarker


def _face():
    """Sixteen anchor points spread over a face in the 512 crop."""
    return np.random.default_rng(3).uniform([170, 170], [340, 380], (len(ANCHORS), 2))


def _placement(i):
    """Frame i of a smooth drift of the real face against the render: +0.4 % scale, +0.3 degrees, (+0.5, -0.4) px per frame."""
    return calibrate._matrix(np.array([0.004 * i, np.radians(0.3) * i, 0.5 * i, -0.4 * i]))


def _apply(M, pts):
    return pts @ M[:2, :2].T + M[:2, 2]


class _Landmarker:
    """Stands in for MediaPipe: the face it finds is looked up by the value of the image's first pixel (None: no face)."""

    def __init__(self, faces):
        self.faces = faces

    def __call__(self, rgb):
        pts = self.faces[int(rgb[0, 0, 0])]
        if pts is None:
            return np.zeros((0, 478, 3))
        out = np.zeros((1, 478, 3))
        out[0, list(ANCHORS), :2] = pts / CROP
        return out


def _scene(n=9):
    """n silent renders of one face, n real crops of it as _placement(i) moves it, and the landmarker that finds the faces."""
    a, faces, renders, frames = _face(), {}, [], []
    for i in range(n):
        renders.append(np.full((CROP, CROP, 3), RENDER + i, np.uint8))
        frames.append(np.full((CROP, CROP, 3), REAL + i, np.uint8))
        faces[RENDER + i], faces[REAL + i] = a, _apply(_placement(i), a)
    return renders, frames, faces


def test_fit_recovers_a_known_similarity():
    src, T = _face(), _placement(8)
    M, rms = fit_similarity(src, _apply(T, src))
    assert np.allclose(M, T, atol=1e-6) and rms < 1e-6


def test_fit_is_not_dragged_by_a_point_that_disagrees():
    src, T = _face(), _placement(8)
    dst = _apply(T, src)
    dst[3] += (25.0, -20.0)  # a brow the render does not raise
    M, _ = fit_similarity(src, dst)
    assert np.allclose(M, T, atol=1e-6)


def test_frames_without_a_usable_fit_get_the_nearest_placement():
    renders, frames, faces = _scene()
    faces[REAL + 4] = None  # no face found on the real crop of frame 4
    faces[REAL + 6] = _apply(_placement(6), _face()) * 1.5  # a fit that far off is a detection failure
    mats, stats = calibration_matrices(renders, range(9), frames, [np.eye(3)] * 9, _Landmarker(faces))
    assert stats["fitted"] == 7
    for i, M in enumerate(mats):
        assert np.allclose(M, _placement(i), atol=1e-6), i


def test_too_few_fits_is_an_error_not_a_guess():
    renders, frames, faces = _scene()
    for i in range(1, 9):
        faces[REAL + i] = None
    with pytest.raises(CalibrationError):
        calibration_matrices(renders, range(9), frames, [np.eye(3)] * 9, _Landmarker(faces))


class _Loop:
    name = "t"

    def __init__(self, frames, M_c2o_lst):
        self.frames, self.info = frames, {"M_c2o_lst": M_c2o_lst}

    def __len__(self):
        return len(self.frames)


def _pipeline(renders):
    """A SpeechPipeline that hands back the given silent renders as its frames."""

    class Pipeline:
        def __init__(self, head, keep_render=False):
            self.frames, self.error = queue.Queue(), None

        def start(self, loop, **kw):
            pass

        def push(self, pcm):
            pass

        def finish(self):
            for i, r in enumerate(renders):
                self.frames.put(SimpleNamespace(idx=i, fidx=i, render=r))
            self.frames.put(None)

        def join(self):
            pass

    return Pipeline


@pytest.mark.parametrize("rows", [3, 2])
def test_the_fit_is_folded_into_the_paste_transform_and_keeps_its_shape(monkeypatch, rows):
    renders, frames, faces = _scene(6)
    monkeypatch.setattr("app.speech_pipeline.SpeechPipeline", _pipeline(renders))
    M_old = np.array([[0.8, 0.0, 20.0], [0.0, 0.8, 30.0], [0.0, 0.0, 1.0]], np.float32)  # crop -> frame, not the identity
    out, stats = calibrate_loop(_Loop(frames, [M_old[:rows]] * 6), None, _Landmarker(faces))
    assert stats["fitted"] == 6
    for i, M in enumerate(out):
        assert M.shape == (rows, 3) and M.dtype == np.float32
        assert np.allclose(M, (M_old.astype(np.float64) @ _placement(i))[:rows], atol=1e-3)  # a render point r lands at M_old @ (S @ r)


def test_the_engine_must_render_every_template_frame_of_silence(monkeypatch):
    renders, frames, faces = _scene(6)
    monkeypatch.setattr("app.speech_pipeline.SpeechPipeline", _pipeline(renders[:3]))
    with pytest.raises(CalibrationError):
        calibrate_loop(_Loop(frames, [np.eye(3, dtype=np.float32)] * 6), None, _Landmarker(faces))


class _AlignLoop:
    """The parts of a Loop that Loop.align touches."""
    name = "t"
    align = avatar.Loop.align

    def __init__(self, n=5, is_image=False):
        self.n, self.coloured = n, 0
        self.info = {"M_c2o_lst": ["old"], "is_image_flag": is_image, "aligned": True}

    def __len__(self):
        return self.n

    def match_colour(self, head):
        self.coloured += 1


def test_align_measures_the_render_the_loop_will_play_and_keeps_the_new_placement(monkeypatch):
    seen = []

    def calibrate_stub(loop, head, landmarker, **kw):
        seen.append(loop.info["aligned"])  # the silent render is made with the aligned rest options
        return ["new"], {"fitted": 5}

    monkeypatch.setattr(avatar, "calibrate_loop", calibrate_stub)
    loop = _AlignLoop()
    assert loop.align(None, None) == {"fitted": 5}
    assert seen == [True] and loop.info["aligned"] is True and loop.info["M_c2o_lst"] == ["new"] and loop.coloured == 1


def test_align_falls_back_to_the_old_placement_when_it_cannot_measure(monkeypatch):
    def fail(*a, **k):
        raise CalibrationError("no face")

    monkeypatch.setattr(avatar, "calibrate_loop", fail)
    loop = _AlignLoop()
    assert loop.align(None, None) is None
    assert loop.info["aligned"] is False and loop.info["M_c2o_lst"] == ["old"] and loop.coloured == 1


def test_a_single_image_is_only_colour_matched(monkeypatch):
    monkeypatch.setattr(avatar, "calibrate_loop", lambda *a, **k: pytest.fail("an image has no template motion to follow"))
    loop = _AlignLoop(n=1, is_image=True)
    assert loop.align(None, None) is None and loop.info["aligned"] is False and loop.coloured == 1
