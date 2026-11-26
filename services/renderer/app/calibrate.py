"""Per-frame placement of the live head on the real face of a body-loop frame.

The live head is the rest face re-posed with the pose, scale and shift that the motion extractor estimated for each template frame.
Those estimates are a few percent and a few pixels away from the real face, and the error is different in every frame (Gulf Man
Black: head scale +-2 % with 4 % frame-to-frame jumps, shift 4-5 px), so a head pasted at the crop of the real face swells, shrinks
and slides while it talks: the wobble.

The head is therefore measured against the real face once, when a loop is registered. For every template frame the live head is
rendered on silence, MediaPipe finds the same face points on that render and on the frame's real crop, and the 2D similarity (scale,
roll, shift) that carries the first onto the second is folded into that frame's paste transform (M_c2o). The head then sits on the
real face and follows its motion, which is smooth, and everything the audio adds (the mouth) goes on top of that.
"""
import logging

import cv2
import numpy as np

CROP = 512

# MediaPipe FaceMesh points that hardly move while speaking: eye corners, nose bridge and tip, brow ends, cheek bones.
ANCHORS = (33, 133, 362, 263, 6, 168, 1, 4, 70, 105, 300, 334, 50, 280, 205, 425)

MAX_SCALE_STEP = 0.15  # a fit further off than this (scale), 10 degrees or 40 px is a detection failure, not a correction
MAX_ROLL_DEG = 10.0
MAX_SHIFT_PX = 40.0

log = logging.getLogger("kka.calibrate")


class CalibrationError(RuntimeError):
    """The live head could not be measured against the template (no face found on enough frames)."""


def anchors(landmarker, rgb):
    """The ANCHORS of the face in a 512x512 RGB crop as float64 [16, 2] pixels, or None when MediaPipe finds no face."""
    res = landmarker(np.ascontiguousarray(rgb))  # [n, 478, 3], x/w and y/h
    if res.ndim != 3 or res.shape[0] == 0:
        return None
    return res[0][list(ANCHORS), :2].astype(np.float64) * np.array([rgb.shape[1], rgb.shape[0]], np.float64)


def _umeyama(src, dst, w):
    w = w / w.sum()
    mu_s, mu_d = w @ src, w @ dst
    s0, d0 = src - mu_s, dst - mu_d
    U, S, Vt = np.linalg.svd((w[:, None] * d0).T @ s0)
    D = np.eye(2)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[1, 1] = -1.0
    R = U @ D @ Vt
    scale = float(np.trace(np.diag(S) @ D) / (w * (s0 ** 2).sum(1)).sum())
    M = np.eye(3)
    M[:2, :2] = scale * R
    M[:2, 2] = mu_d - scale * R @ mu_s
    return M


def fit_similarity(src, dst, iters=4):
    """Similarity (scale, roll, shift) as a 3x3 matrix that takes src points onto dst points, least squares with Tukey weights so a
    point that disagrees with the others (a brow the render does not raise) does not drag the fit. Also returns the rms error."""
    M = _umeyama(src, dst, np.ones(len(src)))
    for _ in range(iters):
        err = np.linalg.norm(dst - (src @ M[:2, :2].T + M[:2, 2]), axis=1)
        sigma = max(1.4826 * float(np.median(err)), 0.5)
        u = err / (4.685 * sigma)
        w = np.where(u < 1.0, (1.0 - u ** 2) ** 2, 0.0)
        if (w > 0).sum() < 6:  # too few agreeing points to fit four parameters
            break
        M = _umeyama(src, dst, w)
    err = np.linalg.norm(dst - (src @ M[:2, :2].T + M[:2, 2]), axis=1)
    return M, float(np.sqrt((err ** 2).mean()))


# The similarity is smoothed and interpolated as (log scale, roll, shift of the face centre): about the centre of the face the four
# numbers do not trade off against each other the way scale, roll and the shift at the crop's corner do.
CENTRE = np.array([CROP * 0.5, CROP * 0.56])  # the middle of the face in the crop (compositor.elliptical_mask)


def _params(M):
    return np.array([np.log(np.hypot(M[0, 0], M[1, 0])), np.arctan2(M[1, 0], M[0, 0]), *(M[:2, :2] @ CENTRE + M[:2, 2] - CENTRE)])


def _matrix(p):
    s, a = np.exp(p[0]), p[1]
    M = np.eye(3)
    M[:2, :2] = s * np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    M[:2, 2] = CENTRE + p[2:] - M[:2, :2] @ CENTRE
    return M


def _median3(x):
    if len(x) < 3:
        return x
    p = np.concatenate([x[:1], x, x[-1:]])
    return np.median(np.stack([p[:-2], p[1:-1], p[2:]]), axis=0)


def real_crop(frame, M_c2o):
    """The 512x512 crop of the frame that its paste transform M_c2o (crop -> frame) maps the head render onto."""
    inv = cv2.invertAffineTransform(np.asarray(M_c2o, np.float64)[:2])
    return cv2.warpAffine(np.ascontiguousarray(frame), inv, (CROP, CROP), flags=cv2.INTER_AREA)


def calibration_matrices(renders, fidx, frames, M_c2o_lst, landmarker):
    """Similarity per template frame that carries the live head's render onto the frame's real face.

    renders: uint8 [512, 512, 3] live-head renders on silence; fidx: the template frame each was made for; frames / M_c2o_lst: the
    template. Returns (list of 3x3 float64, a dict of statistics); frames for which no face was found get the nearest good fit.
    """
    n = len(frames)
    found = {}
    before, after = [], []
    for img, i in zip(renders, fidx):
        if i in found:
            continue
        a = anchors(landmarker, img)
        b = anchors(landmarker, real_crop(frames[i], M_c2o_lst[i]))
        if a is None or b is None:
            continue
        M, rms = fit_similarity(a, b)
        p = _params(M)
        if abs(p[0]) > MAX_SCALE_STEP or abs(np.degrees(p[1])) > MAX_ROLL_DEG or np.abs(p[2:]).max() > MAX_SHIFT_PX:
            continue
        found[i] = p
        before.append(float(np.sqrt(((a - b) ** 2).sum(1).mean())))
        after.append(rms)
    if len(found) < max(3, n // 2):  # a placement interpolated over most of the loop would be a guess
        raise CalibrationError("a usable fit on only %d of %d template frames" % (len(found), n))
    idx = np.array(sorted(found))
    table = np.stack([found[i] for i in idx])
    full = np.stack([np.interp(np.arange(n), idx, table[:, k]) for k in range(4)], axis=1)  # frames without a fit: the nearest ones
    full = _median3(full)
    mats = [_matrix(p) for p in full]
    stats = {"frames": n, "fitted": len(idx), "scale_mean_%": float(np.expm1(full[:, 0].mean()) * 100),
             "scale_std_%": float(np.expm1(full[:, 0]).std() * 100), "roll_std_deg": float(np.degrees(full[:, 1]).std()),
             "shift_std_px": [float(full[:, 2].std()), float(full[:, 3].std())], "anchor_rms_before_px": float(np.mean(before)),
             "anchor_rms_after_px": float(np.mean(after))}
    return mats, stats


def calibrate_loop(loop, head, landmarker, **begin_kw):
    """Run the live head over `loop` on silence and return (M_c2o per frame with the calibration folded in, statistics).

    The render of template frame i goes into the paste transform M_i unchanged, so a render point r lands at M_i @ r; the real face
    sits at M_i @ (S_i @ r) after the fit S_i, hence M_i' = M_i @ S_i.
    """
    from .speech_pipeline import SpeechPipeline  # kept out of the module header: the pipeline needs a registered loop
    from .head_engine import HUBERT_STEP, SAMPLES_PER_FRAME

    n = len(loop)
    pipe = SpeechPipeline(head, keep_render=True)
    pipe.start(loop, **begin_kw)
    zeros = np.zeros(HUBERT_STEP, np.float32)
    for _ in range(0, (n + 5) * SAMPLES_PER_FRAME, HUBERT_STEP):
        pipe.push(zeros)
    pipe.finish()
    got = []
    while (f := pipe.frames.get()) is not None:
        got.append(f)
    if pipe.error:
        raise pipe.error
    pipe.join()
    got = sorted((f for f in got if f.idx < n), key=lambda f: f.idx)  # frames 0..n-1 run forward over the template: fidx == idx
    if len(got) < n:
        raise CalibrationError("the head engine rendered %d of %d frames of silence" % (len(got), n))
    mats, stats = calibration_matrices([f.render for f in got], [f.fidx for f in got], loop.frames, loop.info["M_c2o_lst"], landmarker)
    out = []
    for M_old, S in zip(loop.info["M_c2o_lst"], mats):
        M_old = np.asarray(M_old)
        M3 = np.eye(3)
        M3[:2] = M_old[:2].astype(np.float64)
        out.append((M3 @ S)[:len(M_old)].astype(M_old.dtype))  # same shape (2x3 or 3x3) and dtype as the matrices it replaces
    log.info("calibrated %s: %s", getattr(loop, "name", "?"), {k: (round(v, 2) if isinstance(v, float) else v) for k, v in stats.items()})
    return out, stats
