"""MediaPipe FaceLandmarker (Apache-2.0) helpers that replace Ditto's InsightFace detector + 106-point landmarks.

Ditto's crop code (core/utils/crop.py) understands the iBUG 68-point layout, so the 478 MediaPipe points are
re-indexed into that layout. Only these 0-based 68-layout positions drive the face roll/centre:
36/39 and 42/45 (eye corners) and 48/54 (mouth corners); every other point only contributes to the crop extents.
"""
import numpy as np

# iBUG-68 position -> MediaPipe 478 index.
MP478_TO_68 = np.array(
    [
        # jaw 0-16 (ear to ear through the chin)
        234, 93, 132, 58, 172, 150, 176, 148, 152, 377, 400, 379, 365, 397, 288, 361, 454,
        # brows 17-21 (subject's right) and 22-26 (subject's left)
        70, 63, 105, 66, 107,
        336, 296, 334, 293, 300,
        # nose bridge 27-30, nose base 31-35
        168, 6, 197, 1,
        98, 97, 2, 326, 327,
        # eyes 36-41 (subject's right) and 42-47 (subject's left): outer, upper x2, inner, lower x2
        33, 160, 158, 133, 153, 144,
        362, 385, 387, 263, 373, 380,
        # outer lip 48-59, inner lip 60-67
        61, 40, 37, 0, 267, 270, 291, 321, 314, 17, 84, 91,
        78, 81, 13, 311, 308, 402, 14, 178,
    ],
    dtype=np.int32,
)
assert MP478_TO_68.shape == (68,)


class NoFaceDetectedError(RuntimeError):
    pass


def mp478_to_68(lmk478_norm, width, height):
    """[478, >=2] normalised (x/w, y/h) -> [68, 2] float32 pixel coordinates."""
    pts = np.asarray(lmk478_norm, dtype=np.float32)[MP478_TO_68, :2]
    return (pts * np.array([width, height], dtype=np.float32)).astype(np.float32)


def _face_area(lmk68):
    mn, mx = lmk68.min(0), lmk68.max(0)
    return float((mx[0] - mn[0]) * (mx[1] - mn[1]))


def detect_lmk68(landmarker, img_rgb, min_window=256):
    """Detect the largest face and return its 68-layout landmarks in full-image pixels, or None.

    MediaPipe's face detector is a 128x128 short-range model, so a small face in a full-body frame can be missed.
    When the whole frame finds nothing, square windows of shrinking size are slid over the frame and the
    detection with the largest face wins.
    """
    h, w = img_rgb.shape[:2]

    def run(window, x0, y0):
        res = landmarker(np.ascontiguousarray(window))  # [n, 478, 3] normalised, n == 0 when no face
        if res.ndim != 3 or res.shape[0] == 0:
            return None
        ww, wh = window.shape[1], window.shape[0]
        lmk = mp478_to_68(res[0], ww, wh)
        lmk[:, 0] += x0
        lmk[:, 1] += y0
        return lmk

    lmk = run(img_rgb, 0, 0)
    if lmk is not None:
        return lmk

    best = None
    side = min(h, w)
    for frac in (0.6, 0.4, 0.3, 0.2):
        win = int(max(min_window, side * frac))
        if win >= side:
            continue
        step = max(win // 2, 1)
        for y0 in range(0, h - win + 1, step):
            for x0 in range(0, w - win + 1, step):
                cand = run(img_rgb[y0 : y0 + win, x0 : x0 + win], x0, y0)
                if cand is not None and (best is None or _face_area(cand) > _face_area(best)):
                    best = cand
        if best is not None:
            return best
    return None
