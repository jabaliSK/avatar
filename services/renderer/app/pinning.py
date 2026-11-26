"""Keep a generated body clip pixel-identical to the photo wherever nothing moves."""
import cv2
import numpy as np


def pin_static(frames, photo, lo=6.0, hi=16.0, sigma=2.0, window=5):
    """Take the photo's own pixels wherever a generated clip stayed still.

    The video model re-draws every pixel: a faint colour cast, softer detail and a shimmer of about one grey level over a static
    background. Played next to the photo (the quiet states are the photo plus the live head) that shows as a change of tone and
    texture at each cross-fade, and the parts of the body that should stand still crawl while speaking.

    A pixel that stays within `lo` grey levels of frame 0 (largest colour channel, blurred by `sigma`) during `window` frames
    around the current one becomes the photo's pixel; between `lo` and `hi` the two are blended; above `hi` the generated pixel
    is kept, with the generator's global colour offset removed.

    frames: uint8 [N,H,W,3]; photo: uint8 [H,W,3] of the same size. Returns uint8 [N,H,W,3].
    """
    if window % 2 == 0:
        raise ValueError("window must be odd")
    if photo.shape != frames.shape[1:]:
        raise ValueError(f"photo {photo.shape} does not match the clip frames {frames.shape[1:]}")
    n, h, w = frames.shape[:3]
    photo = photo.astype(np.float32)
    ref = frames[0].astype(np.float32)
    cast = np.median((photo - ref).reshape(-1, 3), axis=0)

    still = np.empty((n, h, w), np.float32)
    for i in range(n):
        motion = np.abs(frames[i].astype(np.float32) - ref).max(axis=2)
        still[i] = np.clip((hi - cv2.GaussianBlur(motion, (0, 0), sigma)) / (hi - lo), 0.0, 1.0)
    pad = window // 2
    padded = np.concatenate([still[:1].repeat(pad, axis=0), still, still[-1:].repeat(pad, axis=0)])
    weight = padded[:n].copy()
    for k in range(1, window):  # a pixel is only pinned if it was still for the whole window: no flicker at the thresholds
        np.minimum(weight, padded[k:k + n], out=weight)

    out = np.empty_like(frames)
    for i in range(n):
        t = weight[i][..., None]
        t = t * t * (3.0 - 2.0 * t)
        out[i] = np.clip(photo * t + (frames[i].astype(np.float32) + cast) * (1.0 - t) + 0.5, 0, 255).astype(np.uint8)
    return out
