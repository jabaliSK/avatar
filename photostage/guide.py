"""Per-frame guides that let the player lay the photo's own fine detail onto the rendered head.

The head renderer outputs 512 px, often below the photo's resolution. For each rendered
frame this module measures how the face moved relative to the neutral reference frame (dense
optical flow) and how much of the frame that motion explains (a warp-match confidence). The
result is packed into a 128x128 8-bit RGB image that travels beside the frame; the player's
shader uses it to fetch the photo's detail from where each pixel came from, and falls back to
the renderer's own pixels where the warp does not explain the frame (open mouth, blinks).

    R, G  flow in 512-space pixels, stored as 128 + 4 * flow  (+-32 px, 0.25 px steps)
    B     confidence that the warped photo detail is valid, 0..255
"""

import base64
import io
import json
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from PIL import Image

# Frames are processed one per pool thread, which scales better than OpenCV's own threading.
cv2.setNumThreads(1)

SIZE = 512
GRID = 128
WORKERS = 8
IDLE_FPS = 25
JPEG_QUALITY = 90
# Correct the renderer's tone against the photo (photo minus neutral reference, 64x64,
# stored as 128 + 8 * level).
BIAS_GRID = 64
BIAS_BLUR = 16.0
BIAS_LEVELS = 8.0
FLOW_LEVELS = 4.0
FLOW_BLUR = 2.0
MATCH_BLUR = 1.5
# Mean residual, in gray levels, over which the warped photo stops explaining the frame.
MATCH_FULL, MATCH_NONE = 6.0, 16.0


class Reference:
    """The neutral head image that every frame's motion is measured against."""

    def __init__(self, image):
        if image.shape != (SIZE, SIZE, 3):
            raise ValueError("The reference frame must be 512x512.")
        self.image = image
        self.gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        x, y = np.meshgrid(np.arange(SIZE, dtype=np.float32), np.arange(SIZE, dtype=np.float32))
        self.grid = np.dstack([x, y])

    def guide(self, frame):
        """Return the packed (GRID, GRID, 3) RGB uint8 guide for one BGR 512x512 frame."""
        # A fresh estimator per call: OpenCV's DIS object is not thread-safe.
        dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        # frame(x) ~ reference(x + flow(x))
        flow = cv2.GaussianBlur(dis.calc(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), self.gray, None), (0, 0), FLOW_BLUR)
        target = self.grid + flow
        warped = cv2.remap(self.image, target[..., 0], target[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        residual = np.abs(self._soften(frame) - self._soften(warped)).mean(axis=2)
        t = np.clip((residual - MATCH_FULL) / (MATCH_NONE - MATCH_FULL), 0.0, 1.0)
        trust = cv2.GaussianBlur(1.0 - t * t * (3.0 - 2.0 * t), (0, 0), FLOW_BLUR)
        small = cv2.resize(np.dstack([flow, trust]), (GRID, GRID), interpolation=cv2.INTER_AREA)
        guide = np.empty((GRID, GRID, 3), np.uint8)
        guide[..., :2] = np.clip(np.rint(128.0 + FLOW_LEVELS * small[..., :2]), 0, 255)
        guide[..., 2] = np.clip(np.rint(255.0 * small[..., 2]), 0, 255)
        return guide

    @staticmethod
    def _soften(image):
        return cv2.GaussianBlur(image.astype(np.float32), (0, 0), MATCH_BLUR)


def encode_png(guide):
    buffer = io.BytesIO()
    Image.fromarray(guide).save(buffer, format="PNG", compress_level=1)
    return buffer.getvalue()


def decode_frame(jpeg):
    frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    if frame is None or frame.shape != (SIZE, SIZE, 3):
        raise ValueError("The renderer produced an unexpected frame.")
    return frame


_pool = ThreadPoolExecutor(WORKERS, thread_name_prefix="guide")


def guide_segment(reference, frames):
    """Base64 JPEG frames in, base64 PNG guides out, in the same order."""
    def one(encoded):
        guide = reference.guide(decode_frame(base64.b64decode(encoded)))
        return base64.b64encode(encode_png(guide)).decode("ascii")

    return list(_pool.map(one, frames))


def write_bias(path, photo_path, crop, reference):
    with Image.open(photo_path) as photo:
        head = np.asarray(photo.convert("RGB").crop(tuple(crop)).resize((SIZE, SIZE), Image.LANCZOS), np.float32)
    soften = lambda image: cv2.GaussianBlur(image.astype(np.float32), (0, 0), BIAS_BLUR)
    small = cv2.resize(soften(head) - soften(reference), (BIAS_GRID, BIAS_GRID), interpolation=cv2.INTER_AREA)
    Image.fromarray(np.clip(np.rint(128.0 + BIAS_LEVELS * small), 0, 255).astype(np.uint8)).save(path)


def build_idle(folder, frames, fps=IDLE_FPS):
    """Write ref.png, bias.png and the player's idle pack (idle.json + idle.bin) from the idle clip.

    idle.bin holds each frame's JPEG followed, when the avatar uses detail transfer, by its guide PNG.
    """
    rig = json.loads((folder / "rig.json").read_text())
    reference = np.median(np.stack(frames), axis=0).astype(np.uint8)
    Image.fromarray(reference).save(folder / "ref.png")
    write_bias(folder / "bias.png", folder / "photo.png", rig["crop"], reference)
    guided = rig["detail"] is not None
    anchor = Reference(np.ascontiguousarray(reference[..., ::-1]))

    def encode(rgb):
        buffer = io.BytesIO()
        Image.fromarray(rgb).save(buffer, format="JPEG", quality=JPEG_QUALITY)
        jpeg = buffer.getvalue()
        # Guides are measured on the decoded JPEG, which is what the player will display.
        return jpeg, encode_png(anchor.guide(decode_frame(jpeg))) if guided else None

    payload, index = bytearray(), []
    for jpeg, png in _pool.map(encode, frames):
        entry = [len(payload), len(jpeg)]
        payload += jpeg
        if png is not None:
            entry += [len(payload), len(png)]
            payload += png
        index.append(entry)
    (folder / "idle.bin").write_bytes(payload)
    (folder / "idle.json").write_text(json.dumps({"fps": fps, "guided": guided, "frames": index}))
