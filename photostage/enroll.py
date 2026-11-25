"""Locate the shared head crop inside the untouched photo.

Run with the FlashHead interpreter for its CPU face detector; it needs no GPU.
FlashHead and the head-only LiveTalk adapter render this crop at 512 px, softer than
the photo's own pixels whether the crop
covers more or fewer than 512 source pixels. So the photo's fine detail is always saved,
at the crop's own resolution, as a layer the player lays onto the moving head.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT / "external" / "FlashHead"))

FLASHHEAD_SIZE = 512
DETAIL_SIDE = 2048
# The player low-passes FlashHead's frame with a mip level of 2 ** 1.5 texels, a blur of
# sigma 2 ** 0.5 texels; the stored detail is the photo minus that same blur.
LOW_PASS_SIGMA = 2 ** 0.5


class CropRecorder:
    """Lets FlashHead's own crop routine report its box instead of pixels."""

    def crop(self, box):
        self.box = [int(value) for value in box]
        return self


def detail_layer(photo, crop):
    """The photo's fine detail inside the crop, stored as 128 + detail in an 8-bit gray image.

    It keeps the crop's own resolution (at most DETAIL_SIDE px), so it is never upsampled.
    """
    import cv2
    import numpy as np
    from PIL import Image

    luma = np.asarray(photo.crop(tuple(crop)).convert("L"), dtype=np.float32)
    if max(luma.shape) > DETAIL_SIDE:
        luma = cv2.resize(luma, None, fx=DETAIL_SIDE / max(luma.shape), fy=DETAIL_SIDE / max(luma.shape),
                          interpolation=cv2.INTER_AREA)
    height_px, width_px = luma.shape
    fine = luma - cv2.GaussianBlur(
        luma, (0, 0), sigmaX=LOW_PASS_SIGMA * width_px / FLASHHEAD_SIZE, sigmaY=LOW_PASS_SIGMA * height_px / FLASHHEAD_SIZE)
    # The full +-128 range keeps strong edges such as glasses rims.
    return Image.fromarray(np.clip(np.rint(128 + fine), 0, 255).astype(np.uint8))


def enroll(folder):
    import numpy as np
    from PIL import Image

    from flash_head.utils.cpu_face_handler import CPUFaceHandler
    from flash_head.utils.facecrop import get_scaled_bbox

    with Image.open(folder / "photo.png") as source:
        photo = source.convert("RGB")
    detector = CPUFaceHandler(min_detection_confidence=0.6)
    try:
        boxes, _ = detector(np.asarray(photo))
    finally:
        detector.detector.close()
    if len(boxes) != 1:
        raise ValueError("The portrait must contain exactly one clear, front-facing face.")
    width, height = photo.size
    x1, y1, x2, y2 = boxes[0]
    face = [x1 * width, y1 * height, x2 * width, y2 * height]
    crop = get_scaled_bbox(face, width, height, ratio=2.0, face_image=CropRecorder()).box
    crop_width, crop_height = crop[2] - crop[0], crop[3] - crop[1]
    if min(crop_width, crop_height) < 96:
        raise ValueError("The face needs more source pixels. Upload a sharper or larger photo.")

    detail_layer(photo, crop).save(folder / "detail.png")
    scale = max(crop_width, crop_height) / FLASHHEAD_SIZE
    rig = {"size": [width, height], "crop": crop, "face": [round(v, 1) for v in face], "detail": {"scale": round(scale, 3)}}
    (folder / "rig.json").write_text(json.dumps(rig))


if __name__ == "__main__":
    try:
        enroll(Path(sys.argv[1]))
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(2)
