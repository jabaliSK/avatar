"""GPU paste-back of a rendered head onto a body-loop frame, with a per-frame colour residual.

Ditto's own PutBack warps the whole 512x512 crop onto the full frame on the CPU and replaces everything under a soft
rectangle. For a waist-up or full-body frame that rewrites hair, background and shoulders with a slightly different
reconstruction and leaves a visible box. Here:

* the paste is limited to the crop's bounding box in the frame (ROI), done on the GPU with one grid_sample;
* the mask is a face-aligned soft ellipse (the crop is rotated to the face, so one ellipse fits every frame);
* each template frame stores a low-frequency residual (original crop - identity reconstruction), so a head rendered with
  the neutral expression reproduces the original pixels and the seam disappears (lighting/colour match by construction);
* everything per-frame (grid, mask, background patch, residual) is precomputed when the loop is registered.
"""
from dataclasses import dataclass

import cv2
import numpy as np
import torch
import torch.nn.functional as F

CROP = 512


def elliptical_mask(cx=0.50, cy=0.56, rx=0.30, ry=0.43, core=0.78):
    """Soft ellipse in crop space (fractions of the crop). Opaque inside `core` x radius, smoothstep to 0 at radius 1."""
    ys, xs = np.mgrid[0:CROP, 0:CROP].astype(np.float32)
    r = np.sqrt(((xs + 0.5) / CROP - cx) ** 2 / rx ** 2 + ((ys + 0.5) / CROP - cy) ** 2 / ry ** 2)
    t = np.clip((r - core) / (1.0 - core), 0.0, 1.0)
    return (1.0 - t * t * (3.0 - 2.0 * t)).astype(np.float32)


@dataclass
class RoiAssets:
    x0: int
    y0: int
    x1: int
    y1: int
    grid: torch.Tensor   # [h, w, 2] fp32, normalised crop coordinates for grid_sample
    mask: torch.Tensor   # [1, h, w] fp16
    bg: torch.Tensor     # [3, h, w] fp16, original frame pixels
    delta: torch.Tensor  # [3, h, w] fp16, colour residual in 0..255 units

    @property
    def shape(self):
        return self.y1 - self.y0, self.x1 - self.x0


def _affine3(M_c2o):
    M = np.eye(3, dtype=np.float64)
    M[:2] = np.asarray(M_c2o, dtype=np.float64)[:2]
    return M


RESIDUAL_GRID = 24


def crop_residual(frame_rgb, M_c2o, identity_recon):
    """Low-frequency (original crop - identity reconstruction) as float32 [24, 24, 3] in 0..255 units.

    identity_recon: float32 [512,512,3] in 0..255, the render with x_d == x_s for this frame.
    """
    inv = cv2.invertAffineTransform(_affine3(M_c2o)[:2])
    crop = cv2.warpAffine(frame_rgb, inv, (CROP, CROP), flags=cv2.INTER_AREA).astype(np.float32)
    return cv2.resize(crop - identity_recon, (RESIDUAL_GRID, RESIDUAL_GRID), interpolation=cv2.INTER_AREA)


def build_roi(frame_rgb, M_c2o, residual, mask_crop, pad=3, device="cuda"):
    """frame_rgb: uint8 [H,W,3]; residual: [24,24,3] from crop_residual() or None for no colour correction."""
    H, W = frame_rgb.shape[:2]
    M = _affine3(M_c2o)
    corners = np.array([[-.5, -.5, 1], [CROP - .5, -.5, 1], [CROP - .5, CROP - .5, 1], [-.5, CROP - .5, 1]]).T
    p = (M @ corners)[:2]
    x0, x1 = int(max(np.floor(p[0].min()) - pad, 0)), int(min(np.ceil(p[0].max()) + pad + 1, W))
    y0, y1 = int(max(np.floor(p[1].min()) - pad, 0)), int(min(np.ceil(p[1].max()) + pad + 1, H))
    if x1 - x0 < 8 or y1 - y0 < 8:
        raise ValueError("face crop lies outside the frame")

    Minv = np.linalg.inv(M)
    ys, xs = np.mgrid[y0:y1, x0:x1].astype(np.float64)
    u = Minv[0, 0] * xs + Minv[0, 1] * ys + Minv[0, 2]
    v = Minv[1, 0] * xs + Minv[1, 1] * ys + Minv[1, 2]
    grid = torch.from_numpy(np.stack([(u + .5) / CROP * 2 - 1, (v + .5) / CROP * 2 - 1], -1).astype(np.float32)).to(device)

    def sample(chw):
        return F.grid_sample(chw[None], grid[None], mode="bilinear", padding_mode="zeros", align_corners=False)[0]

    mask = sample(torch.from_numpy(mask_crop)[None].to(device)).clamp_(0, 1)
    bg = torch.from_numpy(np.ascontiguousarray(frame_rgb[y0:y1, x0:x1].transpose(2, 0, 1))).to(device)

    delta = torch.zeros_like(bg, dtype=torch.float32)
    if residual is not None:
        resid = cv2.resize(np.asarray(residual, np.float32), (CROP, CROP), interpolation=cv2.INTER_LINEAR)
        delta = sample(torch.from_numpy(resid.transpose(2, 0, 1)).to(device))
    return RoiAssets(x0, y0, x1, y1, grid, mask.half(), bg.half(), delta.half())


@torch.inference_mode()
def blend_roi(render01, a: RoiAssets):
    """render01: [1,3,512,512] fp32 on the GPU, nominally 0..1. Returns the blended ROI as uint8 [h,w,3] on the GPU."""
    r = F.grid_sample(render01.clamp(0, 1), a.grid[None], mode="bilinear", padding_mode="zeros", align_corners=False)[0]
    m = a.mask.float()
    out = a.bg.float() * (1.0 - m) + (r * 255.0 + a.delta.float()) * m
    return out.round_().clamp_(0, 255).to(torch.uint8).permute(1, 2, 0).contiguous()


class PatchReader:
    """Device-to-host copy of the blended ROI through reusable pinned buffers."""

    def __init__(self):
        self._pinned = {}

    def read(self, patch_gpu):
        buf = self._pinned.get(tuple(patch_gpu.shape))
        if buf is None:
            buf = torch.empty(tuple(patch_gpu.shape), dtype=torch.uint8).pin_memory()
            self._pinned[tuple(patch_gpu.shape)] = buf
        buf.copy_(patch_gpu, non_blocking=True)
        torch.cuda.current_stream().synchronize()
        return buf.numpy()


def paste(frame_rgb, patch, a: RoiAssets):
    out = frame_rgb.copy()
    out[a.y0:a.y1, a.x0:a.x1] = patch
    return out
