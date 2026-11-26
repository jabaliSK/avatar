"""A registered body loop: template frames, Ditto source info and GPU paste-back assets, with an on-disk cache.

Registration (landmarks, crop, appearance/motion extraction) is the expensive part of onboarding, so it runs once and is
cached; loading a cached loop only re-derives the cheap per-frame ROI tensors on the GPU.
"""
import json
import logging
import os
import pickle

import cv2
import numpy as np
import torch

from core.atomic_components.avatar_registrar import smooth_x_s_info_lst

from .calibrate import CalibrationError, calibrate_loop
from .compositor import RESIDUAL_GRID, build_roi, crop_residual, elliptical_mask

log = logging.getLogger("kka.avatars")

_CROP_KEYS = ("crop_scale", "crop_vx_ratio", "crop_vy_ratio", "crop_flag_do_rot")
_INFO_KEYS = ("x_s_info_lst", "M_c2o_lst", "eye_open_lst", "eye_ball_lst", "sc", "is_image_flag", "aligned")
# 4: the head is placed on each template frame's real face (calibrate.py), M_c2o holds that placement; 3: the crop follows the eyes, not
# the jaw (SteadyCrop); 2: one appearance feature (the first frame's), raw expression, residual against the first frame's face
LOOP_FORMAT = 4


def motion_profile(frames, shrink=8):
    """How much the picture changes from each frame to the next (mean |difference| of a small grey copy), float32 [N].

    Entry i is the change between frames i and i+1; the last entry repeats the one before it. The head engine paces the body's
    walk back to its rest pose with it (HeadEngine.set_tail), so a fast stretch of the template is never played faster than it was
    generated."""
    if len(frames) < 2:
        return np.zeros(len(frames), np.float32)
    small = np.stack([
        cv2.resize(cv2.cvtColor(np.ascontiguousarray(f), cv2.COLOR_RGB2GRAY), None, fx=1.0 / shrink, fy=1.0 / shrink, interpolation=cv2.INTER_AREA)
        for f in frames
    ]).astype(np.float32)
    step = np.abs(np.diff(small, axis=0)).mean(axis=(1, 2))
    return np.append(step, step[-1]).astype(np.float32)


class Loop:
    def __init__(self, name, frames, info, residual, mask_crop, device="cuda"):
        self.name = name
        self.frames = frames  # uint8 [N,H,W,3] (CPU, possibly memory-mapped)
        self.info = info      # Ditto source_info; f_s_gpu_lst holds the first frame's appearance feature (fp16, GPU), body_motion the motion_profile()
        self.info.setdefault("aligned", False)  # True: M_c2o places the live head on each frame's real face (align()); caches older than format 4 lack it
        self.info["body_motion"] = motion_profile(frames)
        self.residual = residual
        self.mask_crop = mask_crop
        self.device = device
        self._build_rois()

    def _build_rois(self):
        self.rois = [
            build_roi(self.frames[i], self.info["M_c2o_lst"][i], self.residual[i] if self.residual is not None else None, self.mask_crop,
                      device=self.device)
            for i in range(len(self.frames))
        ]

    def __len__(self):
        return len(self.frames)

    @property
    def size(self):
        return self.frames.shape[2], self.frames.shape[1]  # (width, height)

    # ------------------------------------------------------------------ place the live head on the real face
    def match_colour(self, head):
        """The colour residual: what is left of each frame once the first frame's face has been moved to that frame's pose and
        expression (the template's own mouth included, so the residual carries no mouth). Rebuilds the paste assets.

        The reference face has the geometry the live head is drawn with (HeadEngine.begin): the rest face's canonical keypoints, and
        once the head is placed on the real face its scale too, so a few percent of the estimated scale do not leak into the residual
        along the edge of the face."""
        f_s = self.info["f_s_gpu_lst"][0].float().cpu().numpy()
        rest = self.info["x_s_info_lst"][0]
        placed = self.info["aligned"]

        def drawn_like_live(info):
            info = {**info, "kp": rest["kp"]}
            if placed:
                info["scale"] = rest["scale"]
            return info

        self.residual = np.stack([
            crop_residual(self.frames[i], self.info["M_c2o_lst"][i], head.reconstruct(f_s, rest, drawn_like_live(self.info["x_s_info_lst"][i])))
            for i in range(len(self))
        ])
        self._build_rois()

    def align(self, head, landmarker, **begin_kw):
        """Place the live head on each template frame's real face (calibrate.py), then match its colour.

        A single image has no template motion to follow and only gets the colour match. `begin_kw` are the HeadEngine.begin()
        arguments of the silent render the placement is measured on. Returns the calibration statistics (None for an image, or when
        the placement could not be measured: the loop then plays as a format-3 loop, and says so in the log)."""
        stats = None
        self.info["aligned"] = False
        if len(self) > 1 and not self.info["is_image_flag"]:
            placement = self.info["M_c2o_lst"]
            self.info["aligned"] = True  # the silent render the placement is measured on is the one this loop plays
            try:
                self.info["M_c2o_lst"], stats = calibrate_loop(self, head, landmarker, **begin_kw)
            except CalibrationError as e:
                log.warning("%s: head not placed on the template's real face (%s); it will play without that correction", self.name, e)
                self.info["M_c2o_lst"], self.info["aligned"] = placement, False
        self.match_colour(head)
        return stats

    # ------------------------------------------------------------------ build from a portrait image or a body video
    @classmethod
    def register(cls, name, source, registrar, head, kw, mask_crop=None, n_frames=-1, device="cuda", frame_filter=None):
        """source: image or video path. registrar: AvatarRegistrar. head: HeadEngine (renders the identity reconstruction).

        frame_filter: optional uint8 [N,H,W,3] -> same-shape function applied to the template frames after the face tracking and
        before the colour residual, so the live head is matched to the frames that are actually played.
        """
        mask_crop = elliptical_mask() if mask_crop is None else mask_crop
        crop_kw = {k: kw[k] for k in _CROP_KEYS if k in kw}
        info = registrar(source, max_dim=kw.get("max_size", 1920), n_frames=n_frames, steady_crop=True, **crop_kw)
        n = len(info["x_s_info_lst"])
        if n > 1 and kw.get("smo_k_s", 1) > 1:
            # The expression is not smoothed: its eye rows carry the template's own blinks (a 13-frame average flattens them), and the
            # colour residual needs each frame's real expression.
            info["x_s_info_lst"] = smooth_x_s_info_lst(info["x_s_info_lst"], ignore_keys=("exp",), smo_k=kw["smo_k_s"])
        frames = np.stack(info.pop("img_rgb_lst"))
        if frame_filter is not None:
            frames = frame_filter(frames)
        # The live head is always rendered from the first frame's appearance (HeadEngine.begin), so only that feature is kept. It is placed
        # on each frame's real face and its colour is matched to that frame (Loop.align).
        f_s = info.pop("f_s_lst")
        info["f_s_gpu_lst"] = [torch.from_numpy(f_s[0]).to(device).half()]
        loop = cls(name, frames, info, None, mask_crop, device)
        loop.align(head, registrar.source2info.landmark478)
        return loop

    # ------------------------------------------------------------------ cache
    def save(self, directory):
        os.makedirs(directory, exist_ok=True)
        np.save(os.path.join(directory, "frames.npy"), np.ascontiguousarray(self.frames))
        np.save(os.path.join(directory, "f_s.npy"), np.stack([f.cpu().numpy() for f in self.info["f_s_gpu_lst"]]))
        np.save(os.path.join(directory, "residual.npy"), self.residual.astype(np.float16))
        with open(os.path.join(directory, "info.pkl"), "wb") as f:
            pickle.dump({k: self.info[k] for k in _INFO_KEYS}, f)
        with open(os.path.join(directory, "loop.json"), "w") as f:
            json.dump({"name": self.name, "frames": len(self), "residual_grid": RESIDUAL_GRID, "format": LOOP_FORMAT}, f)

    @classmethod
    def load(cls, directory, mask_crop=None, device="cuda"):
        mask_crop = elliptical_mask() if mask_crop is None else mask_crop
        with open(os.path.join(directory, "loop.json")) as f:
            meta = json.load(f)
        fmt = meta.get("format", 1)
        if fmt < LOOP_FORMAT:
            log.warning("%s was registered by an older renderer (cache format %d, now %d): its head can %s. Register it again "
                        "(tools/onboard.py).", directory, fmt, LOOP_FORMAT,
                        "show the template's mouth" if fmt < 2 else
                        "swell when the template's mouth opens wide" if fmt < 3 else
                        "wobble: it is not placed on the template's real face frame by frame")
        frames = np.load(os.path.join(directory, "frames.npy"), mmap_mode="r")
        with open(os.path.join(directory, "info.pkl"), "rb") as f:
            info = pickle.load(f)
        f_s = np.load(os.path.join(directory, "f_s.npy"), mmap_mode="r")
        info["f_s_gpu_lst"] = [torch.from_numpy(np.ascontiguousarray(f_s[0])).to(device)]  # older caches hold one feature per frame
        residual = np.load(os.path.join(directory, "residual.npy")).astype(np.float32)
        return cls(meta["name"], frames, info, residual, mask_crop, device)
