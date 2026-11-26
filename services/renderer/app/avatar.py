"""A registered body loop: template frames, Ditto source info and GPU paste-back assets, with an on-disk cache.

Registration (landmarks, crop, appearance/motion extraction) is the expensive part of onboarding, so it runs once and is
cached; loading a cached loop only re-derives the cheap per-frame ROI tensors on the GPU.
"""
import json
import os
import pickle

import numpy as np
import torch

from core.atomic_components.avatar_registrar import smooth_x_s_info_lst

from .compositor import RESIDUAL_GRID, build_roi, crop_residual, elliptical_mask

_CROP_KEYS = ("crop_scale", "crop_vx_ratio", "crop_vy_ratio", "crop_flag_do_rot")
_INFO_KEYS = ("x_s_info_lst", "M_c2o_lst", "eye_open_lst", "eye_ball_lst", "sc", "is_image_flag")


class Loop:
    def __init__(self, name, frames, info, residual, mask_crop, device="cuda"):
        self.name = name
        self.frames = frames  # uint8 [N,H,W,3] (CPU, possibly memory-mapped)
        self.info = info      # Ditto source_info; f_s_gpu_lst holds one fp16 GPU tensor per frame
        self.residual = residual
        self.device = device
        self.rois = [
            build_roi(frames[i], info["M_c2o_lst"][i], residual[i] if residual is not None else None, mask_crop, device=device)
            for i in range(len(frames))
        ]

    def __len__(self):
        return len(self.frames)

    @property
    def size(self):
        return self.frames.shape[2], self.frames.shape[1]  # (width, height)

    # ------------------------------------------------------------------ build from a portrait image or a body video
    @classmethod
    def register(cls, name, source, registrar, head, kw, mask_crop=None, n_frames=-1, device="cuda", frame_filter=None):
        """source: image or video path. registrar: AvatarRegistrar. head: HeadEngine (renders the identity reconstruction).

        frame_filter: optional uint8 [N,H,W,3] -> same-shape function applied to the template frames after the face tracking and
        before the colour residual, so the live head is matched to the frames that are actually played.
        """
        mask_crop = elliptical_mask() if mask_crop is None else mask_crop
        crop_kw = {k: kw[k] for k in _CROP_KEYS if k in kw}
        info = registrar(source, max_dim=kw.get("max_size", 1920), n_frames=n_frames, **crop_kw)
        n = len(info["x_s_info_lst"])
        if n > 1 and kw.get("smo_k_s", 1) > 1:
            info["x_s_info_lst"] = smooth_x_s_info_lst(info["x_s_info_lst"], smo_k=kw["smo_k_s"])
        frames = np.stack(info.pop("img_rgb_lst"))
        if frame_filter is not None:
            frames = frame_filter(frames)
        residual = np.stack([
            crop_residual(frames[i], info["M_c2o_lst"][i], head.identity_render(info["f_s_lst"][i], info["x_s_info_lst"][i]))
            for i in range(n)
        ])
        info["f_s_gpu_lst"] = [torch.from_numpy(f).to(device).half() for f in info.pop("f_s_lst")]
        return cls(name, frames, info, residual, mask_crop, device)

    # ------------------------------------------------------------------ cache
    def save(self, directory):
        os.makedirs(directory, exist_ok=True)
        np.save(os.path.join(directory, "frames.npy"), np.ascontiguousarray(self.frames))
        np.save(os.path.join(directory, "f_s.npy"), np.stack([f.cpu().numpy() for f in self.info["f_s_gpu_lst"]]))
        np.save(os.path.join(directory, "residual.npy"), self.residual.astype(np.float16))
        with open(os.path.join(directory, "info.pkl"), "wb") as f:
            pickle.dump({k: self.info[k] for k in _INFO_KEYS}, f)
        with open(os.path.join(directory, "loop.json"), "w") as f:
            json.dump({"name": self.name, "frames": len(self), "residual_grid": RESIDUAL_GRID}, f)

    @classmethod
    def load(cls, directory, mask_crop=None, device="cuda"):
        mask_crop = elliptical_mask() if mask_crop is None else mask_crop
        with open(os.path.join(directory, "loop.json")) as f:
            meta = json.load(f)
        frames = np.load(os.path.join(directory, "frames.npy"), mmap_mode="r")
        with open(os.path.join(directory, "info.pkl"), "rb") as f:
            info = pickle.load(f)
        f_s = np.load(os.path.join(directory, "f_s.npy"), mmap_mode="r")
        info["f_s_gpu_lst"] = [torch.from_numpy(np.ascontiguousarray(f_s[i])).to(device) for i in range(len(f_s))]
        residual = np.load(os.path.join(directory, "residual.npy")).astype(np.float32)
        return cls(meta["name"], frames, info, residual, mask_crop, device)
