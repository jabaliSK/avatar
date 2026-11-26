"""TensorRT-backed LMDM sampler and warp+decode renderer for the live path (GPU tensors end to end)."""
import os

import numpy as np
import torch

from core.models.lmdm import make_beta

from .trt_runtime import TRTModule


class LMDMTrt:
    """DDIM sampler for Ditto's LMDM denoiser engine. Mirrors core.models.lmdm.LMDM._call_np, but keeps the sampler state
    on the GPU so the only host copies are the conditioning in and the finished motion sequence out."""

    def __init__(self, engine_path=None, seq_frames=80, motion_feat_dim=265, device="cuda", net=None):
        # `net` lets a test inject another denoiser (same keyword call, dict of pred_noise/x_start) with identical sampler maths.
        self.net = net if net is not None else TRTModule(engine_path, device=device)
        self.seq_frames, self.motion_feat_dim, self.device = seq_frames, motion_feat_dim, device
        betas = torch.Tensor(make_beta(n_timestep=1000))
        self.alphas_cumprod = torch.cumprod(1.0 - betas, dim=0).cpu().numpy()
        self.steps = None

    def setup(self, sampling_timesteps):
        if self.steps == sampling_timesteps:
            return
        self.steps = sampling_timesteps
        times = list(reversed(torch.linspace(-1, 999, steps=sampling_timesteps + 1).int().tolist()))
        self.pairs = list(zip(times[:-1], times[1:]))
        self.time_cond = [torch.full((1,), t, dtype=torch.int64, device=self.device) for t, _ in self.pairs]
        self.coeffs = []
        for t, t_next in self.pairs:
            if t_next < 0:
                continue
            a, a_next = self.alphas_cumprod[t], self.alphas_cumprod[t_next]
            sigma = np.sqrt((1 - a / a_next) * (1 - a_next) / (1 - a))
            c = np.sqrt(1 - a_next - sigma ** 2)
            self.coeffs.append((float(np.sqrt(a_next)), float(c), float(sigma)))
        # Upstream draws one noise tensor per step when the sampler is configured and reuses it for every call.
        self.noise = torch.randn(len(self.coeffs), 1, self.seq_frames, self.motion_feat_dim, device=self.device)

    @torch.inference_mode()
    def __call__(self, kp_cond, aud_cond, sampling_timesteps):
        self.setup(sampling_timesteps)
        cond_frame = torch.from_numpy(np.ascontiguousarray(kp_cond, np.float32)).to(self.device)
        cond = torch.from_numpy(np.ascontiguousarray(aud_cond, np.float32)).to(self.device)
        x = torch.randn(1, self.seq_frames, self.motion_feat_dim, device=self.device)
        i = 0
        for k, (_, t_next) in enumerate(self.pairs):
            out = self.net(x=x, cond_frame=cond_frame, cond=cond, time_cond=self.time_cond[k])
            if t_next < 0:
                x = out["x_start"].clone()
                break
            a_sqrt, c, sigma = self.coeffs[i]
            x = out["x_start"] * a_sqrt + out["pred_noise"] * c + self.noise[i] * sigma
            i += 1
        return x.cpu().numpy()


class TrtRenderer:
    """feature_3d + source/driving keypoints -> 512x512 head render, as a GPU tensor [1,3,512,512] (nominally 0..1)."""

    def __init__(self, engine_dir, device="cuda"):
        self.device = device
        self.warp = TRTModule(os.path.join(engine_dir, "warp_network_fp16.engine"), device=device)
        self.decoder = TRTModule(os.path.join(engine_dir, "decoder_fp16.engine"), device=device)

    @torch.inference_mode()
    def __call__(self, f_s, x_s, x_d):
        """f_s: GPU tensor [1,32,16,64,64]; x_s, x_d: numpy [1,21,3]."""
        kp = torch.from_numpy(np.concatenate([x_s, x_d], 0).astype(np.float32)).to(self.device)
        feat = self.warp(feature_3d=f_s, kp_source=kp[0:1], kp_driving=kp[1:2])["out"]
        return self.decoder(feature=feat)["output"]
