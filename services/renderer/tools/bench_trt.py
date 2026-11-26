"""Numerical check and micro-benchmark of the TensorRT engines against Ditto's PyTorch models (same GPU, same inputs).

Real motion is produced once with the PyTorch path; the same (source, driving) keypoints and LMDM conditioning are then
replayed through the engines, so every error figure is measured on in-distribution data.

  python tools/bench_trt.py --engine-dir /engines/sm90_trt10.11.0.33 --image /data/image.png --audio /data/audio.wav --out /work/bench_trt
"""
import argparse
import json
import os
import time

import cv2
import librosa
import numpy as np
import torch

from app.avatar import Loop
from app.compositor import PatchReader, blend_roi, paste
from app.head_engine import HUBERT_STEP, HeadEngine
from app.trt_models import LMDMTrt, TrtRenderer
from core.atomic_components.avatar_registrar import AvatarRegistrar

CROP_KEYS = ("crop_scale", "crop_vx_ratio", "crop_vy_ratio", "crop_flag_do_rot")


def stats_ms(ts):
    a = np.asarray(ts) * 1e3
    return {"mean_ms": round(float(a.mean()), 3), "p50_ms": round(float(np.percentile(a, 50)), 3),
            "p95_ms": round(float(np.percentile(a, 95)), 3)}


def timeit(fn, n=40, warm=8):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return stats_ms(ts)


class TorchNet:
    """Ditto's PyTorch LMDM (fp32) behind the engine call signature, as the reference denoiser."""

    def __init__(self, model):
        self.model = model

    @torch.inference_mode()
    def __call__(self, x, cond_frame, cond, time_cond):
        pred_noise, x_start = self.model(x, cond_frame, cond, time_cond)
        return {"pred_noise": pred_noise, "x_start": x_start}


class Recorder:
    """Stands in for Audio2Motion.lmdm and keeps the (kp_cond, aud_cond) pairs of a real run."""

    def __init__(self, inner):
        self.inner, self.calls = inner, []

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def __call__(self, kp_cond, aud_cond, steps):
        self.calls.append((np.array(kp_cond, copy=True), np.array(aud_cond, copy=True)))
        return self.inner(kp_cond, aud_cond, steps)


def to_img(t):
    return (t[0].clamp(0, 1).permute(1, 2, 0).float() * 255.0).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine-dir", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--audio", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cfg", default="/models/ditto-talkinghead/ditto_cfg/v0.4_hubert_cfg_pytorch.pkl")
    ap.add_argument("--data-root", default="/models/ditto-talkinghead/ditto_pytorch")
    ap.add_argument("--seconds", type=float, default=4.0)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--frames", type=int, default=24)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    report = {"engine_dir": args.engine_dir, "gpu": torch.cuda.get_device_name(0)}

    replace = {"default_kwargs": {"online_mode": True, "overlap_v2": 70, "sampling_timesteps": args.steps}}
    ref = HeadEngine(args.cfg, args.data_root, replace_cfg=replace)
    kw = ref.kw
    registrar = AvatarRegistrar(**ref.registrar_cfg)
    info = registrar(args.image, max_dim=kw.get("max_size", 1920), **{k: kw[k] for k in CROP_KEYS if k in kw})
    audio, _ = librosa.load(args.audio, sr=16000)
    audio = audio[: int(args.seconds * 16000)]

    # Real motion and LMDM conditioning from the PyTorch path.
    lmdm_pt = ref.audio2motion.lmdm
    rec = Recorder(lmdm_pt)
    ref.audio2motion.lmdm = rec
    ref.begin(info)
    motion = []
    for off in range(0, len(audio), HUBERT_STEP):
        motion += ref.feed_audio(audio[off:off + HUBERT_STEP], final=off + HUBERT_STEP >= len(audio))
    ref.audio2motion.lmdm = lmdm_pt
    print(f"{len(motion)} frames of motion, {len(rec.calls)} LMDM calls recorded", flush=True)

    # ---------------------------------------------------------------- warp + decode numerics
    trt = TrtRenderer(args.engine_dir)
    pairs = [ref.stitch(i, x_d) for i, x_d in motion[: args.frames]]
    f_s = info["f_s_lst"][0]
    f_s_gpu = torch.from_numpy(f_s).cuda().half()
    mae, p99, mx, psnr = [], [], [], []
    for k, (fidx, x_s, x_d) in enumerate(pairs):
        a = ref.decode_f3d(ref.warp_f3d(f_s, x_s, x_d))
        b = to_img(trt(f_s_gpu, x_s, x_d))
        d = np.abs(a - b)
        mae.append(d.mean())
        p99.append(np.percentile(d, 99))
        mx.append(d.max())
        psnr.append(20 * np.log10(255.0 / max(np.sqrt((d ** 2).mean()), 1e-6)))
        if k in (0, len(pairs) // 2, len(pairs) - 1):
            panel = np.concatenate([a, b, np.clip(d * 10, 0, 255)], 1).astype(np.uint8)
            cv2.imwrite(os.path.join(args.out, f"render_pt_trt_diff10x_{k:02d}.png"), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
    report["render_vs_pytorch_fp16"] = {
        "frames": len(pairs), "mean_abs_err_0_255": round(float(np.mean(mae)), 3),
        "p99_abs_err_mean": round(float(np.mean(p99)), 3), "max_abs_err": round(float(np.max(mx)), 2),
        "psnr_db_mean": round(float(np.mean(psnr)), 2),
    }
    print("render vs PyTorch:", report["render_vs_pytorch_fp16"], flush=True)

    # ---------------------------------------------------------------- LMDM numerics (same noise, same seed)
    samplers = {
        "torch_fp32": LMDMTrt(net=TorchNet(lmdm_pt.model)),
        "trt_fp32": LMDMTrt(os.path.join(args.engine_dir, "lmdm_v0.4_hubert_fp32.engine")),
        "trt_fp16": LMDMTrt(os.path.join(args.engine_dir, "lmdm_v0.4_hubert_fp16.engine")),
    }
    for s in samplers.values():
        s.setup(args.steps)
    for name in ("trt_fp32", "trt_fp16"):
        samplers[name].noise = samplers["torch_fp32"].noise
    lm = {"trt_fp32": [], "trt_fp16": []}
    for k, (kp_cond, aud_cond) in enumerate(rec.calls[:4]):
        outs = {}
        for name, s in samplers.items():
            torch.manual_seed(100 + k)
            outs[name] = s(kp_cond, aud_cond, args.steps)
        scale = float(np.sqrt((outs["torch_fp32"] ** 2).mean()))
        for name in lm:
            e = outs[name] - outs["torch_fp32"]
            lm[name].append({"rel_rmse": round(float(np.sqrt((e ** 2).mean())) / scale, 5), "max_abs": round(float(np.abs(e).max()), 4)})
    report["lmdm_vs_pytorch_fp32"] = lm
    print("lmdm vs PyTorch fp32:", lm, flush=True)

    # ---------------------------------------------------------------- timings
    fidx, x_s, x_d = pairs[len(pairs) // 2]
    kp = torch.from_numpy(np.concatenate([x_s, x_d], 0).astype(np.float32)).cuda()
    feat = trt.warp(feature_3d=f_s_gpu, kp_source=kp[0:1], kp_driving=kp[1:2])["out"].clone()
    t = {}
    t["warp_trt"] = timeit(lambda: trt.warp(feature_3d=f_s_gpu, kp_source=kp[0:1], kp_driving=kp[1:2]))
    t["decoder_trt"] = timeit(lambda: trt.decoder(feature=feat))
    t["warp+decoder_trt"] = timeit(lambda: trt(f_s_gpu, x_s, x_d))
    t["warp_pytorch_fp16"] = timeit(lambda: ref.warp_f3d(f_s, x_s, x_d), n=15, warm=3)
    f3d = ref.warp_f3d(f_s, x_s, x_d)
    t["decoder_pytorch_fp16"] = timeit(lambda: ref.decode_f3d(f3d), n=15, warm=3)
    xi = motion[len(motion) // 2]
    t["stitch_cpu"] = timeit(lambda: ref.stitch(xi[0], xi[1]), n=40, warm=3)
    kp_cond, aud_cond = rec.calls[-1]
    for name in ("trt_fp32", "trt_fp16"):
        s = samplers[name]
        t[f"lmdm_call_{args.steps}steps_{name}"] = timeit(lambda: s(kp_cond, aud_cond, args.steps), n=15, warm=3)
        x0 = torch.randn(1, 80, 265, device="cuda")
        cf = torch.from_numpy(np.ascontiguousarray(kp_cond, np.float32)).cuda()
        cd = torch.from_numpy(np.ascontiguousarray(aud_cond, np.float32)).cuda()
        t[f"lmdm_one_step_{name}"] = timeit(lambda: s.net(x=x0, cond_frame=cf, cond=cd, time_cond=s.time_cond[0]), n=30, warm=5)
    s = samplers["torch_fp32"]
    t[f"lmdm_call_{args.steps}steps_pytorch_fp32"] = timeit(lambda: s(kp_cond, aud_cond, args.steps), n=10, warm=2)

    # ---------------------------------------------------------------- GPU compositor + whole per-frame chain
    head = HeadEngine(args.cfg, args.data_root, replace_cfg=replace, engine_dir=args.engine_dir)
    t0 = time.perf_counter()
    loop = Loop.register("bench", args.image, registrar, head, head.kw)
    report["loop_register_s"] = round(time.perf_counter() - t0, 2)
    roi = loop.rois[0]
    report["roi_shape_hw"] = list(roi.shape)
    render = trt(f_s_gpu, x_s, x_d)
    reader = PatchReader()
    t["blend_roi_gpu"] = timeit(lambda: blend_roi(render, roi))
    patch = blend_roi(render, roi)
    t["roi_d2h"] = timeit(lambda: reader.read(patch))
    host = reader.read(patch)
    t["paste_cpu"] = timeit(lambda: paste(loop.frames[0], host, roi), n=60)

    def chain():
        return paste(loop.frames[0], reader.read(blend_roi(trt(f_s_gpu, x_s, x_d), roi)), roi)

    t["chain_render+blend+d2h+paste"] = timeit(chain, n=40)
    out = chain()
    cv2.imwrite(os.path.join(args.out, "composite_trt.png"), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(args.out, "frame0_original.png"), cv2.cvtColor(np.ascontiguousarray(loop.frames[0]), cv2.COLOR_RGB2BGR))

    # Round trip of the cache format.
    loop.save(os.path.join(args.out, "loop_cache"))
    again = Loop.load(os.path.join(args.out, "loop_cache"))
    same = np.array_equal(again.frames[0], loop.frames[0]) and again.rois[0].shape == loop.rois[0].shape
    report["loop_cache_roundtrip_ok"] = bool(same)

    report["timings"] = t
    with open(os.path.join(args.out, "bench_trt.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({"timings_ms": {k: v["mean_ms"] for k, v in t.items()}}, indent=2))


if __name__ == "__main__":
    main()
