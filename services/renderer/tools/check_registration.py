"""Registration sanity check for the MediaPipe-only Ditto build.

Registers a portrait, writes debug images (68-point overlay, the 512 crop, identity self-reconstruction) and prints stage
timings. The self-reconstruction (warp with x_d == x_s, then decode) exercises the appearance extractor, warp network and
decoder weights end to end: it must look like the crop, or something is wrong with the checkpoints/config.

  python tools/check_registration.py --image /data/portrait.png --out /work/reg_check
"""
import argparse
import json
import os
import time

import cv2
import numpy as np
import torch

from core.atomic_components.avatar_registrar import AvatarRegistrar
from core.atomic_components.cfg import parse_cfg
from core.atomic_components.decode_f3d import DecodeF3D
from core.atomic_components.motion_stitch import transform_keypoint
from core.atomic_components.warp_f3d import WarpF3D
from core.utils.mp_landmarks import detect_lmk68


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cfg", default="/models/ditto-talkinghead/ditto_cfg/v0.4_hubert_cfg_pytorch.pkl")
    ap.add_argument("--data-root", default="/models/ditto-talkinghead/ditto_pytorch")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    reg_cfg, _ch, _lmdm, _stitch, warp_cfg, dec_cfg, _w2f, kw = parse_cfg(args.cfg, args.data_root)
    t0 = time.perf_counter()
    registrar = AvatarRegistrar(**reg_cfg)
    warp, dec = WarpF3D(warp_cfg), DecodeF3D(dec_cfg)
    print(f"models loaded in {time.perf_counter() - t0:.1f}s")

    crop_kw = {k: kw[k] for k in ("crop_scale", "crop_vx_ratio", "crop_vy_ratio", "crop_flag_do_rot") if k in kw}
    for _ in range(2):  # second pass is the warm timing
        t0 = time.perf_counter()
        info = registrar(args.image, max_dim=kw.get("max_size", 1920), **crop_kw)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
    print(f"register(): {dt * 1e3:.0f} ms for {len(info['x_s_info_lst'])} frame(s)")

    rgb = info["img_rgb_lst"][0]
    h, w = rgb.shape[:2]
    lmk68 = detect_lmk68(registrar.source2info.landmark478, rgb)
    vis = rgb.copy()
    for x, y in lmk68:
        cv2.circle(vis, (int(round(x)), int(round(y))), max(1, min(h, w) // 300), (0, 255, 0), -1)
    # project the 512 crop square back onto the source frame
    M = info["M_c2o_lst"][0]
    sq = np.array([[0, 0, 1], [511, 0, 1], [511, 511, 1], [0, 511, 1]], np.float32) @ M[:2].T
    cv2.polylines(vis, [sq.astype(np.int32).reshape(-1, 1, 2)], True, (255, 0, 0), max(2, min(h, w) // 250))
    cv2.imwrite(os.path.join(args.out, "landmarks_crop_box.png"), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

    x_s_info = info["x_s_info_lst"][0]
    x_s = transform_keypoint(x_s_info)
    f_s = info["f_s_lst"][0]
    recon = None
    for _ in range(2):
        t0 = time.perf_counter()
        recon = dec(warp(f_s, x_s, x_s))
        torch.cuda.synchronize()
        dt_r = time.perf_counter() - t0
    print(f"warp+decode (identity): {dt_r * 1e3:.0f} ms")

    inv = cv2.invertAffineTransform(M[:2])
    crop = cv2.warpAffine(rgb, inv, (512, 512), flags=cv2.INTER_AREA)
    recon_u8 = np.clip(recon, 0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(args.out, "crop512_vs_recon.png"),
                cv2.cvtColor(np.concatenate([crop, recon_u8], 1), cv2.COLOR_RGB2BGR))

    err = np.abs(crop.astype(np.float32) - recon.astype(np.float32))
    summary = {
        "image": args.image,
        "size_wh": [w, h],
        "face_box_wh": [float(np.ptp(lmk68[:, 0])), float(np.ptp(lmk68[:, 1]))],
        "recon_mean_abs_err_0_255": float(err.mean()),
        "register_ms": round(dt * 1e3, 1),
        "warp_decode_ms": round(dt_r * 1e3, 1),
        "eye_open": np.asarray(info["eye_open_lst"][0]).tolist(),
    }
    print(json.dumps(summary, indent=2))
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
