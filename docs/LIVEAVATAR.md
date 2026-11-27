# LiveAvatar: one avatar across two H100s

An isolated renderer experiment, not a replacement for the Studio, gateway or idle clips. The benchmark binds to
`127.0.0.1:18160`. It renders **one full WAV for one portrait** and publishes independently playable HLS segments as they
become available. It is not incremental TTS input, WebRTC, barge-in, or an Internet-facing service.

Upstream [revision c3c47d0](https://github.com/Alibaba-Quark/LiveAvatar/tree/c3c47d031d8bf3247428333c1fe610579c71a551)
implements either offline single-GPU inference or distributed TPP with four denoising ranks and a fifth decoding rank.
This branch keeps four denoising steps and the upstream single-process sampling algorithm. GPU 0 runs all four FP8 DiT
steps. GPU 1 runs the Wan VAE, preserving temporal context and decoding one 48-frame clip while GPU 0 denoises the
next. The first clip drops three frames as in upstream. Only latents cross devices (via host memory for vGPU
compatibility); no cross-device NCCL or duplicate avatar workers. This is **not** two-rank TPP.

## Deploy or restart

Run from this branch's worktree with two reserved GPUs and a compatible LiveAvatar Python environment. On the test
host the existing weights and environment in the adjacent project can be reused without modifying them. Allow the
process to exit before reclaiming its devices. Do not start the old avatar stack on the same cards during measurements.

```bash
git clone https://github.com/Alibaba-Quark/LiveAvatar work/upstream
git -C work/upstream checkout --detach c3c47d031d8bf3247428333c1fe610579c71a551
git -C work/upstream apply "$PWD/services/liveavatar/two-device.patch"

export PATH=/home/kranthi/KK_Avatar/.venv-liveavatar/bin:$PATH
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export ENABLE_COMPILE=false PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=5
python -u services/liveavatar/bench.py --upstream work/upstream \
  --weights /home/kranthi/KK_Avatar/models/liveavatar \
  --image /home/kranthi/KK_Avatar_New/avatars/gulf-man-black/source.png \
  --audio /home/kranthi/KK_Avatar_New/services/renderer/tools/samples/reply_clean.wav --out work/runs
```

Use `nvidia-smi` to check the device order before starting; UUIDs can be used instead of indices. `ffmpeg` with libx264
and AAC must be on PATH. Benchmark environment: PyTorch 2.9.0+cu128, FP8 linear layers, eager execution, Torch SDPA
fallback for cross-attention. Inputs, weights, upstream checkout, recordings and logs stay outside Git. Audit the
specific checkpoints and dependencies before any commercial redistribution; upstream describes **most** of the
repository as Apache-2.0, not all artifacts.

In another shell:

```bash
curl -fsS http://127.0.0.1:18160/healthz
curl -fsS -X POST -H 'X-LiveAvatar-Benchmark: 1' http://127.0.0.1:18160/benchmark
# Use the returned run ID. index.m3u8 appears after the first segment:
ffplay http://127.0.0.1:18160/runs/RUN_ID/index.m3u8
curl -fsS http://127.0.0.1:18160/runs/RUN_ID/result.json
```

One job at a time; a second concurrent request returns 409. To use another WAV, relaunch with `--audio`. Use only trusted
local sources. Shut down the foreground process with Ctrl-C; if launched as a service stop that instance specifically.

## Measure and compare

`result.json` reports model-resident request time, **not** user-end-of-speech or browser latency. `renderer_first_chunk_s`
ends when the first encoded segment can be served; `last_chunk_s` ends when the final segment is available. `fps` is
presented frames / elapsed seconds, `after_first_chunk_fps` excludes startup, and `audio_rtf` is compute wall time /
speech duration (lower is better). The per-chunk `cpu_rgb_s` is measured after usable RGB reaches host memory, before
the media encoder. Report initialization separately. Publish both the first model-resident request and repeated runs
if making capacity claims.

On a separately reserved five-H100 host, prepare the same checkout/patch, weights, image and WAV. Use the upstream
five-GPU dependencies (including its supported attention kernel), then run from `work/upstream`:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4 WEIGHTS=/abs/weights IMAGE=/abs/source.png AUDIO=/abs/reply.wav \
    OUT=/abs/new-run bash ../../services/liveavatar/five.sh
```

`decoded.json` records only decoded CPU frame timings on the VAE rank; it does **not** pretend to measure HLS or browser
latency. Compare `cpu_rgb_s` at the same output frame counts on both systems (first 45 frames, then the full reply);
cap the five-GPU frame count at the reply length, since upstream can generate a padded tail. Match the portrait, WAV,
frame size, four steps, seed, FP8 and compilation mode. Record physical vs virtual devices, interconnect and kernels.
The two schedulers/VAEs differ, so even this is a comparison of deployment configurations, not isolated GPU-count scaling.
Upstream's [45 fps on multi-H800](https://github.com/Alibaba-Quark/LiveAvatar#-highlights) is a reference, not a five-H100 measurement.

### Two-GPU measurements, 2026-11-27

Gulf Man Black, the same 10.828 s English reply as the approved EchoMimic take, 384 x 704, 271 presentation frames at 25 fps.
Both H100L-94C devices reported unlicensed vGPU status and `SYS` interconnect (cross-NUMA, not NVLink). The results apply
to this host and this experimental scheduler. They are not a ceiling for bare-metal H100s. Initialization took 170.58 s
before the first request; it is excluded in the table.

| Request | First encoded segment | Final segment | Overall generated fps | Generated fps after first segment | Wall time / audio time |
|---|---:|---:|---:|---:|---:|
| First, model resident | 68.85 s | 217.00 s | 1.25 | 1.53 | 20.04x |
| Repeat, including cached prompt | 53.89 s | 201.89 s | 1.34 | 1.53 | 18.65x |

Peak allocated memory was 57.69 GiB on the DiT GPU and 12.28 GiB on the VAE GPU. Each full 48-frame segment covers
1.92 s of playback but successive segments took about 30 s. The HLS output works progressively; an interactive
viewer cannot play it continuously as it is generated. Five-H100 timings are **not measured**.
