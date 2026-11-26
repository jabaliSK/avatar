# Renderer: what differs from upstream

The live head renderer is built on **Ditto** (`antgroup/ditto-talkinghead`, Apache-2.0), pinned to commit `c3e47eee2e62`
(see `scripts/fetch_sources.sh`). `services/renderer/ditto/` is that source tree with the patches in section 1.
`services/renderer/app/` is new code written for this project (section 2).

Audit the vendored tree against upstream at any time (`third_party/` is filled by `scripts/fetch_sources.sh`):

```bash
diff -ruN -x __pycache__ -x '*.pyc' -x .git third_party/ditto-talkinghead services/renderer/ditto
```

## 1. Vendored Ditto (`services/renderer/ditto/`)

The only functional change is that **InsightFace is gone** and MediaPipe does its job. InsightFace's detector and 106-point
landmarker are restricted to non-commercial research use, which is incompatible with the Apache-2.0 / MIT-only requirement.

| File | Change |
|---|---|
| `core/aux_models/insightface_det.py`, `insightface_landmark106.py`, `modules/retinaface.py`, `modules/landmark106.py` | Removed. |
| `core/aux_models/modules/__init__.py` | No longer imports `RetinaFace` / `Landmark106`. |
| `core/atomic_components/source2info.py` | First-frame face detection uses `detect_lmk68()` on MediaPipe FaceLandmarker (Apache-2.0) instead of InsightFace detection + 106 landmarks. Raises `NoFaceDetectedError` instead of returning `None`. |
| `core/atomic_components/avatar_registrar.py`, `stream_pipeline_online.py` (docstring only) | The two InsightFace configs are no longer constructor arguments. |
| `core/atomic_components/cfg.py` | Drops `insightface_det_cfg` and `landmark106_cfg` from the configuration pickles, which still carry them. The InsightFace model files are never downloaded either (`scripts/fetch_models.py` deny-list). |
| `core/utils/mp_landmarks.py` (new) | Re-indexes MediaPipe's 478 points into the iBUG-68 layout that Ditto's `crop.py` expects. `detect_lmk68()` retries on sliding windows when the face is small in a full-body frame (MediaPipe's detector is a 128x128 short-range model). |
| `core/utils/blend/__init__.py` | Uses a pre-built blend extension when present and falls back to `pyximport` (needs a C compiler). |
| `setup_blend.py` (new) | Pre-builds that extension without `-march=native`, so the binary does not depend on the build machine's CPU. |
| `environment.yaml`, `README.md`, `.gitignore` | Not vendored. |

Untouched: the models' Python wrappers, `crop.py`, `eye_info.py`, the LMDM / stitch / warp / decode code, HuBERT streaming and
`inference.py` / `stream_pipeline_offline.py` (kept for reference; the live path does not use them).

### Model files

`scripts/fetch_models.py` fetches only what the live path needs from `digital-avatar/ditto-talkinghead`: the configuration
pickles, the PyTorch weights, the ONNX files (the TensorRT engines are built from them) and the three auxiliary models
`landmark203.onnx`, `face_landmarker.task` and `hubert_streaming_fix_kv.onnx`. The Hugging Face bundle also contains
InsightFace-derived files; those are on the deny-list and are not on disk.

Two points for the HPE licence review:

* The Hugging Face model card carries no licence tag (the GitHub repository's `LICENSE` is Apache-2.0).
* `landmark203.onnx` is byte-identical to the file LivePortrait ships (MIT), and the provenance of the streaming HuBERT ONNX is
  not documented by Ditto.

## 2. New code in `services/renderer/app/`

| Module | Replaces / adds |
|---|---|
| `head_engine.py` | Ditto's `StreamSDK` is a one-shot, file-writing pipeline with six worker threads. `HeadEngine` runs the same atomic components (HuBERT -> LMDM -> stitch -> warp -> decode) chunk by chunk, restartable for every reply and without a video writer. It also adds the silent-tail walk-back (`set_tail`, `template_index`, `TAIL_SPEED`). |
| `trt_runtime.py`, `trt_models.py` | TensorRT 10 runner that binds torch device pointers directly (Ditto's `TRTWrapper` round-trips every tensor through host memory), the GPU-resident DDIM sampler for the LMDM engine and the warp + decode renderer. |
| `engine_cache.py`, `tools/build_engines.py` | Engines are built for the GPU in use and cached in `engines/sm<CC>_trt<VERSION>/`. The prebuilt engines and plugin from the Ditto bundle target TensorRT 8.6 and do not load on this host, so none of them is used. |
| `compositor.py` | Replaces Ditto's CPU `PutBack` (it rewrites the whole frame under a soft rectangle) with a GPU paste-back limited to the head's bounding box, a face-aligned elliptical mask and a per-frame colour residual. |
| `avatar.py`, `avatar_store.py`, `pinning.py` | Avatar onboarding and caching. `pin_static` copies the photo's pixels into every Stage A clip wherever nothing moves, removing the generator's colour cast and background shimmer. |
| `player.py`, `speech_pipeline.py`, `server.py` | The 25 fps player (state machine, cross-fades, barge-in) and the WebSocket API. |

The **GridSample3D** TensorRT plugin comes from `SeanWangJS/grid-sample3d-trt-plugin` (Apache-2.0, pinned in
`scripts/fetch_sources.sh`). `docker/Dockerfile.renderer` compiles it against TensorRT 10.11 with SASS for sm_89 (L40S) and
sm_90 (H100 / H200); another GPU generation needs `--build-arg PLUGIN_ARCHS="80 89 90"` (list every architecture it has to run
on). The prebuilt plugin shipped with Ditto is not used: it fails with "unsupported toolchain" on a 550-series driver.
