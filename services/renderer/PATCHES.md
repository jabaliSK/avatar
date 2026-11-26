# Renderer: what differs from upstream

The live head renderer is built on **Ditto** (`antgroup/ditto-talkinghead`, Apache-2.0), pinned to commit `c3e47eee2e62`
(see `scripts/fetch_sources.sh`). `services/renderer/ditto/` is that source tree with the patches in section 1.
`services/renderer/app/` is new code written for this project (section 2).

Audit the vendored tree against upstream at any time (`third_party/` is filled by `scripts/fetch_sources.sh`):

```bash
diff -ruN -x __pycache__ -x '*.pyc' -x .git third_party/ditto-talkinghead services/renderer/ditto
```

## 1. Vendored Ditto (`services/renderer/ditto/`)

The functional changes are three. **InsightFace is gone** and MediaPipe does its job: InsightFace's detector and 106-point
landmarker are restricted to non-commercial research use, which is incompatible with the Apache-2.0 / MIT-only requirement.
The live head is **driven from a neutral face** (`motion_stitch.py`), so the talking mouth of a generated body clip never
shows through the audio-driven one. And the face crop of a body clip **follows the eyes instead of the whole face**
(`crop.py`), so a wide-open mouth in the clip does not make the rendered head swell.

| File | Change |
|---|---|
| `core/aux_models/insightface_det.py`, `insightface_landmark106.py`, `modules/retinaface.py`, `modules/landmark106.py` | Removed. |
| `core/aux_models/modules/__init__.py` | No longer imports `RetinaFace` / `Landmark106`. |
| `core/atomic_components/source2info.py` | First-frame face detection uses `detect_lmk68()` on MediaPipe FaceLandmarker (Apache-2.0) instead of InsightFace detection + 106 landmarks. Raises `NoFaceDetectedError` instead of returning `None`. `__call__(..., steady=)` and `_crop` can build and reuse a `SteadyCrop` (below); `None` keeps upstream behaviour. |
| `core/atomic_components/avatar_registrar.py`, `stream_pipeline_online.py` (docstring only) | The two InsightFace configs are no longer constructor arguments. `AvatarRegistrar.register(..., steady_crop=False)`: for a video, `True` crops every frame with a `SteadyCrop` calibrated on the first one. |
| `core/utils/crop.py` | `SteadyCrop` and its use in `_estimate_similar_transform_from_pts` / `crop_image` (`steady=`); `None` keeps upstream behaviour. |
| `core/atomic_components/cfg.py` | Drops `insightface_det_cfg` and `landmark106_cfg` from the configuration pickles, which still carry them. The InsightFace model files are never downloaded either (`scripts/fetch_models.py` deny-list). |
| `core/utils/mp_landmarks.py` (new) | Re-indexes MediaPipe's 478 points into the iBUG-68 layout that Ditto's `crop.py` expects. `detect_lmk68()` retries on sliding windows when the face is small in a full-body frame (MediaPipe's detector is a 128x128 short-range model). |
| `core/atomic_components/motion_stitch.py` | `MotionStitch.setup(..., rest_info=, rest_opts=)`: the neutral-face driving described below, and which parts of the template frame still drive the head (scale, shift, keypoints, expression rows). `None` keeps upstream behaviour. |
| `core/atomic_components/audio2motion.py` | Three Chinese code comments translated to English. No code change. |
| `core/utils/blend/__init__.py` | Uses a pre-built blend extension when present and falls back to `pyximport` (needs a C compiler). |
| `setup_blend.py` (new) | Pre-builds that extension without `-march=native`, so the binary does not depend on the build machine's CPU. |
| `environment.yaml`, `README.md`, `.gitignore` | Not vendored. |

Untouched: the models' Python wrappers, `eye_info.py`, the LMDM / warp / decode code, HuBERT streaming and
`inference.py` / `stream_pipeline_offline.py` (kept for reference; the live path does not use them).

### Neutral-face driving (`motion_stitch.py`)

Ditto adds the audio-driven expression offset to the expression of the *template frame*. A generated body clip is itself talking
(or breathing with a half-open mouth), so the mouth moved twice, moved with the clip's own rhythm instead of the audio's, and
kept moving after the last word: the face read as a layer sliding over a photo. With `rest_info` (the `x_s_info` of the
template's first frame, the rest pose) the head is built like this:

* the warp always starts from the **first frame's appearance feature** and canonical keypoints (a talking frame's own feature
  carries its open mouth, and the warp cannot close it again);
* the driving face takes the **pose and shift of the current template frame**, so the head follows the body, and the audio-driven
  lip offset (scaled by the avatar's `expression` setting) goes on top of it;
* `rest_opts` says what else comes from the template frame, and the rest face's own values are used for the rest.
  `HeadEngine.begin` picks one of two sets (`rest_options` in `head_engine.py`):

  | | scale | expression rows taken from the template frame |
  |---|---|---|
  | `REST_ALIGNED`: a loop whose head was placed on the template's real face (below; cache format 4) | the rest face's | all but the lips' (brows, eyes, cheeks, ...) |
  | `REST_LEGACY`: any other video loop (a cache of format 3, or one whose placement could not be measured) | the template frame's estimate | the eyes' only |

* the eye rows keep the template's own expression (so its blinks survive) or get Ditto's synthetic blinks when the template
  never blinks;
* the stitching network is given the rest face in the current pose as its source, so it only sees the lip difference and keeps the
  paste-back alignment.

The renderer therefore needs a template whose frame 0 is the rest pose, which is what the Stage A clips are (`idle` is pinned to
it, `speaking` starts on it). The rest pose of `gulf-man-black` is a smile with teeth, so its mouth never closes fully or puckers.

### Steady crop (`crop.py`)

Ditto sizes and centres the face crop on the bounding box of all 203 landmarks, jaw included. In a generated body clip the mouth
opens wide on some frames, the box grows, the crop zooms out and the head is pasted back 10 to 15 % larger for two to five frames:
a visible lunge on every wide-open vowel. `SteadyCrop` calibrates on the clip's first frame (a closed mouth). It keeps the ratio of
crop size to outer-eye-corner distance, the centre offset and the rotation offset measured there, and applies them to the eyes of
every later frame, so the crop follows the head and ignores the jaw. Frame 0 is reproduced exactly, and the eye distance is
clamped to half its first-frame value so one bad landmark frame cannot collapse the crop. Only `AvatarRegistrar.register(...,
steady_crop=True)` uses it, which is what `Loop.register` passes for a video; a still photo and the vendored default keep the
upstream crop.

Head-size pulse (outer-eye-corner distance against its own +-7 frame median; the noise floor is 0.6 to 1 %), p95 / max in percent
over a spoken reply, same audio, old crop against steady crop:

| Avatar | Speech, old crop | Speech, steady crop | After the last word (p95), old to steady |
|---|---|---|---|
| neri | 8.98 / 14.8 | 4.11 / 7.57 | 8.29 to 2.75 |
| seated-man | 9.48 / 13.7 | 6.28 / 10.46 | 8.51 to 1.94 |
| gulf-man-white-2 | 5.13 / 8.38 | 2.36 / 5.21 | 5.45 to 3.11 |
| gulf-man-black | 5.28 / 8.96 | 1.98 / 3.57 | 4.92 to 3.21 |
| gulf-man-black-closeup | 4.37 / 9.24 | 3.70 / 5.45 | 8.72 to 4.73 |
| gulf-man-white | not measured | 3.01 / 3.70 | 3.17 |

What is left is the clip's own head motion (a head turn changes the eye distance as well); neri's Stage A clip played back
unchanged measures 3.28 / 4.25. Jaw and lip-gap p95 stay within the run-to-run spread, and the surround of the face moves less
than before. That spread is real: the audio-to-motion model samples, so repeat runs of one build differ by about 0.02 in jaw p95
and 0.5 in pulse p95 (closeup 3.70, 2.78 and 3.19; white-2 2.36, 2.63 and 2.16).

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
| `head_engine.py` | Ditto's `StreamSDK` is a one-shot, file-writing pipeline with six worker threads. `HeadEngine` runs the same atomic components (HuBERT -> LMDM -> stitch -> warp -> decode) chunk by chunk, restartable for every reply and without a video writer. It also sets the neutral-face driving up (`begin`, `reconstruct`), picks synthetic blinks for templates that never blink (`has_blinks`) and adds the silent-tail walk-back: after the last word the body returns to its rest pose on a speed profile paced by the template's own motion (`set_tail`, `tail_caps`, `tail_path`). |
| `trt_runtime.py`, `trt_models.py` | TensorRT 10 runner that binds torch device pointers directly (Ditto's `TRTWrapper` round-trips every tensor through host memory), the GPU-resident DDIM sampler for the LMDM engine and the warp + decode renderer. |
| `engine_cache.py`, `tools/build_engines.py` | Engines are built for the GPU in use and cached in `engines/sm<CC>_trt<VERSION>/`. The prebuilt engines and plugin from the Ditto bundle target TensorRT 8.6 and do not load on this host, so none of them is used. |
| `compositor.py` | Replaces Ditto's CPU `PutBack` (it rewrites the whole frame under a soft rectangle) with a GPU paste-back limited to the head's bounding box, a face-aligned elliptical mask and a per-frame colour residual. |
| `calibrate.py` | Registration-time placement of the live head on every template frame's real face, which is what keeps the head from wobbling (below). |
| `avatar.py`, `avatar_store.py`, `pinning.py` | Avatar onboarding and caching. A registered loop (cache format 4) holds one neutral appearance feature, the template's raw expression, a small colour residual, the steady crop's transforms with the head placement folded in, and an `aligned` flag. A cache of an older format still loads and plays as before, but logs a warning (its head can wobble) and should be registered again (`tools/onboard.py`). `avatar_store` renders each quiet state once at load time (the live head on silence over a full cycle of the Stage A clip) and keeps the per-avatar `settings.json` (expression, head motion). `pin_static` copies the photo's pixels into every Stage A clip wherever nothing moves, removing the generator's colour cast and background shimmer. |
| `player.py`, `speech_pipeline.py`, `server.py` | The 25 fps player (state machine, base-loop playback, eased cross-fades, barge-in) and the WebSocket API. |

The **GridSample3D** TensorRT plugin comes from `SeanWangJS/grid-sample3d-trt-plugin` (Apache-2.0, pinned in
`scripts/fetch_sources.sh`). `docker/Dockerfile.renderer` compiles it against TensorRT 10.11 with SASS for sm_89 (L40S) and
sm_90 (H100 / H200); another GPU generation needs `--build-arg PLUGIN_ARCHS="80 89 90"` (list every architecture it has to run
on). The prebuilt plugin shipped with Ditto is not used: it fails with "unsupported toolchain" on a 550-series driver.

### Head placement on the real face (`calibrate.py`)

**The wobble.** The live head is the rest face re-posed with the pose, scale and shift that Ditto's motion extractor estimated for
each template frame, and pasted at that frame's face crop. Those estimates are a few percent and a few pixels away from the real
face, and the error is different in every frame (`gulf-man-black`: head scale +-2 % with jumps of up to 4 % from one frame to the
next, shift 4 to 5 px). While the avatar talked, the head swelled, shrank and slid against its own body. The first build (commit
`eff36fd`) warped every frame from that frame's own appearance, where the same errors cancel, so its head kept its size and place on
the face (face width within 0.72 % of the template's, one standard deviation); but its mouth was the template's talking mouth plus
the audio's, which is what the neutral-face driving removed.

**The placement.** When a loop is registered (`Loop.register`, then `Loop.align`) the live head is rendered on silence for every
template frame. MediaPipe finds 16 steady face points (eye corners, nose, brow ends, cheek bones) on that render and on the real
crop of the frame, and a Tukey-weighted least-squares similarity (scale, roll, shift) takes the first onto the second. The fits are
smoothed with a median over three frames, as (log scale, roll, shift of the face centre), and folded into the frame's paste transform
(`M_i' = M_i @ S_i`): the head sits on the real face and moves with it, and the audio only adds the mouth. A fit further off than 15 %
in scale, 10 degrees or 40 px is a detection failure, not a correction, and is interpolated from the neighbouring frames. With a good
fit on fewer than half of the frames `CalibrationError` is raised: the loop is registered without the placement (`aligned` False),
says so in the log and plays like a format 3 loop. The colour residual (`Loop.match_colour`) is measured against the head as it is
drawn live, so no scale error leaks into it along the edge of the face.

Registration must run the engine the server runs. Ditto's own defaults (overlap 10, 50 sampling steps) produced only about 30 of the
81 frames of silence, which is a `CalibrationError`; the server and the Studio use overlap 70 and 10 steps, and `tools/onboard.py`
now defaults to the same (`--overlap`, `--steps`, or `KKA_OVERLAP` / `KKA_STEPS`). Any other tool that calls `Loop.register` needs
the same settings.

**Effect** (`gulf-man-black`, one spoken reply of 10.8 s, `tools/wobble_probe.py` and `tools/motion_probe.py`; the first three columns
come from renderers side by side on the same GPU, the last one from the live renderer after the swap):

| | first build (`eff36fd`) | second build (neutral face, no placement), expression 0.4 | placed, expression 0.4 | placed, expression 0.8 (live) |
|---|---|---|---|---|
| jaw opening (`jawOpen`), p95 | 0.649 | 0.244 | 0.204 | 0.321 |
| inner-lip gap / face height, p95 | 0.187 | 0.115 | 0.096 | 0.130 |
| face width against the template's, std / HF95, % | 0.72 / 1.29 | 1.86 / 1.90 | 0.74 / 1.21 | 1.10 / 1.84 |
| head-size pulse (eye corners), p95 / max, % | 3.33 / 7.25 | 2.74 / 3.79 | 1.22 / 3.21 | 1.27 / 2.68 |
| eyes, nose, brows, cheeks, HF95, px | 0.65 / 1.58 / 0.95 / 0.86 | 1.07 / 1.71 / 2.03 / 1.66 | 0.69 / 1.38 / 1.19 / 1.24 | 0.76 / 1.66 / 1.05 / 1.38 |

The placement of the speaking loop fitted 81 of 81 frames and left an anchor error of 2.6 px (3.98 before); the idle loop 81 of 81
and 0.8 px (2.03 before). The pulse row is not a wobble figure on its own: the template's own eye corners, over the template frames
played in these recordings, pulse by 3 to 4 % at p95 (7 to 8 % at the maximum; its real blinks and landmark noise), and the first
build, which copies the template's face, follows them (3.3 to 3.7 %). Between expression 0.4 and 1.0 the jaw p95 goes from 0.20 to
0.36 and the face width std from 0.74 to 1.2 %,
and the eyes, nose, brows and cheeks stay where they are. The mouth opens to 0.16 of the face width at p95 at 0.8 (first build 0.24,
second build 0.14). The remaining face-width figure grows with the expression because the jaw contour is part of it: the template's jaw
is not in step with the audio.

Only `gulf-man-black` was registered again, and it is the only avatar in the library: the six others made on this host were deleted
in the Studio. An avatar from an older build (format 3 or below) keeps the second build's behaviour (`REST_LEGACY`) until it is
registered again from its clips; the renderer says so in its log when it loads it.
