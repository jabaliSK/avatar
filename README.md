# KK Avatar: a real-time talking avatar from one portrait

You talk to a video avatar in a browser. It listens, answers in a synthetic voice, lip-syncs, blinks, changes expression and
gestures with its hands. One portrait is all it needs. Everything runs in Docker on **one GPU** (GPU 0 on this host, enforced in
[compose.yaml](compose.yaml) and by [scripts/gpu_guard.sh](scripts/gpu_guard.sh)). Every model and every component chosen for the
stack is Apache-2.0 or MIT, except the coturn TURN relay (BSD-3-Clause); the libraries underneath are permissive apart from the
exceptions listed under [Licences](#licences).

```
browser ──WebRTC (audio + H.264)──► gateway: OpenAvatarChat 0.6.0
                                     │  Silero VAD ─► Faster-Whisper ─► Qwen3-8B FP8 (vLLM) ─► Kokoro-82M (CPU)
                                     │       barge-in      ASR              LLM streams text        │ 24 kHz PCM
                                     └──────────────────────────────────────────────────────────────┘
                                                              │ WebSocket
                                              renderer: Ditto head on TensorRT ─► GPU compositor ─► 25 fps frames
                                                              ▲
                              body = the portrait, or the Stage A gesture loop while speaking (made once per avatar)
```

Other machines on the network reach the page over HTTPS on port 8282; their audio and video go through a coturn relay on port 3478.

* Operating it, configuration, troubleshooting: [docs/RUNBOOK.md](docs/RUNBOOK.md)
* What was changed in the vendored Ditto, and why: [services/renderer/PATCHES.md](services/renderer/PATCHES.md)

## Where this stands

Measured on 2026-11-26 on GPU 0 of this host, at HD frame sizes (704 x 1280 for a portrait) and with a 700 ms end-of-speech delay.
**That GPU is an unlicensed vGPU** (`NVIDIA H100L-94C`, NVIDIA Virtual Compute
Server, "Unlicensed"): bf16 matrix multiply peaks at 93 TFLOPS and memory throughput at 0.5 to 0.8 TB/s depending on the test,
several times below what a healthy H100 sustains. NVIDIA documents reduced performance for a vGPU without a licence, which is the
likely cause; confirming it needs a licence token from an administrator, and nothing here works around it. Treat the numbers as a
floor. They have not been measured on an L40S or an unthrottled H100.

| | Target in the brief | Measured here | |
|---|---|---|---|
| Live head and lip-sync | 25-30+ fps | 19.5-20.8 distinct frames/s on a 25 fps clock (video reaches the client at 25.0 fps; 18-23% of the frames are repeats), no audio underruns, all 7 test images | below |
| Lip-sync accuracy | phoneme-accurate | the mouth is 29-44 ms behind the sound at the renderer output (viewers notice from 45 ms), tracked on all 7 recordings; +7 ms (lips) and +16 ms (jaw) through the whole gateway on one run | within, near the limit |
| Compositor | < 5 ms per frame | at HD, standalone ([bench_compositor.py](services/renderer/tools/bench_compositor.py)): median 0.6-0.8 ms, **mean 2.9-4.3 ms** (GPU blend 2.4-3.4, read-back 0.1-0.5, CPU paste 0.3-0.4). About 30% of the calls wait about 8.5 ms for the throttled vGPU, so p95 is 9-11 ms | met on the mean, not at p95 |
| Render stage | | 43-45 ms per frame including compositing, against the 40 ms a 25 fps clock allows (TensorRT FP16; warp 17-21 ms and decoder 18-21 ms on the 512 px head crop) | |
| First frame after first audio | | 0.49-0.52 s through the gateway (0.56-0.60 s renderer alone; 0.95-0.99 s for a 13 s question) | |
| ASR | < 120 ms | 215-400 ms decode (Faster-Whisper large-v3-turbo, FP16, beam 1) | not met |
| LLM | first clause to TTS < 200 ms | first token 60-160 ms after the transcript, first clause about 0.25 s; vLLM median time-to-first-token 53 ms, about 42 tokens/s | close |
| TTS first audio | < 150 ms | 117-171 ms on an idle GPU, but 1.2-3 s while the LLM decodes on the same GPU, so Kokoro runs on the CPU: 0.43-0.77 s for the first clause | not met |
| Whole turn | < 500 ms | first reply sound **2.7-3.0 s** after the user stops speaking (2.68-2.96 s on all 7 avatars through the full gateway; 3.4-3.7 s for a 13 s question). 0.7 s of that is the end-of-speech wait, which was 0.25 s in the earlier measurements | not met |
| Barge-in | instant | the gateway cuts the reply once it has heard 256-288 ms of speech (0.57-0.64 s after the user's first word), the renderer stops 28-40 ms later, and the old sound has ended at the client 0.79 s after the first word | works, not instant |
| Onboarding (Stage A) | about 30 s per image | about 2 min to load the models once, then about 5.5 min per 3.2 s HD clip, 26 GB peak | not met |

`services/gateway/tools/merge_timeline.py` turns a test run into a per-turn budget so the same table can be produced on any
other GPU. See [Measuring latency](docs/RUNBOOK.md#measuring-latency); the recordings, timelines and sheets of every avatar are in
[reports/](reports). The earlier 768 px / 250 ms run is kept in [reports/earlier_768px_250ms/](reports/earlier_768px_250ms): HD
frames cost no measurable frame rate or render time, and the longer end-of-speech wait is what moved the first reply sound, by
0.4-0.65 s per avatar (0.45 s of that is the wait itself).

**Reachable from other machines.** This host serves `https://10.25.73.101:8282` (self-signed certificate, so the browser warns
once). The page is OpenAvatarChat's own, in Chinese by default: click the first button to allow the microphone, then the second to
start. Audio and video go through a coturn relay (UDP 3478, TCP fallback). It was tested with headless Chromium and a fake
microphone on this host, through the relay: 3 of 3 back-to-back conversations on the final build at 704 x 1280 H.264, 24.9 fps, no
dropped frames, first reply sound about 2.5-2.65 s after the end of speech (judged by audio level, so approximate); and with the
aiortc test client over a UDP-only and a TCP-only relay path. **Not tested: another machine, a real microphone, real speakers.**
There is no login, see [Limits](#limits).

## What is deployed, layer by layer

| Layer | Deployed | Differs from the brief because |
|---|---|---|
| 1. Transport, turn-taking | OpenAvatarChat 0.6.0 + Silero VAD, plus the `kka` handlers in `services/gateway/handlers/kka` (barge-in VAD, ASR, TTS, renderer bridge), a coturn TURN relay for other machines, and runtime fixes for reconnects and memory in `services/gateway/rtc_session.py` | Stock OAC is half-duplex; barge-in is a custom handler. coturn is BSD-3-Clause, not Apache-2.0 or MIT |
| 2. ASR | Faster-Whisper large-v3-turbo | SenseVoice-Small's weights are under the FunASR model licence, not Apache-2.0 |
| 3. LLM | Qwen3-8B FP8 on vLLM 0.27.1, thinking off, replies kept to two short sentences | 32B was not deployed: it would not leave room on one shared GPU |
| 4. TTS | Kokoro-82M, voice `am_michael`, on the CPU, with a pronunciation table for HPE and Qatar names | English only; CosyVoice was not deployed |
| 5. Stage A, body and gestures | EchoMimicV3-Flash on Wan2.1-Fun-V1.1-1.3B-InP, **one speaking loop per avatar** | Silent audio gives a shimmering body, so no idle/listening/thinking clips are shipped; those states use the portrait plus the live head |
| 6. Stage B, live head | **Ditto**, MediaPipe only, patched, TensorRT FP16 | SoulX-FlashHead Pro ran at 2.6 fps here (28-frame chunks of 10.8 s), not real time. It is kept as a measured option: `scripts/bench_flashhead.py`, result in `reports/bench_flashhead_native_nocompile.json` |
| 7. Compositor | own GPU blend and CPU paste-back, in `services/renderer/app/compositor.py` | |

The body is the portrait, so the avatar stands still while it listens or thinks, and the head is live (blinks and small
motions). While speaking the Stage A loop supplies the hands and shoulders and the live head is blended over it. After the
last word the body walks back to the resting pose, which adds at most about 1.6 s before the avatar is idle again.

## Quick start

Needs Docker with the NVIDIA runtime, Python 3 with `huggingface_hub`, and about 100 GB of disk: Docker images about 47 GB
(vLLM 21 GB, the rest shares one base image), weights about 45 GB (6.5 GB of that is the optional SoulX-FlashHead benchmark), plus
the TensorRT engines (0.5 GB) and 5-15 MB per avatar, 0.55 GB more with a speaking loop.

```bash
cp .env.example .env                  # set GPU_ID and GPU_PCI_BUS (nvidia-smi --query-gpu=index,pci.bus_id --format=csv)
scripts/fetch_sources.sh              # third-party sources at pinned commits -> third_party/
python3 scripts/fetch_models.py       # allow-listed weights at pinned revisions -> models/
docker build -f docker/Dockerfile.base -t kka-base:cu128 .
docker compose build renderer gateway && docker compose --profile tools build bodygen
scripts/gpu_guard.sh && docker compose up -d
```

A fresh clone has no avatars (`images/`, `avatars/` and `loops/` are not in the repository), so make one first, see [Make an
avatar](#make-an-avatar). Then open <http://localhost:8282>, allow the microphone and talk. Use a headset (see [Limits](#limits)).
On a GPU model the renderer has not seen before, the first start builds the TensorRT engines (20-25 minutes, once). Switch the
avatar for the next session with `echo neri > run/avatar`.

**For other machines** (how this host is set up), add these to `.env` with the host's LAN IP, then `scripts/gpu_guard.sh && docker
compose up -d`, and open <https://10.25.73.101:8282>. Without HTTPS a browser will not give the page a microphone, and without the
relay the media cannot reach the gateway, whose own address is private to this host:

```bash
GATEWAY_BIND_ADDR=10.25.73.101        # the web UI and the TURN relay listen here; the LLM and renderer ports stay on 127.0.0.1
GATEWAY_TLS=1                         # self-signed certificate for this name
GATEWAY_TLS_CN=10.25.73.101
GATEWAY_TLS_SAN=IP:10.25.73.101
COMPOSE_PROFILES=network              # starts the coturn relay
TURN_HOST=10.25.73.101
TURN_PASSWORD=<letters, digits and . _ ~ - only>
```

### Make an avatar

```bash
# Stage A, once per portrait: the speaking gesture loop (about 2 min of model loading + about 5.5 min per clip at HD)
docker compose run --rm bodygen --job aisha=/images/aisha.png --subject "aisha=A woman in a navy blazer" \
    --states speaking --speech /speech/sample_en.wav --sample-area 960
# Register it with the live renderer (15-20 s). Without --loop the avatar is the portrait plus the live head only.
docker compose run --rm onboard --image /images/aisha.png --name aisha --loop speaking=/loops/aisha/speaking.mp4
# Use it from the next session on (no restart)
echo aisha > run/avatar
```

Put portraits in `images/`. Waist-up or full-body framing with visible hands gives the best gestures; the face is rendered at up
to 512 px, so a tight close-up looks soft and a very small face is worse. More in the runbook.

## Licences

Everything below is Apache-2.0 or MIT except coturn (BSD-3-Clause), from the repository `LICENSE` files or model cards checked on
2026-11-25 (coturn: its `LICENSE` at the pinned image revision, 2026-11-26). Exact
revisions are pinned in [scripts/fetch_sources.sh](scripts/fetch_sources.sh) and [scripts/fetch_models.py](scripts/fetch_models.py),
which also hold the deny-list of files that are never downloaded.

| Component | Licence |
|---|---|
| OpenAvatarChat and its web UI, vLLM, Qwen3-8B-FP8, Kokoro-82M (and misaki), EchoMimicV3 (code and Flash weights), Wan2.1-Fun-V1.1-1.3B-InP, Ditto (code), grid-sample3d TensorRT plugin, MediaPipe, SoulX-FlashHead Pro + VAE_Wan (benchmark only) | Apache-2.0 |
| Silero VAD, Faster-Whisper large-v3-turbo, chinese-wav2vec2-base, `landmark203.onnx` (identical to LivePortrait's) | MIT |
| coturn, the TURN relay for other machines (separate container, pinned `coturn/coturn` image; optional, compose profile `network`) | BSD-3-Clause |

Left out on purpose: SenseVoice-Small weights (FunASR model licence), SoulX-FlashHead Lite and its LTX VAE (Open-RAIL-M), and
every InsightFace file Ditto can use for face detection (non-commercial research). Ditto's face detection runs on MediaPipe
instead; the changes are listed in [PATCHES.md](services/renderer/PATCHES.md).

**For HPE legal to review:**
* coturn is BSD-3-Clause, so the "Apache-2.0 or MIT only" rule has one exception, the optional TURN relay.
* The Ditto weight bundle on Hugging Face has no licence tag (the code repository is Apache-2.0), and the provenance of its HuBERT
  model is undocumented.
* The Python libraries are permissive (MIT, BSD, Apache, ISC, PSF) except: NVIDIA's CUDA, cuDNN and TensorRT wheels
  (proprietary, redistributable), `certifi`, `tqdm` and `orjson` (MPL-2.0, used unmodified), and `num2words` (gateway only) and
  `soxr` (LGPL). `scripts/licence_audit.py` produces this list from the images and fails on GPL or unrecognised licences; none were
  found in the gateway, renderer and bodygen images.
* Also in the images: the Ubuntu base system and ffmpeg and x264 (GPL/LGPL, used as separate programs and libraries).
* espeak-ng (GPL-3) is not used and is removed from the final images, but it is still in the lower layers of the shared base
  image until that is rebuilt.

## Layout

```
compose.yaml  .env.example    one stack, one GPU
docker/                       Dockerfile.{base,renderer,gateway,bodygen}, pip constraints, gateway and coturn entrypoints
scripts/                      fetch_sources.sh, fetch_models.py, gpu_guard.sh, licence_audit.py, bench_flashhead.py
services/renderer/            live head renderer (app/), tools/ (onboard, ws_client, lipsync_probe, contact_sheet, test_images, benchmarks), ditto/ (vendored, patched), tests/
services/gateway/             OpenAvatarChat config (config/kka.yaml), the kka handlers, launch.py and rtc_session.py (runtime patches), tools/ (e2e_client, browser_e2e, merge_timeline, check_pronunciations, bench_asr_tts), tests/
services/bodygen/             Stage A: generate_loops.py, prompts.py, speech/
docs/RUNBOOK.md
reports/                      the evidence behind the numbers: timelines, metrics, recordings, contact sheets
third_party/  models/  engines/  avatars/  loops/  images/  run/  logs/  work/    not in the repository: fetched, built or produced locally (.gitignore)
```

## Limits

* **GPU 0 is an unlicensed vGPU and several times slower than a healthy H100**, so latency and frame rate are worse than the
  brief's targets (above). An L40S or a licensed H100 should be better; the plugin is built for sm_89 and sm_90, but nothing has
  run on an L40S.
* **Frame size follows Stage A**: a portrait photo gives 704 x 1280 pixels (landscape 1280 x 720 or 1216 x 752), because the clips
  are made at `--sample-area 960`. The face itself is rendered at up to 512 px, so a larger frame adds body and background, not face
  detail. The web page shows the video in a portrait card, so landscape avatars are letterboxed.
* **Barge-in needs echo cancellation or a headset.** Without it the avatar hears itself through the speakers and cuts itself off.
  Browsers apply echo cancellation to the microphone, but that was not tried with real speakers here. Set `barge_in: false` in
  `services/gateway/config/kka.yaml` to fall back to half-duplex.
* **The end of a question is detected 700 ms after the last word** (`end_delay: 11200` samples at 16 kHz). A normal pause in the
  middle of a question no longer splits it into two turns, and every reply starts 0.4-0.65 s later than it did with 250 ms
  (`end_delay: 4000`; 0.45 s of that is the wait itself). Lower the value for faster replies.
* English only. Pronunciations of the Arabic names in `kka.yaml` are approximations and should be reviewed by a speaker.
  `services/gateway/tools/check_pronunciations.py` lists names the TTS would drop; a Whisper round trip (Marhaba, ProLiant,
  SimpliVity and Kubernetes fine; Msheireb, Lusail, Ooredoo and Kahramaa clearly off) is in the runbook.
* **Switching avatars makes the first session after the switch deaf for about 9 s**: the renderer keeps one avatar loaded and
  loads the new one when a session starts. After changing `run/avatar`, open the page once and wait about 10 s. To avoid it after
  a restart, set `RENDERER_PRELOAD` to the avatar you will use.
* The speaking loop is generic, not driven by the words, and 3.2 s long (played back and forth), so a long answer repeats the
  gesture roughly every 6.5 s. Hands are the weakest part of it: in a fast gesture a hand can blur, or vanish behind the body for a
  moment (see the `neri` sheets in `reports/`), and in `gulf-man-white-2`, whose hands start at the bottom edge of the picture, they
  disappear while the cuffs flare.
* **There is no login.** Anyone who can reach `10.25.73.101:8282` can use the GPU and the LLM, one session at a time; this is for a
  trusted network. The LLM and renderer ports stay on `127.0.0.1`. The TURN user and password are handed to every browser that loads
  the page, and OpenAvatarChat prints them, and the transcripts of what was said, in the gateway log. Docker publishes ports around
  the host firewall (ufw), so `GATEWAY_BIND_ADDR` is what limits where the page and the relay listen.
* **One session at a time** (`concurrent_limit: 1`). The same machine reconnecting (reload, flaky network) replaces its own stale
  session at once; `GATEWAY_RECONNECT_TAKEOVER=0` turns that off. A second machine is refused until the first session has ended,
  up to about 30 s after its tab was closed, and the page shows nothing when that happens: the gateway logs a warning.
* **Memory.** The gateway process idles at about 2.3 GB and takes 3.1-4.3 GB during and just after a session. It hands the memory
  back 10 s after the last session ends. Watch `docker stats`; no memory limit is set.
* **Tested in a headless browser only**: Chromium with a fake microphone on this host, through the relay. Not tested: another
  machine, a real microphone, real speakers, other browsers. The audio/video sync through the gateway was measured on one run.
* Four avatars come from Gemini images and show the Gemini sparkle watermark in the corner. Three (`seated-man`, `neri`,
  `headshot-pn`) are photos of real people: get their consent before showing them to anyone.
* From another machine the microphone needs HTTPS (`GATEWAY_TLS=1`) and the media needs the TURN relay, see Quick start. The server
  side makes no STUN requests to Google (OpenAvatarChat's default is switched off in `services/gateway/launch.py`).
* H.264 is encoded on the CPU (NVENC is unavailable with this driver).
