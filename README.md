# KK Avatar: a real-time talking avatar from one portrait

**Experimental branch:** [LiveAvatar on two H100s and five-GPU comparison](docs/LIVEAVATAR.md).
The description below is the base application, not the LiveAvatar experiment.

You talk to a video avatar in a browser. It listens, answers in a synthetic voice, lip-syncs, blinks and moves like a person
standing in front of you: breathing and swaying while it waits, gesturing with its hands while it speaks. One photo, a portrait or
a full-body shot, is all it needs. The **Avatar Studio** web page (English, HPE theme) lists the avatars, starts conversations and
creates a new avatar from an uploaded photo. Everything runs in Docker on **one GPU** (GPU 1 on this host, enforced in
[compose.yaml](compose.yaml) and by [scripts/gpu_guard.sh](scripts/gpu_guard.sh)). Every model and every component chosen for the
stack is Apache-2.0 or MIT, except the coturn TURN relay (BSD-3-Clause); the libraries underneath are permissive apart from the
exceptions listed under [Licences](#licences).

```
browser ──WebRTC (audio + H.264)──► gateway: OpenAvatarChat 0.6.0 + Avatar Studio (the page: library, conversation, new avatar)
                                     │  Silero VAD ─► Faster-Whisper ─► Qwen3-8B FP8 (vLLM) ─► Kokoro-82M (CPU)
                                     │       barge-in      ASR              LLM streams text        │ 24 kHz PCM
                                     └──────────────────────────────────────────────────────────────┘
                                                              │ WebSocket
                                              renderer: Ditto head on TensorRT ─► GPU compositor ─► 25 fps frames
                                                              ▲
        body = a full-body video of the person, made once per avatar (Stage A): the idle clip while it waits or listens, the
        speaking clip while it talks. The live head is drawn over that video, from the photo's own resting face.

Studio ─► worker: photo ─► Stage A (EchoMimicV3-Flash) ─► idle + speaking clips ─► renderer registers the avatar
```

Other machines on the network reach the page over HTTPS on port 8282; their audio and video go through a coturn relay on port 3478.

* Operating it, configuration, the Studio, troubleshooting: [docs/RUNBOOK.md](docs/RUNBOOK.md)
* What was changed in the vendored Ditto, and why: [services/renderer/PATCHES.md](services/renderer/PATCHES.md)

## Where this stands

The stack has run on **GPU 1** of this host since 2026-11-26, 12:45 UTC. Both GPUs of this VM are unlicensed vGPUs (`NVIDIA H100L-94C`,
NVIDIA Virtual Compute Server, "Unlicensed"), and neither has been dependable. GPU 0 computed at about a fifth of an H100's speed for
two days and then stopped accepting compute. **GPU 1 was fast for only the first 20 to 28 minutes**: by 13:11 UTC it computed at GPU 0's
slow speed, and a repeat of the measurement at 13:30 gave GPU 0's numbers ([below](#gpu-1-slowed-down)). It still accepts compute, and
it may refuse it as GPU 0 did (see [Limits](#limits)). Nothing here works around the licensing.

**On GPU 1, while it was fast** (12:44 to 13:04 UTC), bf16 matrix multiply measured 515 TFLOPS and memory throughput 3.3 TB/s. This is a
spot check of the current build on `gulf-man-black` (three questions through the full gateway, HD frames, the 700 ms end-of-speech
delay, 12:48 to 13:05 UTC); the per-stage numbers are in the [runbook](docs/RUNBOOK.md#results-on-gpu-1) and the evidence in
[reports/gpu1_spot_check/](reports/gpu1_spot_check):

| | Target in the brief | Measured on GPU 1, while it was fast | |
|---|---|---|---|
| Live head and frame rate | 25-30+ fps | 25.0 fps at the client (mean interval 40.0 ms, p95 44.2 ms), none of 767 frames repeated; a headless browser decoded 24.9-25.0 fps with no dropped frame and no freeze | met |
| Render stage | | 8.8-9.7 ms per frame including compositing, against the 40 ms a 25 fps clock allows | within budget |
| First frame after first audio | | 106-136 ms through the gateway | |
| ASR | < 120 ms | 53-73 ms decode (Faster-Whisper large-v3-turbo, FP16, beam 1); the text is ready 0.88-1.02 s after the user stops, 0.7 s of which is the end-of-speech wait | met (decode) |
| TTS first audio | < 150 ms | Kokoro runs on the CPU: 0.74-0.87 s for the first clause | not met |
| Whole turn | < 500 ms | first reply sound **2.07-2.25 s** after the user stops speaking: 0.7 s of that is the end-of-speech wait and about 0.8 s the speech of the first clause | not met |

Not measured on GPU 1 while it was fast: lip-sync accuracy, the compositor alone, barge-in, Stage A, the steadiness of the head,
avatars other than `gulf-man-black`, and anything from another machine. Kokoro on the GPU and `RENDERER_OVERLAP=75`, which might
have shortened the first reply with the render stage that far below budget, were not tried, and the budget is gone now.

<a id="gpu-1-slowed-down"></a>**Then GPU 1 slowed down.** ASR decode in a live conversation took 241-244 ms at 13:11 UTC (50-83 ms up to 13:03), and the
same three questions repeated at 13:30, with no other process on the GPU, gave GPU 0's numbers: render 42.8-44.3 ms per frame, 19-22 %
of the frames repeated, ASR decode 233-289 ms, first frame 463-527 ms after the first audio and first reply sound 2.79-3.02 s
([reports/gpu1_slowdown/](reports/gpu1_slowdown)). The slowdown was already there before the Stage A job below started, and it
was still there after the job ended, so the job did not cause it. The GPU still accepts compute and no log shows an error. The VM cannot
see why it slowed down, and the quick GPU check was not repeated, so as not to open another CUDA context on a GPU that may refuse. The
table above is what GPU 1 did for 20 to 28 minutes, not what it does now.

**Stage A on GPU 1** ran once, through the Studio on the real motion model: a full-motion HD job for a new avatar, started from the
Studio page at 13:14:51 and ready at 13:29:15 (14.4 minutes): 154 s loading the models, 322 s for the idle clip, 318 s for the
speaking clip and 62 s registering the avatar with the renderer. That is GPU 0's time (about 330 s per clip), with GPU 1 in the
slowed state described above. Whether the clips are limited by the CPU (the generator sat at about 100 % of one core while the GPU
showed 0 to 8 % utilization, in two windows of 12 and 20 samples) or by the slowed GPU was not profiled. Each clip has 81 frames at
704 x 1280 and 25 fps. They were checked by numbers (the first frame matches the photo, the idle clip moves little from frame to
frame and ends near where it began), not by eye, and no conversation with the new avatar was measured.

**On GPU 0** the first build (`eff36fd`) was measured earlier the same day, with seven avatars (six were deleted from the library
afterwards, see [Limits](#limits)), at HD frame sizes (704 x 1280 for a portrait) and with the same 700 ms end-of-speech delay, while
that GPU computed at 93 TFLOPS and 0.5 to 0.8 TB/s of memory throughput depending on the test, several times below what a healthy H100
sustains. NVIDIA documents reduced performance for a vGPU without a licence; that may be the cause, but it is not established. Later
the same day GPU 0 stopped giving any compute at all. The current build, recorded on `gulf-man-black` only, was in the same ranges
there: first frame 0.57 s after the first audio, 43.0 ms per frame, 21 % of the frames repeated, no audio underruns.

| | Target in the brief | Measured on GPU 0 | |
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

**Head steadiness** (`gulf-man-black`, the same 10.8 s spoken reply each time, measured from the face landmarks of the recorded
frames with [wobble_probe.py](services/renderer/tools/wobble_probe.py) and [motion_probe.py](services/renderer/tools/motion_probe.py)).
The head used to swell, shrink and slide against its own body while the avatar talked. It is now placed on the real face of every
frame of the body clips when the avatar is registered ([PATCHES.md](services/renderer/PATCHES.md#head-placement-on-the-real-face-calibratepy)).
The width of the head's face, relative to the face in the body video, varies with a standard deviation of 0.72 % in the first build
(`eff36fd`, whose mouth opened wide: jaw opening 0.65 at p95), 1.86 % in the build that made the mouth subtle without placing the head
(jaw 0.24), and 0.74 % (jaw 0.20) with the placement at expression 0.4 or 1.10 % (jaw 0.32) at 0.8, the value set for this avatar. So
the head is as steady as in the first build at 0.4 and a little less so at 0.8, with a far calmer mouth; the width includes the jaw
line, so it grows with the expression. `gulf-man-black` is the only avatar that was measured this way; a second one, `newavatar`, was
made through the Studio at 13:29 and its head was placed too, but its steadiness was not measured, see [Limits](#limits). The
test suites (117 gateway, 57 renderer, 15 worker) pass, see [Tests](docs/RUNBOOK.md#tests).

**Reachable from other machines.** This host serves `https://10.25.73.101:8282` (self-signed certificate, so the browser warns
once, and again whenever the gateway container is re-created: it makes a new certificate). The page is the Avatar Studio: open an
avatar, press **Start conversation** and allow the microphone. Audio and video go
through a coturn relay (UDP 3478, TCP fallback). It was tested with headless Chromium and a fake
microphone on this host, through the relay: 3 of 3 back-to-back conversations on the first build (`eff36fd`) at 704 x 1280 H.264,
24.9 fps, no dropped frames, first reply sound about 2.5-2.65 s after the end of speech (judged by audio level, so approximate); and
with the aiortc test client over a UDP-only and a TCP-only relay path. The current build was checked with the same headless
browser on `gulf-man-black` (direct path, not through the relay), once on GPU 0 and twice on GPU 1 while it was fast: 24.9-25.0 fps, no dropped
frames, no freezes (its question reached the gateway as two turns, so its reply time is not comparable). **Not tested: another
machine, a real microphone, real speakers.**
There is no login for talking to an avatar, and an optional key for creating and deleting them, see [Limits](#limits).

## What is deployed, layer by layer

| Layer | Deployed | Differs from the brief because |
|---|---|---|
| 1. Transport, turn-taking | OpenAvatarChat 0.6.0 + Silero VAD, plus the `kka` handlers in `services/gateway/handlers/kka` (barge-in VAD, ASR, TTS, renderer bridge), a coturn TURN relay for other machines, runtime fixes for reconnects and memory in `services/gateway/rtc_session.py`, and the **Avatar Studio** (`services/gateway/studio/`), an English web page that replaces OpenAvatarChat's own, Chinese one | Stock OAC is half-duplex; barge-in is a custom handler. coturn is BSD-3-Clause, not Apache-2.0 or MIT |
| 2. ASR | Faster-Whisper large-v3-turbo | SenseVoice-Small's weights are under the FunASR model licence, not Apache-2.0 |
| 3. LLM | Qwen3-8B FP8 on vLLM 0.27.1, thinking off, replies kept to two short sentences, **English only**: the handler (`handlers/kka/llm/english_llm.py`) strips Chinese, Japanese and Korean characters from the stream and answers errors in English. It also stops the words of a reply that was cancelled (the person spoke again) from leaking into the next one, a race in the stock handler | 32B was not deployed: it would not leave room on one shared GPU |
| 4. TTS | Kokoro-82M, voice `am_michael`, on the CPU, with a pronunciation table for HPE and Qatar names | English only; CosyVoice was not deployed |
| 5. Stage A, body and gestures | EchoMimicV3-Flash on Wan2.1-Fun-V1.1-1.3B-InP, **two clips per avatar: idle and speaking**. The `worker` service runs it for the Studio | Two clips, not three: listening and thinking play the idle clip |
| 6. Stage B, live head | **Ditto**, MediaPipe only, patched, TensorRT FP16. Drawn over the body clips from the photo's own resting face and placed on the real face of each clip frame, with the mouth driven by the audio at a subtle, adjustable strength | SoulX-FlashHead Pro ran at 2.6 fps here (28-frame chunks of 10.8 s), not real time. It is kept as a measured option: `scripts/bench_flashhead.py`, result in `reports/bench_flashhead_native_nocompile.json` |
| 7. Compositor | own GPU blend and CPU paste-back, in `services/renderer/app/compositor.py` | |

**The body is a real video of the person.** Stage A makes an *idle* clip (breathing, weight shifting, relaxed arms, mouth closed;
both ends are the photo's pose) and a *speaking* clip (hand and body gestures). Each is 3.2 s long and is played back and forth, so a
movement repeats about every 6.5 s. While the avatar waits, listens or thinks, the idle clip plays, so it is never a frozen picture.
While it speaks, the speaking clip supplies hands, shoulders and head pose. In both cases the live head is rendered over the clip from
the **resting face** of its first frame (the photo's own face), so the mouth of the generated video never shows through and the only
mouth movement is the audio-driven one. The jaw and lips move at a subtle strength (`expression`, 0.6 by default; change it per
avatar in the Studio), the head takes its pose and position from the clip's current frame, so it moves with the generated body
(`head_motion`, 0.5, only matters to a still photo, which has no motion of its own), and blinks come from the clip or, when it has
none, are synthesised. The head crop follows the eyes, not the jaw, so the head does not swell when the clip's mouth opens wide.
The pose and size that Ditto estimates for a clip frame are a few percent off, differently in every frame, and used as they are they
made the head swell, shrink and slide against the body while it talked. So when an avatar is registered, the renderer now renders
the head on silence for every clip frame, measures where it sits on the clip's real face and stores a correction for each frame
([PATCHES.md](services/renderer/PATCHES.md#head-placement-on-the-real-face-calibratepy)). `gulf-man-black` and `newavatar`, the two
avatars in the library, were registered this way; an avatar from an older build can be registered again from its clips on the command line, in
about a minute (see [Limits](#limits)).
After the last word the head and body walk back to the clip's first pose, which takes about 1.3 s on average and at most about 2.7
s, depending on where the gesture stopped; the idle clip then takes over with a cross-fade between two identical poses, so a reply
ends without a jolt. An avatar without an idle clip falls back to the still photo plus the live head, which is what the quick
preview in the Studio is.

## Quick start

Needs Docker with the NVIDIA runtime, Python 3 with `huggingface_hub`, and about 100 GB of disk: Docker images about 47 GB
(vLLM 21 GB, the rest shares one base image), weights about 45 GB (6.5 GB of that is the optional SoulX-FlashHead benchmark), plus
the TensorRT engines (0.5 GB) and 5-15 MB per avatar, about 0.45 GB more for the two Stage A clips of an HD avatar.

```bash
cp .env.example .env                  # set GPU_ID and GPU_PCI_BUS (nvidia-smi --query-gpu=index,pci.bus_id --format=csv)
scripts/fetch_sources.sh              # third-party sources at pinned commits -> third_party/
python3 scripts/fetch_models.py       # allow-listed weights at pinned revisions -> models/
docker build -f docker/Dockerfile.base -t kka-base:cu128 .
docker compose build renderer gateway worker      # the worker and the bodygen tool share one image
scripts/gpu_guard.sh && docker compose up -d
```

A fresh clone has no avatars (`images/`, `avatars/`, `loops/` and `studio/` are not in the repository). Open <http://localhost:8282>
and create one with **New avatar** (see [Make an avatar](#make-an-avatar)); the Studio shows its progress. Then open the avatar,
press **Start conversation**, allow the microphone and talk. Use a headset (see [Limits](#limits)). On a GPU model the renderer has
not seen before, the first start builds the TensorRT engines (20-25 minutes, once). Opening an avatar's page selects it for the
conversation (the Studio writes `run/avatar`, which `echo <name> > run/avatar` sets by hand).

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

**In the Studio** (the usual way): press **New avatar**, drop a PNG, JPEG or WebP photo, a portrait or a full-body shot, name it
and describe the person in English in a sentence ("A woman in a navy blazer in a bright office"). Choose **Full motion** (the idle
and speaking body video; 14.4 minutes at HD quality, measured once through the Studio on GPU 1, and about 8 at Standard, estimated from GPU 0 runs and not timed on GPU 1) or **Quick preview** (the photo plus the live head,
about a minute). The library shows the progress, and the avatar can be used as soon as it is registered. A quick preview can
be upgraded later with **Add full motion**. The settings dialog sets how subtle the face is.

**From the command line** (the same result without the Studio):

```bash
# Stage A, once per portrait: the idle and speaking clips (about 2 to 2.5 min of model loading + about 5.4 min per clip at HD, on GPU 0 and on GPU 1)
docker compose run --rm bodygen --job aisha=/images/aisha.png --subject "aisha=A woman in a navy blazer" \
    --states idle,speaking --speech /speech/sample_en.wav --sample-area 960
# Register it with the live renderer (about a minute for the two clips: it also places the head on every frame's real face).
# Without --loop the avatar is the photo plus the live head only.
docker compose run --rm onboard --image /images/aisha.png --name aisha \
    --loop idle=/loops/aisha/idle.mp4 --loop speaking=/loops/aisha/speaking.mp4
# Use it from the next session on (no restart)
echo aisha > run/avatar
```

Put portraits in `images/`. Waist-up or full-body framing with visible hands gives the best gestures; the face is rendered at up
to 512 px, so a tight close-up looks soft and a very small face is worse. To register an avatar that exists again (for the head
placement, say), see [Registering an avatar again](docs/RUNBOOK.md#registering-an-avatar-again). More in the runbook.

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
services/renderer/            live head renderer (app/, including the /v1 admin API the Studio uses), tools/ (onboard, ws_client, motion_probe, wobble_probe, lipsync_probe, contact_sheet, test_images, benchmarks), ditto/ (vendored, patched), tests/
services/gateway/             OpenAvatarChat config (config/kka.yaml), the kka handlers, launch.py and rtc_session.py (runtime patches), studio/ (the Avatar Studio: API and the page), tools/ (e2e_client, browser_e2e, merge_timeline, check_pronunciations, bench_asr_tts), tests/
services/bodygen/             Stage A: generate_loops.py, prompts.py, speech/; worker.py, the Studio's job runner
docs/RUNBOOK.md
reports/                      the evidence behind the numbers: timelines, metrics, recordings, contact sheets
third_party/  models/  engines/  avatars/  loops/  studio/  images/  run/  logs/  work/    not in the repository: fetched, built or produced locally (.gitignore)
```

## Limits

* **Both GPUs of this VM are unlicensed vGPUs, and neither has been dependable.** GPU 0 computed at about a fifth of an H100's speed
  (93 TFLOPS) for about two days, which is why the first measurements are far from the brief's targets, and from about 11:13 UTC on
  2026-11-26 it refuses all CUDA work: `cuCtxCreate` returns error 801 from the host and from every container, and vLLM's engine
  stopped with `CUDA error: device doesn't have valid Grid license`. GPU 1 refused too at 12:39, with processes of other workloads on
  it; those were stopped on the owner's instruction at about 12:41, and GPU 1 accepted CUDA from 12:43 and was fast (515 TFLOPS, 9 ms
  to render a frame) for 20 to 28 minutes. By 13:11 it computed at GPU 0's slow speed: at 13:30 a frame took 43 ms, a fifth of the
  frames repeated and the first reply sound came after about 3 s (see [Where this stands](#gpu-1-slowed-down)). It has not refused
  work since. Why GPU 1 differs from GPU 0, why it was fast at first and why it slowed down are not known, and it may refuse work as
  GPU 0 did: if it does the stack stops (`docker compose stop` ends the restart loops) and the runbook says what to look at.
  `nvidia-smi -q` reports `License Status : Unlicensed` on both, and the VM has no licence client (no token in
  `/etc/nvidia/ClientConfigToken/`, no `gridd.conf`, no `nvidia-gridd` service). The way to a dependable GPU is an NVIDIA License
  System administrator (a client configuration token), or a licensed vGPU or physical GPU on another VM or host; nothing here works
  around it. See [the runbook](docs/RUNBOOK.md#gpu-placement-and-the-vgpu-licence). An L40S or a licensed H100 has not been tried; the
  plugin is built for sm_89 and sm_90.
* **Frame size follows Stage A**: a portrait photo gives 704 x 1280 pixels (landscape 1280 x 720 or 1216 x 752), because the clips
  are made at `--sample-area 960`. The face itself is rendered at up to 512 px, so a larger frame adds body and background, not face
  detail. The stage page shows the video in the avatar's own proportions; the library crops every thumbnail to a portrait (4:5)
  window, so landscape and square avatars are cut off there.
* **Barge-in needs echo cancellation or a headset.** Without it the avatar hears itself through the speakers and cuts itself off.
  Browsers apply echo cancellation to the microphone, but that was not tried with real speakers here. Set `barge_in: false` in
  `services/gateway/config/kka.yaml` to fall back to half-duplex.
* **The end of a question is detected 700 ms after the last word** (`end_delay: 11200` samples at 16 kHz). A normal pause in the
  middle of a question no longer splits it into two turns, and every reply starts 0.4-0.65 s later than it did with 250 ms
  (`end_delay: 4000`; 0.45 s of that is the wait itself). Lower the value for faster replies.
* English only, in the page, the replies and the speech. The LLM is told to answer in English and a filter drops any Chinese,
  Japanese or Korean characters it still produces. Pronunciations of the Arabic names in `kka.yaml` are approximations and should
  be reviewed by a speaker.
  `services/gateway/tools/check_pronunciations.py` lists names the TTS would drop; a Whisper round trip (Marhaba, ProLiant,
  SimpliVity and Kubernetes fine; Msheireb, Lusail, Ooredoo and Kahramaa clearly off) is in the runbook.
* **Switching avatars, or changing an avatar's settings, reloads it into the renderer**: 13-15 s in the renderer's log, because the
  idle clip is rendered once through the live head at load time. The stage page shows "Loading the face model" and enables **Start
  conversation** when the avatar is ready, so nothing said in that time is lost. Starting a conversation by other means (the
  stock WebRTC client, `e2e_client.py`) needs `--lead-in 18` or more after a switch (not run: the figure is the load time plus a
  margin). To avoid the wait after a restart, set `RENDERER_PRELOAD` to the avatar you will use.
* The body clips are generic, not driven by the words, and 3.2 s long. Both are played back and forth, so a movement repeats about
  every 6.5 s: a long answer repeats its gesture, and the idle clip, which only breathes and sways, is hard to notice. Hands are the
  weakest part of the speaking clip: in a fast gesture a hand can blur, or vanish behind the body for a
  moment (see the `neri` sheets in `reports/`), and in `gulf-man-white-2`, whose hands start at the bottom edge of the picture, they
  disappear while the cuffs flare. A source photo with an open mouth (`neri`) keeps a slightly open mouth at rest, because the
  live face starts from the photo's; `gulf-man-black` smiles with teeth in its photo, so its mouth never closes fully or purses.
* **The library holds two avatars.** `gulf-man-black` has the head placed on the real face of each clip frame (cache format 4,
  expression 0.8; 0.4-0.6 is calmer and steadier, 1.0 the most expressive). `newavatar` came from the Studio page at 13:15-13:29 UTC
  on 2026-11-26 (full motion, HD, created with the default expression 0.6); at 13:33 it was set to 1.2, the highest value the page
  allows (the jaw opening was measured up to 1.0 and rises with the value; lower it in the Studio for the subtle look). The renderer placed its head
  too (cache format 4, 81 of 81 frames fitted in each clip), and its clips were checked by numbers only. The six others made on this
  host were deleted through the Studio on 2026-11-26, after the measurements above, and their clips went with them (the photos are
  still in `images/`); the Studio's delete cannot be undone, and anyone on the network can use it while `STUDIO_KEY` is empty. An
  avatar from an older build (cache format 3 or below) wobbles a little until it is registered again from its clips, which the Studio
  has no button for ([runbook](docs/RUNBOOK.md#registering-an-avatar-again)). One Studio job has run on the real motion model, the
  one above; a Standard-quality job, cancel and retry were not run on it.
* **Talking to an avatar needs no login.** Anyone who can reach `10.25.73.101:8282` can use the GPU and the LLM, one session at a
  time; this is for a trusted network. Creating, changing and deleting avatars is open too unless `STUDIO_KEY` is set in `.env`;
  the Studio then asks for the key once per browser tab (it is a shared password, not user accounts). The LLM, the renderer
  (including its `/v1` admin API) and the job runner have no login and stay on the compose network and `127.0.0.1`. The TURN user
  and password are handed to every browser that loads the page, and OpenAvatarChat prints them, and the transcripts of what was
  said, in the gateway log. Docker publishes ports around the host firewall (ufw), so `GATEWAY_BIND_ADDR` is what limits where
  the page and the relay listen. Uploaded photos are of real people: the Studio does not check that the uploader has the right
  to animate them.
* **The logo in the Studio is a placeholder**, a generic mark drawn for this page (`services/gateway/studio/static/brand/mark.svg`),
  not HPE's logo. Replace it with the approved asset before a customer sees it.
* **One session at a time** (`concurrent_limit: 1`). The same machine reconnecting (reload, flaky network) replaces its own stale
  session at once; `GATEWAY_RECONNECT_TAKEOVER=0` turns that off. A second machine is refused until the first session has ended,
  up to about 30 s after its tab was closed, and the page shows nothing when that happens: the gateway logs a warning.
* **Memory.** The gateway process idles at about 2.3 GB and takes 3.1-4.3 GB during and just after a session. It hands the memory
  back 10 s after the last session ends. Watch `docker stats`; no memory limit is set.
* **Tested in a headless browser only**: Chromium with a fake microphone on this host, through the relay. Not tested: another
  machine, a real microphone, real speakers, other browsers. The audio/video sync through the gateway was measured on one run. The
  Studio's create flow was run end to end on the real motion model once (the full-motion HD job above, started from the Studio page);
  the rest of it (quick preview, Standard quality, cancel, retry, upgrade) was run only with a stand-in for the motion model (the
  tests and a real renderer). The clips of the six avatars made earlier were made with `docker compose run bodygen`, which is the
  generator the worker uses, not through the
  Studio.
* Four of the avatars made here came from Gemini images and show the Gemini sparkle watermark in the corner. Three (`seated-man`,
  `neri`, `headshot-pn`) are photos of real people: get their consent before showing them to anyone; their test recordings and
  contact sheets are in `reports/`.
* From another machine the microphone needs HTTPS (`GATEWAY_TLS=1`) and the media needs the TURN relay, see Quick start. The server
  side makes no STUN requests to Google (OpenAvatarChat's default is switched off in `services/gateway/launch.py`).
* H.264 is encoded on the CPU (NVENC is unavailable with this driver).
