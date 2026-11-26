# Runbook

Starting, configuring, measuring and fixing the avatar stack. Commands are run from the project root. Everything was run on this
host (GPU 0) on 2026-11-26 (the earlier 768 px / 250 ms run of 2026-11-25 is kept in `reports/earlier_768px_250ms/`); anything that
was not run is marked **untested** or listed under [Not verified](#not-verified).

* [What runs where](#what-runs-where)
* [Start, stop, update](#start-stop-update)
* [GPU placement and the vGPU licence](#gpu-placement-and-the-vgpu-licence)
* [Configuration](#configuration)
* [Avatars](#avatars)
* [Measuring latency](#measuring-latency)
* [Pronunciations](#pronunciations)
* [Tests](#tests)
* [Network, security and privacy](#network-security-and-privacy)
* [One session at a time, reconnects and memory](#one-session-at-a-time-reconnects-and-memory)
* [Moving to another GPU](#moving-to-another-gpu)
* [Licence audit](#licence-audit)
* [Troubleshooting](#troubleshooting)
* [Not verified](#not-verified)

## What runs where

Four containers (compose project `kkavatar`). The first three share GPU 0; coturn does not use a GPU.

| Container | Does | Host port | GPU memory |
|---|---|---|---|
| `kka-vllm` | Qwen3-8B FP8 on vLLM 0.27.1, OpenAI-compatible API, thinking off | `127.0.0.1:18100` | 21.3 GB (8.8 GiB of weights, the rest KV cache; `LLM_GPU_MEM_UTIL`) |
| `kka-renderer` | Ditto head on TensorRT, GPU compositor, 25 fps player. WebSocket `/ws`, `GET /healthz` | `127.0.0.1:18110` | 5.4 GB |
| `kka-gateway` | OpenAvatarChat web UI and WebRTC, Silero VAD with barge-in, Faster-Whisper (GPU), Kokoro (CPU), bridge to the renderer | `GATEWAY_BIND_ADDR:8282`, here `10.25.73.101:8282` (HTTPS) | 3.0 GB |
| `kka-coturn` | TURN relay (coturn 4.19.2, BSD-3-Clause) for browsers on other machines; only with `COMPOSE_PROFILES=network` | `GATEWAY_BIND_ADDR:3478`, UDP and TCP | none |

The three GPU containers hold 29,619 MiB of the card's 96,256 MiB (`nvidia-smi`; the GB figures above are the MiB values divided by 1000).

`onboard` and `bodygen` (compose profile `tools`) are one-shot containers that make avatars; they use the same GPU.

| On the host | Holds |
|---|---|
| `models/` | weights, mounted read-only |
| `images/` | portraits (input of Stage A and onboarding) |
| `loops/<name>/` | Stage A output: `speaking.mp4`, `speaking.json` (settings and QA numbers), `speaking_sheet.jpg` |
| `avatars/<name>/` | registered avatars the renderer loads: `meta.json`, `source.<ext>`, `still/`, `speaking/` |
| `engines/sm<cc>_trt<version>/` | TensorRT engines, built on the first start on a GPU model (here `sm90_trt10.11.0.33`) |
| `run/avatar` | the avatar the next session uses |
| `third_party/` | pinned sources: OpenAvatarChat, Ditto, EchoMimicV3, SoulX-FlashHead, the grid-sample3d plugin |
| `reports/` | recorded test results (see [Results on this host](#results-on-this-host)) |
| `logs/`, `work/`, `.cache/vllm` | build logs, scratch output of the commands below, vLLM's compile cache |

Everything in this table except `reports/` is fetched, built or produced on the host and is not in the git repository (see
`.gitignore`), and so is `.env` (copy `.env.example`). A fresh clone therefore has no avatar: make one under
[Avatars](#avatars) first, then write its name into `run/avatar`.

## Start, stop, update

```bash
scripts/gpu_guard.sh && docker compose up -d   # vLLM first (about 105 s), then the renderer (about 20 s), the gateway (about 36 s) and coturn
docker compose ps                              # all four "healthy" (coturn only with COMPOSE_PROFILES=network)
docker compose logs -f gateway                 # follow one service
docker compose stop                            # `down` also removes the containers; models, avatars and engines stay on the host
```

`up -d` waits for each health check before it starts the next service, so it does not return for 2.5 to 3 minutes (run it in the
background from a script). Health by hand:

```bash
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:18100/health   # vLLM: 200
curl -s http://127.0.0.1:18110/healthz                                   # renderer: {"status":"ready","avatars":[...],"loaded":[...],"session_active":false,...}
curl -skI https://10.25.73.101:8282/ | head -1                           # gateway: 307, redirects to /ui/index.html (-k: self-signed certificate; plain http://127.0.0.1:8282 without GATEWAY_TLS)
```

* **First start on a GPU model the renderer has not seen:** it builds the TensorRT engines into `engines/` (20 to 25 minutes, once),
  answers 503 on `/healthz` meanwhile, and the health check allows 35 minutes. Watch `docker logs -f kka-renderer`.
* **Changing settings:** `.env` changes apply with `docker compose up -d`. `services/gateway/config/kka.yaml`, the gateway handlers
  and the renderer code are baked into the images: `docker compose build gateway renderer` (about 1 min 40 s), then `up -d`.
* **Versions are pinned** (vLLM digest in `compose.yaml`, sources and weights by commit in `scripts/fetch_*`). Do not move to `latest`.
* **After a host reboot** Docker restarts the containers itself (`restart: unless-stopped`) but does not honour the start order.
  This was not tested (the host is shared). If something is unhealthy: `docker compose stop && scripts/gpu_guard.sh && docker compose up -d`.

## GPU placement and the vGPU licence

* Every service gets `device_ids: ["${GPU_ID}"]` in `compose.yaml`, so a container sees one GPU and nothing else.
  `scripts/gpu_guard.sh` refuses to continue unless `GPU_ID` in `.env` is the card whose PCI address is `GPU_PCI_BUS`. GPU 1 on this
  host belongs to other workloads: never point the stack at it.
* Check the pinning (each line must end in `[0]`) and what each container sees:

  ```bash
  docker inspect kka-vllm kka-renderer kka-gateway --format '{{.Name}} {{range .HostConfig.DeviceRequests}}{{.DeviceIDs}}{{end}}'
  docker exec kka-renderer nvidia-smi -L
  ```

* **GPU 0 is an unlicensed vGPU** (`NVIDIA H100L-94C`, NVIDIA Virtual Compute Server):

  ```bash
  nvidia-smi -i 0 -q | grep -i -E "licens|vGPU Software|Product Name"     # License Status : Unlicensed
  ```

  NVIDIA documents that a vGPU for Compute without a licence runs with reduced performance until one is acquired
  ([NVIDIA vGPU for Compute licensing](https://docs.nvidia.com/ai-enterprise/release-7/latest/infra-software/vgpu/licensing.html)),
  and GPU 0 measures several times below a healthy H100 (below). That is the likely reason the latency targets are missed. It was
  not confirmed, because that needs a client configuration token from the organisation's NVIDIA License System administrator. The
  stack does nothing about it: do not try to bypass licensing.
* **Quick GPU check** (10 seconds; run it on any new GPU and compare). Here it prints 93 TFLOPS and 0.65 to 0.77 TB/s of memory
  throughput, against ratings of several hundred TFLOPS and 2 to 4 TB/s for an H100:

  ```bash
  docker run --rm --gpus '"device=0"' --entrypoint python kka-renderer:latest -c "
  import torch, time
  def bench(fn, n=20):
      fn(); torch.cuda.synchronize(); t = time.time()
      for _ in range(n): fn()
      torch.cuda.synchronize(); return (time.time() - t) / n
  a = torch.randn(8192, 8192, device='cuda', dtype=torch.bfloat16); b = torch.randn_like(a)
  print(f'bf16 matmul  {2 * 8192 ** 3 / bench(lambda: a @ b) / 1e12:6.0f} TFLOPS')
  x = torch.empty(1 << 30, device='cuda', dtype=torch.uint8); y = torch.empty_like(x)
  print(f'memory copy  {2 * x.numel() / bench(lambda: y.copy_(x)) / 1e9:6.0f} GB/s')"
  ```

* Two processes on one GPU slow each other down. Latency tests need a quiet GPU: nobody talking to the avatar, no Stage A run.

## Configuration

### `.env`

`.env.example` documents every key. The ones that matter:

| Key | Default | Meaning |
|---|---|---|
| `GPU_ID`, `GPU_PCI_BUS` | `0`, `00000000:03:00.0` | the one GPU; checked by `scripts/gpu_guard.sh` |
| `BIND_ADDR` | `127.0.0.1` | address the vLLM and renderer ports are published on (and the gateway's, unless `GATEWAY_BIND_ADDR` is set). Those two services have no login: leave them on `127.0.0.1` |
| `GATEWAY_BIND_ADDR` | empty (= `BIND_ADDR`) | address of the gateway web UI and the TURN relay only. Set to this host's LAN IP (here `10.25.73.101`) to serve other machines; there is no login (see [Network](#network-security-and-privacy)) |
| `GATEWAY_PORT`, `LLM_PORT`, `RENDERER_PORT` | `8282`, `18100`, `18110` | host ports; pick free ones (`ss -ltn`). Other services on this host use, among others, 5432, 6333, 8001, 8060, 8888, 9000 to 9106 (in part), 11434, 19530 and 27017 |
| `LLM_GPU_MEM_UTIL`, `LLM_MAX_MODEL_LEN` | `0.22`, `8192` | vLLM's share of the card's memory and its context length |
| `GATEWAY_AVATAR` | `gulf-man-white` (compose's fallback when empty: the name of an avatar a fresh clone does not have) | seeds `run/avatar` the first time only; afterwards edit that file |
| `RENDERER_PRELOAD` | empty (`gulf-man-white` on this host) | avatar loaded at boot; empty loads none and each avatar loads when a session first asks for it. The renderer keeps one avatar in memory |
| `RENDERER_OVERLAP` | `70` | `70` = about 1.0 s of speech before the first frame; `75` = about 0.4 s at about twice the motion cost. Left at 70 because the render stage is already over budget here; worth trying on a faster GPU (**untested**) |
| `RENDERER_STEPS` | `10` | sampling steps of Ditto's motion model; not re-tuned |
| `GATEWAY_TLS`, `GATEWAY_TLS_CN`, `GATEWAY_TLS_SAN` | `0`, `localhost`, empty | HTTPS with a self-signed certificate. Browsers allow the microphone only on https or `localhost`, so another machine needs `GATEWAY_TLS=1`, CN set to the address it opens and SAN to `IP:<that address>` (here `10.25.73.101` for both). The certificate (365 days) is generated when the container is created and is not in a volume: stop and start keep it, but re-creating the container (a changed `.env` value, a rebuild, `--force-recreate`) makes a new one and the browser warns again (see [Network](#network-security-and-privacy)) |
| `COMPOSE_PROFILES` | empty | `network` starts the coturn relay. The gateway publishes only its web port, so without the relay only a browser on this Docker host can get media through |
| `TURN_HOST`, `TURN_USER`, `TURN_PASSWORD` | empty, `kka`, empty | where browsers reach coturn (the LAN IP), and its single credential. The credential is handed to every browser that loads the page: letters, digits and `. _ ~ -` only, and not one you use elsewhere |
| `GATEWAY_H264_KBPS`, `GATEWAY_H264_MAX_KBPS` | `4000`, `8000` | video bitrate the WebRTC encoder may use; the streams measured 0.9 to 1.4 Mbit/s |
| `GATEWAY_RECONNECT_TAKEOVER` | `1` | a new connection from the same client IP replaces a session that is still registered; `0` is strict (see [One session at a time](#one-session-at-a-time-reconnects-and-memory)) |
| `MODELS_DIR`, `AVATARS_DIR`, `IMAGES_DIR`, `LOOPS_DIR` | `./models` ... | host paths |

### `services/gateway/config/kka.yaml`

Handler option names are checked by pydantic, which **ignores unknown keys silently**: after an edit, rebuild the gateway and check
the `Registered handler ... with config:` lines of `docker logs kka-gateway` to see that the value arrived.

| Setting | Now | Effect |
|---|---|---|
| `logger.log_level` | `INFO` | INFO logs what was said and answered (`ASR: '...'`, `TTS: '...'`) and the timing trace that [merge_timeline.py](../services/gateway/tools/merge_timeline.py) needs. `WARNING` silences both |
| `BargeInVad.end_delay` | `11200` | silence that ends the user's turn, in samples at 16 kHz: 700 ms (it was `4000` = 250 ms in the first run). The longer wait delays every reply by 0.4 to 0.65 s (0.45 s of it is the wait; first sound 2.2 to 2.4 s then, 2.7 to 3.0 s now) and keeps a thinking pause of up to 0.7 s from splitting one question into two. The stock endpointing also re-joins speech that restarts within 0.5 s after the turn ended |
| `BargeInVad.barge_in` | `true` | `false` is half-duplex: the user cannot interrupt. Needed on loudspeakers without echo cancellation |
| `BargeInVad.barge_in_window_ms`, `barge_in_min_speech_ms` | `512`, `256` | the user has to speak 256 ms within 512 ms to interrupt. Lower is quicker but coughs and clicks start to trigger it |
| `FasterWhisper.language`, `beam_size` | `en`, `1` | English only |
| `FasterWhisper.initial_prompt` | HPE product names | biases the transcript towards these terms; add new product names |
| `LLMOpenAICompatible.model_name` | `qwen3-8b` | must equal `LLM_SERVED_NAME` |
| `LLMOpenAICompatible.system_prompt` | two sentences, about 25 words | everything the avatar writes is spoken, so no markdown or lists. The model still sometimes writes more (four sentences for a long question were seen) |
| `KokoroTTS.voice` | `am_michael` | one voice for every avatar. American (`af_*`, `am_*`) or British (`bf_*`, `bm_*`) only, all of one kind; commas blend voices. The other files in `models/Kokoro-82M/voices` need espeak-ng (GPL-3) and are refused |
| `KokoroTTS.device` | `cpu` | on this GPU Kokoro took 1.2 to 3 s for a clause while vLLM decoded; on the CPU 0.5 to 0.65 s |
| `KokoroTTS.pronunciations` | 26 words | see [Pronunciations](#pronunciations) |
| `RtcClient.output_video_fps`, `RendererAvatar.fps` | `25` | must be equal, and equal to the renderer's frame rate |
| `RtcClient.connection_ttl` | `3600` | the longest one conversation can last, in seconds: FastRTC closes the connection after that. Keep it long; a connection that ends earlier is released at once (see [One session at a time](#one-session-at-a-time-reconnects-and-memory)) |
| `chat_engine.concurrent_limit` | `1` | one conversation at a time; the renderer serves one session |

## Avatars

**Choosing a portrait.** One image is enough. Waist-up or full-body framing with visible hands gives the best gestures. The face
is rendered at up to 512 px, so a tight close-up looks soft and a very small face looks worse. Only onboard portraits you have the
right to animate (likeness and consent), especially for a customer-facing delivery.

**Stage A: the speaking loop** (once per portrait, offline). EchoMimicV3-Flash animates the portrait to a speech clip and gives a
3.2 s, 81-frame, 25 fps clip of body, hands and head movement. The live head replaces the face at run time.

```bash
docker compose run --rm bodygen --job aisha=/images/aisha.png --subject "aisha=A woman in a navy blazer" \
    --states speaking --speech /speech/sample_en.wav --sample-area 960
```

* About 2 minutes to load the models, then 320 to 335 s per clip (HD, `--sample-area 960`) with a 25.9 to 26.1 GB peak. Several `--job`
  arguments share the one load. The result is `loops/<name>/speaking.{mp4,json}` and `speaking_sheet.jpg`. Look at the sheet.
* `--speech` is any clip of natural speech of at least 3.3 s (`--speech-offset` picks the start). The gestures follow its rhythm,
  not the words of later replies.
* `--sample-area` caps the clip at that many pixels squared, aspect ratio kept, which gives the frame size. **The default is 768
  (about 560 x 1024 for a portrait photo); the avatars on this host were made with 960**, which gives 704 x 1280 for a portrait, 1280 x
  720 for 16:9 and 1216 x 752 for a wide stage shot, at the same frame rate and render time (see
  [Results on this host](#results-on-this-host)). Pass `--sample-area 960` every time, and use `--overwrite` to replace an existing
  clip. Larger values were not tried and the 960 run already peaks at 26 GB; if the card runs out of memory the job logs it and goes on
  to the next one. `--seed` varies the take. Keep `--teacache 0`: TeaCache is broken in this version.
* Only the `speaking` state ships. Idle, listening and thinking clips shimmer because the generator has no sound to follow, so
  those states are the portrait with the live head.
* While it runs the card is shared with the live stack and both slow down. On a 48 GB card stop `vllm` first (Stage A needs 26 GB
  next to the 8.4 GB of the renderer and the gateway).
* QA numbers are in `speaking.json`: `motion_mae_mean` of 10 to 19 is healthy; `step_mae_max` of 12 or more means fast motion, so
  look at the sheet (`seated-man` has the highest, 13.2, and its sheet is clean); `first_vs_source_mae` of 2 to 6 is normal. Hands
  can still blur or vanish in a fast gesture: `neri` shows it, and in `gulf-man-white-2`, whose hands start at the bottom edge of
  the picture, they disappear while the cuffs flare (its QA numbers are healthy, so the numbers do not catch this): look at every
  new clip, and prefer a portrait that shows the hands well inside the frame.

**Onboarding** registers an avatar with the live renderer (15 to 20 s, needs the engines):

```bash
docker compose run --rm onboard --image /images/aisha.png --name aisha --loop speaking=/loops/aisha/speaking.mp4
```

Without `--loop` the avatar is the portrait with the live head only. With a loop, the clip's size becomes the frame size and pixels
that do not move in the clip are replaced by the photo's (this removes the generator's shimmer; `--no-pin` turns it off).
`--images-dir` registers every image of a folder as a head-only avatar; `--max-size` (default 1280) caps the frame size. A loop
avatar takes about 545 MB on disk, a head-only one 5 to 15 MB. A re-onboarded avatar is reloaded at the next session (the
renderer watches `meta.json`).

**Switching.** `echo neri > run/avatar` selects the avatar of the **next** session; no restart. The renderer holds one avatar, so
the first session on a different avatar spends about 9 s loading it, and speech during that time is lost. Open the page once and
wait about 10 s after switching; later sessions on that avatar are immediate.

### Avatars on this host

| Avatar | Source | Frame (W x H) | Stage A (s / motion mean / step max) |
|---|---|---|---|
| `gulf-man-white` (default) | `Gemini_Generated_Image_n3filtn3filtn3fi.png` | 704 x 1280 | 326 / 18.8 / 7.6 |
| `gulf-man-white-2` | `Gemini_Generated_Image_so828hso828hso82.png` | 704 x 1280 | 321 / 13.9 / 5.3 |
| `gulf-man-black` | `Gemini_Generated_Image_pmz50dpmz50dpmz5.png` | 704 x 1280 | 320 / 10.0 / 6.5 |
| `gulf-man-black-closeup` | `Gemini_Generated_Image_o480yzo480yzo480.png` | 784 x 1168 | 329 / 14.6 / 11.8 |
| `seated-man` | `maxresdefault.jpg` | 1280 x 720 | 335 / 14.6 / 13.2 |
| `neri` | `Neri.png` | 1216 x 752 | 330 / 12.5 / 6.0 |
| `headshot-pn` | `pn.png` (300 x 300) | 300 x 300 | head only, no loop |

The sources are in `images/`. `neri` is a blurry source: it animates, with less detail. `headshot-pn` has no loop (the photo is too
small to animate), so only its face moves. A cartoon source (`Gemini_Generated_Image_uiorgduiorgduior.png`) is also in `images/`
but is **not** an avatar: it was dropped and its loop deleted.

The four `gulf-man-*` avatars come from Gemini images and show Gemini's sparkle watermark in the corner. `seated-man`, `neri` and
`headshot-pn` are photographs of real people: get their consent before showing them to anyone. The UI card is portrait, so the
landscape and square avatars (`seated-man`, `neri`, `headshot-pn`) are shown smaller, with bars around them.

## Measuring latency

Four tools, from narrow to wide. All need a quiet GPU: nobody talking to the avatar, no Stage A run, nothing else on the card.

**1. The renderer alone** (head, compositor, player; no ASR, LLM or TTS). Plays a 10.8 s spoken reply into the renderer and records
what a viewer sees and hears. The audio arrives three times faster than real time (`--tts-speed 3`), as a TTS would deliver it.

```bash
docker run --rm --network host --entrypoint python -v $PWD/work:/work -v $PWD/services/renderer/tools:/tools:ro kka-renderer:latest \
    /tools/ws_client.py --url ws://127.0.0.1:18110/ws --avatar gulf-man-white --audio /tools/samples/reply_clean.wav \
    --out /work/runs/head_test.mp4 --tts-speed 3 --format raw
```

It prints the renderer's `speak_started` and `speak_done` events and a summary: time from the first audio to the first speaking
frame (here 601 ms), frame arrival interval (mean 40.0 ms, p95 42.3 ms), and `distinct_frames_per_s` (here 19.6, for
`gulf-man-white` at 704 x 1280; the saved output is [reports/live_renderer/gulf-man-white.txt](../reports/live_renderer/gulf-man-white.txt)).
`speak_done` holds the frames sent, `repeated` (frames that repeat the previous one because the next was not rendered in time),
`audio_underruns` (must be 0) and `render_ms` (the time to produce a frame; a 25 fps clock allows 40). `--interrupt-after 2.0`
interrupts the reply 2 s after its first frame.

**2. The compositor alone.** [bench_compositor.py](../services/renderer/tools/bench_compositor.py) times what happens to a rendered face before it
is part of a frame, at each avatar's real frame size: the GPU blend into the face rectangle, the read-back to the CPU and the paste
into the body frame. The face is random pixels (the cost does not depend on them), so it needs no engine, only the GPU. The target
is under 5 ms per frame.

```bash
docker run --rm --gpus '"device=0"' --entrypoint python -v $PWD/avatars:/avatars:ro -v $PWD/services/renderer/tools:/tools:ro \
    kka-renderer:latest /tools/bench_compositor.py gulf-man-white seated-man neri --frames 600
```

**3. The whole conversation.** `e2e_client.py` is a headless WebRTC client: it plays recorded questions into the gateway as a
microphone and records the avatar's audio and video as a browser would receive them. `merge_timeline.py` joins its record with the
gateway's trace into a budget per turn.

```bash
T0=$(date -u +%Y-%m-%dT%H:%M:%SZ)
docker run --rm --network kkavatar_default --entrypoint python \
    -v $PWD/services/gateway/tools:/opt/kka-tools:ro -v $PWD/work/e2e:/out kka-gateway:latest \
    /opt/kka-tools/e2e_client.py --url https://kka-gateway:8282 --insecure --out /out/run1 \
    --step say:/opt/kka-tools/samples/q_greenlake.wav --step reply
docker logs --since $T0 kka-gateway 2>&1 | python3 services/gateway/tools/merge_timeline.py work/e2e/run1 | tee work/e2e/run1/timeline.txt
```

`--insecure` accepts the gateway's self-signed certificate (with `GATEWAY_TLS=0` use `http://kka-gateway:8282` and drop it). This
needs `log_level: INFO`. Use `--lead-in 14` for the first session after switching avatars (the session is deaf for about 9 s
then). `--save-frames N`, `--no-mp4`, `--offer-codec`, `--level-db` and `--hold` are described by `--help`. The questions in
`services/gateway/tools/samples/` (`q_greenlake`, `q_servers`, `q_short`, `q_help`, `q_interrupt`, `q_long`) were synthesised by
`bench_asr_tts.py`, which also measures ASR decode time and TTS first-audio time on their own. Steps: `say:<clip>`, `reply[:timeout]`,
`noreply:<s>`, `sleep:<s>`, `say_on_audio:<delay>:<clip>`, `say_after_end:<delay>:<clip>`. A real run (`gulf-man-white`, excerpt):

```
turn 1  q_greenlake.wav (say)
  heard : 'Hi there, can you tell me what HPE GreenLake is?'
  after speech end    step    stage
    +0.000 s      +0.000  you stop speaking
    +1.192 s      +1.192  ASR text ready  (decode 259 ms, includes the VAD end-of-speech wait)
    +1.258 s      +0.066  first LLM token reaches TTS
    +2.063 s      +0.805  first clause synthesised  (TTS 592 ms for the clause)
    +2.077 s      +0.014  renderer receives the first audio
    +2.604 s      +0.527  renderer: first frame ready  (521 ms after its first audio)
    +2.621 s      +0.017  gateway receives the first video frame
    +2.766 s      +0.145  reply sound reaches the client  (WebRTC transport + jitter buffer)
  -> first reply sound 2.77 s after speech end; the longest single step is 'ASR text ready' (1.19 s)
  renderer: 297 frames, 67 repeated (23%), 45.5 ms to render a frame (a 25 fps clock allows 40)
```

The renderer plays sound and picture together: it holds the reply until the first frame is ready, so the sound starts with the
picture. The biggest steps are the end-of-speech wait plus ASR (1.2 s, of which 0.7 s is `end_delay`), the first clause (about
0.8 s: the LLM writes it, then Kokoro on the CPU speaks it) and the first frame (about 0.5 s).

**Barge-in:** play a long question, then speak over the reply 3 s after its sound starts:

```bash
... /opt/kka-tools/e2e_client.py --url https://kka-gateway:8282 --insecure --out /out/bargein \
    --step say:/opt/kka-tools/samples/q_long.wav --step say_on_audio:3:/opt/kka-tools/samples/q_interrupt.wav --step reply
```

The report gets one more line: `barge-in: the gateway cut the reply 0.57 s after you started talking (it waits for 256 ms of speech);
the renderer stopped 40 ms later; the old reply's sound ended at the client 0.79 s after you started talking`. The interrupted turn
reports `the reply was interrupted after N frames`. Lowering `barge_in_min_speech_ms` makes it quicker.

**4. The path a browser on another machine takes** (HTTPS page, TURN relay). Both tests run on this host against this host's LAN
address, because a second machine was not available (see [Not verified](#not-verified)).

* `e2e_client.py` through the relay. `--relay-only` lets no media travel except through TURN, so a pass proves the relay path;
  `transport=tcp` instead of `udp` tests the TCP fallback for networks that block UDP.

  ```bash
  docker run --rm --network host --entrypoint python -e KKA_TURN_PASSWORD="$(sed -n 's/^TURN_PASSWORD=//p' .env)" \
      -v $PWD/services/gateway/tools:/opt/kka-tools:ro -v $PWD/work/e2e:/out kka-gateway:latest \
      /opt/kka-tools/e2e_client.py --url https://10.25.73.101:8282 --insecure --out /out/relay_udp \
      --turn-url "turn:10.25.73.101:3478?transport=udp" --relay-only \
      --step say:/opt/kka-tools/samples/q_greenlake.wav --step reply
  ```

* Headless Chromium: [browser_e2e.py](../services/gateway/tools/browser_e2e.py) drives it with a fake camera and microphone. It
  opens the page, clicks the two buttons, lets the fake microphone say the question and records what Chromium's own WebRTC stack
  reports: frame size and rate, dropped frames, freezes, and when the reply sound starts (judged by audio level, so approximate).
  It runs on the host and takes about 30 s (it stops 3 s after the reply's sound ends; `--seconds`, default 60, is the limit). The
  microphone file needs several seconds of silence before the question, because the fake microphone starts playing the moment the
  page asks for it:

  ```bash
  pip install playwright && playwright install chromium       # once, on the host
  docker run --rm --entrypoint ffmpeg -v $PWD/services/gateway/tools/samples:/s:ro -v /tmp:/out kka-renderer:latest -v error -y \
      -f lavfi -t 9 -i anullsrc=r=16000:cl=mono -i /s/q_greenlake.wav -f lavfi -t 48 -i anullsrc=r=16000:cl=mono \
      -filter_complex "[1:a]aresample=16000[q];[0:a][q][2:a]concat=n=3:v=0:a=1" -ac 1 -c:a pcm_s16le /out/kka_mic.wav
  python3 services/gateway/tools/browser_e2e.py --url https://10.25.73.101:8282 --mic /tmp/kka_mic.wav --out work/browser/run1 --relay-only
  ```

  It writes `summary.json`, `samples.json` and three screenshots, and exits 1 with the reason in `summary.json` when the page does not
  connect or no reply is heard. A refused offer (`concurrency_limit_reached`, which the page itself does not show; see
  [One session at a time](#one-session-at-a-time-reconnects-and-memory)) is reported as such, with the offers, the ICE log and the
  console. Without `--relay-only` the page may pick a direct path, which only works from this host. With it, `webrtc.pair` in
  `summary.json` must show the browser's side with a `relayProtocol` and a port in coturn's relay range (49160-49199), as in the
  last run (`udp`, port 49190, 1 ms round trip to the gateway's address on the compose network).

**Repeated frames.** On this GPU 18 to 23% of frames repeat the previous one: the render stage needs 42 to 46 ms for a frame that a 25 fps
clock allows 40 ms for, so the face updates about 20 times a second while the clock, the sound and the body stay at 25. H.264
re-encodes identical frames slightly differently, so only the renderer's own count (the `renderer:` line) shows it reliably.

**Looking at it.** A contact sheet of a recording (pose, gestures, mouth shapes, seams) without a video player; `--crop x0,y0,x1,y1`
(fractions of the frame) enlarges the head:

```bash
docker run --rm --entrypoint python -v $PWD/work:/work -v $PWD/services/renderer/tools:/tools:ro kka-renderer:latest \
    /tools/contact_sheet.py /work/e2e/run1/session.mp4 /work/e2e/run1/sheet.jpg --frames 12 --start 2 --end 12 --cols 6 --tile-height 360
```

**Lip sync.** `lipsync_probe.py` tracks the mouth with MediaPipe in a recording and correlates its opening with the loudness of the
sound track. Positive lag means the mouth is behind the sound; one video frame is 40 ms.

```bash
docker run --rm --entrypoint python -v $PWD/work:/work -v $PWD/models/ditto-talkinghead:/models/ditto-talkinghead:ro \
    -v $PWD/services/renderer/tools:/tools:ro kka-renderer:latest /tools/lipsync_probe.py /work/runs/head_test.mp4 --json /work/runs/sync.json
```

A mouth follows phonemes, not loudness, so the correlation is only 0.2 to 0.4: read the peak as a lag, not as a score, and distrust it
when `corr_best_elsewhere` is close to `corr_at_peak`. A face under about 128 px needs `--crop`. Delaying the sound of a recording by exactly
200 ms moved the reading by -203 ms (lips) and -201 ms (jaw), so shifts are measured to about 10 ms; the absolute offset against a
click track was not checked. Viewers do not notice a mouth up to 45 ms behind the sound or 125 ms ahead of it (ITU-R BT.1359).
The e2e client does not estimate sync (its loudness-versus-picture estimate scattered over +-400 ms), but its `session.mp4` holds both
tracks as a browser would receive them, so `lipsync_probe.py` on that file gives the figure through the gateway.

### Results on this host

All seven avatars, 2026-11-26, GPU 0 (unlicensed vGPU), HD frames, 700 ms end-of-speech wait, through the full gateway: the question
"Hi there, can you tell me what HPE GreenLake is?", warm, nobody else on the GPU. All were heard correctly; video arrived as H.264 at
25.0 fps with a mean interval of 40.0 ms (p95 41.7 to 49.0 ms, longest gap 58 to 92 ms) after WebRTC.

| Avatar | Frame | ASR text after speech end | First reply sound after speech end | First frame after first audio | Repeated frames | Render per frame | Video |
|---|---|---|---|---|---|---|---|
| `gulf-man-white` | 704 x 1280 | 1.19 s | 2.77 s | 521 ms | 23% | 45.5 ms | 1018 kbit/s |
| `gulf-man-white-2` | 704 x 1280 | 1.18 s | 2.88 s | 511 ms | 22% | 45.1 ms | 1092 kbit/s |
| `gulf-man-black` | 704 x 1280 | 1.18 s | 2.90 s | 487 ms | 23% | 43.7 ms | 1164 kbit/s |
| `gulf-man-black-closeup` | 784 x 1168 | 1.18 s | 2.96 s | 498 ms | 22% | 43.5 ms | 1392 kbit/s |
| `seated-man` | 1280 x 720 | 1.11 s | 2.75 s | 523 ms | 22% | 43.2 ms | 930 kbit/s |
| `neri` | 1216 x 752 | 1.17 s | 2.89 s | 511 ms | 22% | 45.0 ms | 1244 kbit/s |
| `headshot-pn` | 300 x 300 | 1.18 s | 2.68 s | 498 ms | 18% | 42.2 ms | 1146 kbit/s |

The earlier run at 768 px and 250 ms (`reports/earlier_768px_250ms/`) had a first reply sound of 2.18 to 2.38 s, ASR text at 0.68 to
0.78 s and the same repeated frames (18 to 22%) and render time (41 to 44 ms): HD costs no measurable frame rate or render time, and
the later first sound is the longer end-of-speech wait.

* **Renderer alone** (same images, a 10.8 s reply, `reports/live_renderer/`): 19.5 to 20.8 distinct frames per second, first speaking
  frame 0.56 to 0.60 s after the first audio, 42.9 to 45.2 ms per frame, 17 to 22% repeated frames, no audio underruns; the frames
  arrive every 40.0 ms on average (p95 40.2 to 42.4 ms, longest gap 43 to 47 ms).
* **Compositor alone** (target: under 5 ms per frame; `bench_compositor.py`, 600 frames per avatar, `reports/live_renderer/compositor_bench_hd.txt`):
  median 0.2 to 0.8 ms and mean 1.1 to 4.3 ms, so the target is met on average; the p95 is 8.8 to 10.6 ms and 10 to 40% of the frames
  take more than 5 ms, so it is **not met at the tail**. The timings are bimodal: most calls take under 1 ms and about 30% take
  about 9 ms. That looks like the vGPU being time-sliced, but it was not traced, and the live renderer reports no per-stage times, so
  how much of it reaches a live frame is not known.
* **Lip sync at the renderer output** (positive = the mouth is behind the sound; viewers notice more than 45 ms behind or more than
  125 ms ahead, ITU-R BT.1359). Lips / jaw in ms: `gulf-man-white` 36 / 30, `gulf-man-white-2` 34 / 37, `gulf-man-black` 29 / 43,
  `gulf-man-black-closeup` 32 / 44, `seated-man` 37 / 36, `neri` 41 / 39, `headshot-pn` 44 / 50. That is **at the edge** of what is
  noticed, not comfortably inside it, and the 90% intervals are 30 to 110 ms wide, so these are estimates. Through the gateway (one
  run, `gulf-man-white`, measured on the e2e `session.mp4`): lips +7 ms, jaw +16 ms, with a correlation of only 0.18 and 0.14:
  indicative, and no worse than at the renderer.
* **Barge-in** (two runs, a 12.9 s question interrupted 3 s into the reply): the gateway cut the reply 0.57 and 0.64 s after the
  first word (it waited for 256 and 288 ms of speech), the renderer stopped 40 and 28 ms later, and the old sound ended at the client
  0.79 s after the first word in both. The new reply ("Sure, what would you like to ask?") started 2.61 and 2.57 s after the
  interrupting question ended.
* **Long question** (12.9 s, the first turn of those two runs): first reply sound 3.68 and 3.44 s after it; the first frame came
  0.95 and 0.99 s after the first audio.
* **The relay path** (made from this host to its own LAN address, relay only): the aiortc client over the relay
  on UDP and on TCP, `gulf-man-white`: 25.0 fps, mean interval 40.0 ms, p95 44.7 ms, longest gap 81 ms (UDP) and 70 ms (TCP),
  1061 and 1117 kbit/s. Headless Chromium with `--relay-only`, fake microphone, final image: 3 of 3 passes at 704 x 1280 H.264, 24.9
  fps decoded, no dropped frames, no freezes, reply sound 2.5 to 2.65 s after the end of the question (judged by audio level, so
  approximate).
* **Looks:** the contact sheets show sharp faces, varied mouth shapes and gestures through H.264. Hands are the weak spot of the
  generated loops: in `neri` they blur or vanish in fast gestures, and in `gulf-man-white-2` they start at the bottom edge of the
  picture and disappear while the empty cuffs flare. The other avatars' sheets show clean hands.
* **Files:** `reports/live_renderer/` (per avatar: the renderer-only recording, its contact sheet and the output text, which is from a
  repeat run made after the recording the lip-sync was measured on; `compositor_bench_hd.txt`), `reports/e2e_gateway/<avatar>/`
  (`timeline.txt`, `metrics.json`, `session.mp4`, `session_sheet.jpg`) with `bargein/` and `bargein_run2/`, `reports/lipsync/` (one
  JSON per avatar, `validation_gulf-man-white_audio_delayed_200ms.json`, `gateway_hd_gulf-man-white.json`), `reports/stage_a/<avatar>/`
  (QA numbers and sheet of the loop, six avatars), `reports/network/` (the two relay runs), `reports/reconnect/` (`before_fix`,
  `after_fix`, `final_image`), `reports/earlier_768px_250ms/` (the run before HD and the longer wait), `reports/bench_asr_tts.json`,
  `reports/bench_trt.json`, `reports/bench_flashhead_native_nocompile*` (SoulX-FlashHead Pro, which is not used: see the README).

## Pronunciations

Kokoro's G2P (misaki) is dictionary-only here, because its espeak-ng fallback is GPL-3 and is not loaded. A word that is not in the
dictionaries is skipped silently, so product names get a `KokoroTTS.pronunciations` entry in `kka.yaml`: a whole-word,
case-insensitive match, replaced either by a respelling (`GreenLake: "Green Lake"`) or by exact phonemes
(`ProLiant: "[ProLiant](/pɹOlˈIənt/)"`, misaki's American phoneme set). Without an entry ProLiant is spoken as "Pro".

```bash
docker run --rm --entrypoint python -v $PWD/services/gateway:/opt/kka-gw:ro -v $PWD/models/Kokoro-82M:/models/Kokoro-82M:ro \
    -e HF_HUB_OFFLINE=1 -w /opt/oac kka-gateway:latest /opt/kka-gw/tools/check_pronunciations.py \
    "Marhaba! Welcome to HPE GreenLake at Lusail." "A ProLiant server in Msheireb."
```

[check_pronunciations.py](../services/gateway/tools/check_pronunciations.py) takes sentences (or `--file`), applies the live table and prints what the
TTS will say, the phonemes, and any word it would not say; it exits 1 when it finds one. It runs on the CPU. Run it on the
sentences the avatar is likely to say whenever the table or the LLM prompt changes. The phoneme strings are checked against
Kokoro's vocabulary by `services/gateway/tests/test_pronunciations.py`.

**The Arabic names are approximations** (Marhaba, Assalamu Alaikum, Ooredoo, Kahramaa, Lusail, Msheireb, Katara, Khalifa). Kokoro
has no Arabic. A round trip through Whisper recognised "Marhaba, Assalamualaikum" perfectly but heard Msheireb as "Mesherib", Lusail
as "Lucille", Ooredoo as "Iridu" and Kahramaa as "Karamasar". Rare names are a poor test for an ASR, so this proves little; a native
speaker should listen to them and the phonemes should be adjusted.

## Tests

```bash
# gateway: text normalisation, the pronunciation table against Kokoro's vocabulary, the timeline merger, the TURN configuration,
# and the session handling (reconnect takeover, time-limit release, idle trim): 49 tests
docker run --rm --entrypoint sh -e KOKORO_CONFIG=/models/Kokoro-82M/config.json -v $PWD/services/gateway:/opt/kka-gw:ro \
    -v $PWD/models/Kokoro-82M:/models/Kokoro-82M:ro -w /opt/oac kka-gateway:latest \
    -c "pip install -q pytest; python -m pytest -q -p no:cacheprovider /opt/kka-gw/tests"

# renderer: background pinning and avatar reload (6 tests)
docker run --rm --entrypoint sh -v $PWD/services/renderer:/r:ro -w /r kka-renderer:latest \
    -c "pip install -q pytest; python -m pytest -q -p no:cacheprovider tests"
```

The tests need no GPU. What they cannot show is covered by the measurements under [Measuring latency](#measuring-latency): the tools
there are `services/gateway/tools/{e2e_client,browser_e2e,merge_timeline}.py` and, in the renderer image, `ws_client.py`,
`lipsync_probe.py` and `bench_compositor.py`. The others under `services/renderer/tools/`: `test_images.py` (every image of a folder
through the live head path, with `clip.mp4`, `contact.jpg`, `face_strip.jpg`), `check_registration.py` (registration debug images),
`run_head_clip.py`, `bench_trt.py` (TensorRT engines against PyTorch), `bench_graph.py`, `build_engines.py`.

## Network, security and privacy

What this host publishes (`GATEWAY_BIND_ADDR=10.25.73.101`, `COMPOSE_PROFILES=network`):

| Port | What | Listens on |
|---|---|---|
| 8282/tcp | gateway: the web UI and the WebRTC signalling, HTTPS with a self-signed certificate | `10.25.73.101` |
| 3478/udp and 3478/tcp | coturn, the TURN relay that carries the media | `10.25.73.101` |
| 18100/tcp | vLLM (`/v1`, no key) | `127.0.0.1` |
| 18110/tcp | the renderer's WebSocket | `127.0.0.1` |

Nothing else is published.

* **No login anywhere**, by decision, for a trusted network. Anyone who can reach port 8282 can talk to the avatar and use the GPU,
  one session at a time. The renderer's WebSocket and vLLM's API accept anyone who can reach them, which is why they stay on
  `127.0.0.1` (`BIND_ADDR`). Docker publishes ports around the host firewall (ufw), so `GATEWAY_BIND_ADDR` and `BIND_ADDR` are what
  limit where the services listen, not ufw rules. A network that is not trusted needs authentication in front of port 8282, which
  is not built.
* **Microphone and HTTPS.** Browsers allow the microphone only on `localhost` or over HTTPS, so `GATEWAY_TLS=1` makes the gateway
  serve HTTPS. The entrypoint creates a self-signed certificate when the container is created (365 days; CN `GATEWAY_TLS_CN`, SAN
  `localhost`, `127.0.0.1` and `GATEWAY_TLS_SAN`, here `IP:10.25.73.101`). It lives in the container, not in a volume: stop and start
  keep it, re-creating the container (an image rebuild, `--force-recreate`) makes a new one. Every browser shows a certificate
  warning on the first visit, which the user has to accept. Checked: a certificate with the requested CN and SAN, `https://.../`
  redirects to `/ui/index.html`, plain HTTP on that port is refused. For anything beyond a demo, mount a certificate of the
  organisation's CA over `/opt/kka/tls` as `cert.pem` and `key.pem`.
* **Media for browsers on other machines goes through TURN.** WebRTC media is UDP to the gateway container's private address, which
  another machine cannot reach, so the browser sends it to coturn (`kka-coturn`, BSD-3-Clause), which passes it on to the gateway
  over the compose network: browser, `10.25.73.101:3478` (UDP, or TCP for a network that blocks UDP), relay, gateway. In `.env`:
  `COMPOSE_PROFILES=network`, `TURN_HOST` (the address browsers use for this host), `TURN_USER` and `TURN_PASSWORD` (letters,
  digits and `. _ ~ -` only). At start the gateway entrypoint writes `/tmp/kka.yaml` from the baked `kka.yaml` with a `turn_config`
  for `TURN_HOST:3478`, and OpenAvatarChat hands it to every browser as its ICE servers, so nothing about TURN is in the image.
  Without coturn a browser on another machine gets no media; only a browser on the Docker host works.

  The relay is restricted: it relays only to the compose subnet (`allowed-peer-ip`; every other peer is denied), not over TCP, not to
  multicast; the container is read-only, has every capability dropped but `NET_BIND_SERVICE`, `no-new-privileges`, and its image is
  pinned by digest. The relay ports 49160-49199 stay inside the container (the gateway answers the relay's private address), so
  only 3478 is published. The traffic to 3478 is not TLS (`no-tls`); the media it carries is encrypted by WebRTC itself (DTLS-SRTP).
* **The TURN credential is not a secret.** Every browser that loads the page is given it, and OpenAvatarChat prints it once in the
  gateway log at start (`Parsing RTC config`). It decides who may use port 3478; what the relay can reach is limited as above.
  Do not reuse that password anywhere else.
* **Logs contain what was said.** At `INFO` the gateway logs every transcript and reply. Docker keeps them until the container is
  removed, rotating at 20 MB x 5 files per container. `log_level: WARNING` stops that and also the timing trace; use it for
  production if the transcripts are sensitive.
* **Offline at run time.** The models load from disk, vLLM and the gateway run with `HF_HUB_OFFLINE=1`, usage statistics are off, and
  aiortc's built-in fallback to Google's public STUN server (a DNS lookup and a UDP request per session) is switched off in
  `launch.py`. Outbound traffic in general was not audited.

**What was tested.** From this host to its own LAN address `https://10.25.73.101:8282`: the aiortc client through the relay over UDP
and over TCP, and headless Chromium with `--relay-only` (3 of 3 runs; commands under [Measuring latency](#measuring-latency), results
under [Results on this host](#results-on-this-host)).
The Chromium runs ignore certificate errors and use a fake microphone and camera, so these are **untested**: the certificate warning
and the microphone permission prompt in a real browser, a real microphone and real speakers (echo; barge-in without a headset), and a
second machine, Wi-Fi, or a network that blocks UDP. See [Not verified](#not-verified).

## One session at a time, reconnects and memory

The gateway serves one conversation at a time (`concurrent_limit: 1` in `kka.yaml`): the renderer holds one avatar, and the VAD and
the avatar choice are single-session. OpenAvatarChat and FastRTC, as shipped, then behaved badly in two ways, which
[rtc_session.py](../services/gateway/rtc_session.py) puts right. [launch.py](../services/gateway/launch.py) applies it at start;
OpenAvatarChat's own source is not modified.

**Reconnecting.** A browser that leaves without a goodbye (a closed or reloaded tab, a laptop that went to sleep) keeps its session
until aiortc's ICE consent checks give up, 28 to 33 s later. Every offer in that time is answered `concurrency_limit_reached`, which
the web page swallows: clicking start does nothing. Before the fix a run started right after another one failed this way (runs 2, 4
and 6 of a series, `reports/reconnect/before_fix/`: "not connected 25 s after clicking start"); after it 6 of 6 runs in a row
connected (`after_fix/`), and 3 of 3 with the final image (`final_image/`).

* An offer from the same client address as a live session closes that session first (it waits up to 3 s for the teardown), so a
  reload connects at once. The log says `Offer from 10.25.73.101: closing its earlier session <id> (reload or reconnect)`.
  `GATEWAY_RECONNECT_TAKEOVER=0` in `.env` turns this off: first come, first served.
* An offer from another address while a session is live is refused, and the gateway log says `WARNING ... Offer from X refused: the
  session of Y is still active (concurrent_limit=1)`. The page shows nothing; it works again within about 30 s of the first browser
  leaving. `browser_e2e.py` reports it as `the gateway refused the offer: concurrency_limit_reached (another session is still
  active)` and exits 1 (`reports/reconnect/final_image/refused_other_client.json`).
* Two people behind one NAT or proxy share an address, so the second replaces the first.

**Memory.** Left alone, the gateway process grew with every session: after 8 sessions it held more than 7 GB, over 5 GB of it freed
memory that was still retained. Three causes, three fixes:

| Cause | Fix |
|---|---|
| glibc keeps the heap pages it freed (the 2.7 MB video frames, the TTS tensors) in per-thread arenas | `MALLOC_ARENA_MAX=2` (in `compose.yaml`), and `malloc_trim(0)` when idle |
| a finished session leaves reference cycles (its handler contexts) that Python collects only after many new objects | `gc.collect()` when idle |
| FastRTC's `wait_for_time_limit` held every closed peer connection (about 5 MB each) for `connection_ttl`, an hour | it now ends with the connection |

Ten seconds after a session ends, and only if no other connection is live by then, the gateway collects and trims in a worker thread
(0.4 to 0.6 s) and logs `Idle: garbage collected and freed heap pages returned to the OS, RSS 3506 -> 2259 MB (531 ms)`. Expect about
2.8 GB when fresh, 3.1 to 3.6 GB during and just after a session (4.2 to 4.35 GB after several browser sessions in a row), and 2.25
to 2.30 GB once the idle step has run. No `mem_limit` is set: watch `docker stats kka-gateway`; memory that keeps climbing between
conversations is the symptom to report.

**What the gateway patches at run time** (`launch.py`, so that OpenAvatarChat itself stays unmodified):

* the H.264 bitrate limits that its RTC client hard-codes to 0.5 to 2.5 Mbit/s: now 1 / 4 / 8 Mbit/s as minimum / default / maximum
  (`GATEWAY_H264_KBPS`, `GATEWAY_H264_MAX_KBPS`); the video measured 0.9 to 1.4 Mbit/s;
* aiortc's Google STUN fallback is switched off;
* the reconnect takeover, the time-limit release and the idle trim above;
* the interpreter's thread switch interval, 0.5 ms instead of 5 ms (`KKA_GIL_SWITCH_MS`): the TTS needs the GIL back after every
  small torch call.

## Moving to another GPU

Nothing below was run on an L40S or an unthrottled H100.

1. Set `GPU_ID` and `GPU_PCI_BUS` in `.env` (`nvidia-smi --query-gpu=index,pci.bus_id --format=csv`) and run `scripts/gpu_guard.sh`.
2. The first start builds the TensorRT engines for that architecture (20 to 25 minutes). The grid-sample3d plugin is compiled for
   sm_89 and sm_90; an L40S is sm_89.
3. `LLM_GPU_MEM_UTIL` is a fraction of the card: 0.22 of 94 GB is 21 GB. On a 48 GB card start from about 0.40 (estimate) and lower it
   if the renderer or gateway run out of memory.
4. Stage A (26 GB at `--sample-area 960`, 22.7 GB at the default 768) next to the live stack (29.6 GB here) does not fit in 48 GB:
   stop `vllm` first.
5. Run the [quick GPU check](#gpu-placement-and-the-vgpu-licence) and the measurements above. The figures in this runbook are what an
   unlicensed vGPU gives; a healthy GPU should be faster. If the render stage drops below 40 ms per frame the repeats go away and
   `RENDERER_OVERLAP=75` and `KokoroTTS.device: cuda` (rebuild the gateway) are worth trying to shorten the first reply.

## Licence audit

```bash
for s in gateway renderer bodygen; do docker run --rm -i --entrypoint python kka-$s:latest - < scripts/licence_audit.py; done
```

Lists the Python packages of each image with their licences and exits 1 on GPL, AGPL or an unrecognised licence. None were found in
the three images. The weak-copyleft and proprietary items (MPL-2.0 `certifi`, `tqdm`, `orjson`; LGPL `num2words` and `soxr`; NVIDIA's
redistributable CUDA, cuDNN and TensorRT wheels) and the GPL/LGPL programs in the base system (ffmpeg, x264) are listed in the
README for HPE legal review. espeak-ng (GPL-3) is removed from the final images but still sits in the lower layers of the shared base
image until it is rebuilt (`docker build -f docker/Dockerfile.base -t kka-base:cu128 .`, then rebuild the three images).

The audit covers the Python packages of those three images only. The fourth container, coturn (`coturn/coturn:4.19.2-trixie`, pinned
by digest), is BSD-3-Clause: its image ships no licence file, so this was read from the upstream `LICENSE` at the pinned revision on
2026-11-26, and HPE legal should confirm it. Playwright (Apache-2.0) and the Chromium build it downloads are used only by
`browser_e2e.py`, on the host, as a test tool: they are not part of the deployment, and their licences were not audited.

## Troubleshooting

| Symptom | Cause | What to do |
|---|---|---|
| `docker compose up -d` does not return | it waits for vLLM (about 105 s) and then for the renderer, which builds engines on a new GPU model (20 to 25 min) | `docker compose logs -f`; run it in the background |
| renderer `unhealthy`, `/healthz` answers 503 | still loading or building engines | `docker logs kka-renderer`; wait up to 35 minutes on a new GPU model |
| `gl_context ... GPU suport is not available` in the renderer log | MediaPipe falls back to the CPU; harmless | none |
| `gpu_guard: GPU_ID=... is at PCI ...` | the GPU order on the host changed | correct `GPU_ID` in `.env`; never use GPU 1 |
| vLLM refuses to start, "free memory ... less than desired" | something else holds memory on GPU 0 | `nvidia-smi -i 0`; start vLLM first and the rest after it |
| page loads but there is no microphone prompt | not a secure context: plain HTTP on any name but `localhost` | use `https://10.25.73.101:8282` (`GATEWAY_TLS=1`), or `http://localhost:8282` on this host |
| the browser warns about the certificate | it is self-signed | accept it once per browser, or mount the organisation's certificate, see [Network](#network-security-and-privacy) |
| the page connects to nothing, or "start" does nothing, for up to 30 s | another session is still active (a browser that left without a goodbye is noticed after about 30 s) and the page does not show the refusal | wait; `docker logs kka-gateway` shows `Offer from ... refused`. The same machine reloading replaces its own session at once, see [One session at a time](#one-session-at-a-time-reconnects-and-memory) |
| the page loads from another machine but there is no video or sound | the media cannot reach the TURN relay: coturn is not running (`COMPOSE_PROFILES=network`), `TURN_HOST` is not the address the browser uses, or 3478 is blocked | `docker ps` shows `kka-coturn` healthy; `docker logs kka-coturn`; test with `e2e_client.py --relay-only` from that machine |
| `kka-coturn` exits at start | `TURN_HOST`, `TURN_USER` or `TURN_PASSWORD` is missing or holds a character other than letters, digits and `. _ ~ -` | fix `.env`, then `docker compose up -d --no-deps --force-recreate coturn`; if `TURN_*` changed, recreate `gateway` too (it also makes a new certificate) |
| the gateway's memory keeps climbing between conversations | the idle trim did not run, or something holds sessions | look for the `Idle:` lines in `docker logs kka-gateway`; see [Memory](#one-session-at-a-time-reconnects-and-memory) |
| black or frozen video | the renderer is down, loading, or the avatar name is wrong | `curl http://127.0.0.1:18110/healthz`; `cat run/avatar`; the name must be in `avatars` of that reply. An unknown name is ignored with a warning in the gateway log and the default is used |
| speech right after switching avatar is ignored | the new avatar loads for about 9 s while the session starts | wait about 10 s after connecting; the next session is immediate |
| the avatar interrupts itself or answers its own voice | loudspeakers feed back into the microphone with barge-in on | use a headset, or `barge_in: false` in `kka.yaml` and rebuild |
| one long question is answered in pieces | a pause of more than 0.7 s ended the turn | raise `end_delay` (`11200` = 700 ms, `14400` = 900 ms); every reply is delayed by the same amount |
| `asr_skip reason=cancelled_during_decode` in the log | normal: a newer speech segment replaced one that was still being decoded | none |
| replies take about 3 s | expected on this GPU with the 700 ms end-of-speech wait, see [Results](#results-on-this-host) | check that nothing else uses GPU 0 and the licence state |
| 18 to 23% of frames repeat | the render stage takes 42 to 46 ms against a 40 ms budget on the unlicensed vGPU; sound and body stay in step | a faster GPU |
| a word is mispronounced or missing | Kokoro skips words that are not in its dictionaries | add an entry, see [Pronunciations](#pronunciations) |
| answers are too long or contain lists | the model ignored the prompt | tighten `system_prompt`; `normalise` already strips markdown |
| Stage A runs out of memory | its peak is 26 GB at `--sample-area 960` and another process holds memory | stop `vllm` (or the other user) or use a lower `--sample-area` (the default 768 peaked at 22.7 GB) |
| onboarding prints `[fail] <name>` with a face error on a Stage A loop | the generated clip has a frame without a detectable face | regenerate with `--seed 1` (or another) and `--overwrite` |
| the loop's hands move too fast | `step_mae_max` of 12 or more in `speaking.json` | regenerate with another `--seed` and compare the sheets |
| a port is already in use | another service on this host | change it in `.env` |

## Not verified

* **A second machine, and real audio hardware.** Headless Chromium (fake microphone and camera, certificate errors ignored) and an
  aiortc client, both made from this host to its own LAN address and both through the TURN relay, passed. Not exercised: a browser
  on another machine, Wi-Fi, a network that blocks UDP, a person accepting the certificate warning and the microphone permission
  prompt, and a real microphone and real speakers (see [Network](#network-security-and-privacy)).
* **Other GPUs.** An L40S (sm_89) engine build and an unthrottled H100 were not run, so the throttled numbers are a floor and the
  targets of the brief may well be met there.
* **Loudspeakers.** Barge-in was measured with a clean line-in; with speakers it needs echo cancellation or a headset.
* **A reboot of the host** (the containers restart by policy, which was not tried), outbound network traffic, Stage A above
  `--sample-area 960`, and long runs: the memory behaviour was checked over series of sessions, not over days.
* **Scale.** One conversation at a time; the renderer, the VAD configuration and the avatar selection are single-session.
* **Content.** The speaking loop is generic (3.2 s, played back and forth, about every 6.5 s), the TTS is English only, Arabic names
  are approximate, and one voice serves every avatar.
