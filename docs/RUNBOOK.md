# Runbook

Starting, configuring, measuring and fixing the avatar stack. Commands are run from the project root. Everything was run on this
host on 2026-11-26: on GPU 0 until it stopped accepting compute, then on GPU 1 (see
[GPU placement and the vGPU licence](#gpu-placement-and-the-vgpu-licence); the earlier 768 px / 250 ms run of 2026-11-25 is kept in
`reports/earlier_768px_250ms/`); anything that was not run is marked **untested** or listed under [Not verified](#not-verified).

* [What runs where](#what-runs-where)
* [Start, stop, update](#start-stop-update)
* [GPU placement and the vGPU licence](#gpu-placement-and-the-vgpu-licence)
* [Configuration](#configuration)
* [Avatar Studio](#avatar-studio)
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

Five containers (compose project `kkavatar`). The first four use the one GPU named by `GPU_ID` in `.env` (GPU 1 now); coturn does not use a GPU.

| Container | Does | Host port | GPU memory |
|---|---|---|---|
| `kka-vllm` | Qwen3-8B FP8 on vLLM 0.27.1, OpenAI-compatible API, thinking off | `127.0.0.1:18100` | 21.3 GB (8.8 GiB of weights, the rest KV cache; `LLM_GPU_MEM_UTIL`) |
| `kka-renderer` | Ditto head on TensorRT, GPU compositor, 25 fps player. WebSocket `/ws`, `GET /healthz`, and the avatar admin API `/v1` that the Studio and the worker use | `127.0.0.1:18110` | 5.2 GB (one HD avatar loaded; it keeps one avatar in memory) |
| `kka-gateway` | The Avatar Studio web page and WebRTC (OpenAvatarChat), Silero VAD with barge-in, Faster-Whisper (GPU), Kokoro (CPU), the English-only LLM handler, bridge to the renderer | `GATEWAY_BIND_ADDR:8282`, here `10.25.73.101:8282` (HTTPS) | 3.0 GB (Faster-Whisper) |
| `kka-worker` | The motion generator behind "New avatar" in the Studio: EchoMimicV3-Flash (Stage A) makes the idle and speaking clips and has the renderer register the avatar. One job at a time. Port 8300, inside the compose network only | none | none while idle; about 26 GB while a job runs (the models load when a job starts and are freed when it ends) |
| `kka-coturn` | TURN relay (coturn 4.19.2, BSD-3-Clause) for browsers on other machines; only with `COMPOSE_PROFILES=network` | `GATEWAY_BIND_ADDR:3478`, UDP and TCP | none |

The three always-on GPU containers hold 29,457 MiB of the card's 96,256 MiB on GPU 0 (the figures above) and 29,025 MiB on GPU 1
(vLLM 21,295, renderer 4,783, gateway 2,945, just after the start), by `nvidia-smi -i <GPU_ID> --query-compute-apps=pid,used_memory
--format=csv`, idle, 2026-11-26; the GB figures above are the GPU 0 MiB values divided by 1000. The worker holds none until a job runs.

`onboard` and `bodygen` (compose profile `tools`) are the same two steps as one-shot containers, for scripts and for avatars made
without the Studio; they use the same GPU.

| On the host | Holds |
|---|---|
| `models/` | weights, mounted read-only |
| `images/` | portraits for the command-line route (the Studio keeps the photos it is given in `studio/`) |
| `studio/` | the Studio's uploads (`uploads/<avatar>/source.png`) and its job records (`jobs/`); the gateway, the worker and the renderer share it |
| `loops/<name>/` | Stage A output: `idle.mp4` and `speaking.mp4`, each with a `.json` (settings and QA numbers) and a `_sheet.jpg` |
| `avatars/<name>/` | registered avatars the renderer loads: `meta.json`, `source.<ext>`, `thumb.jpg`, `still/`, `idle/`, `speaking/`, and once set `profile.json` (title, description) and `settings.json` (expression, head movement). The renderer writes here |
| `engines/sm<cc>_trt<version>/` | TensorRT engines, built on the first start on a GPU model (here `sm90_trt10.11.0.33`) |
| `run/avatar` | the avatar the next session uses (the Studio writes it when you open an avatar) |
| `third_party/` | pinned sources: OpenAvatarChat, Ditto, EchoMimicV3, SoulX-FlashHead, the grid-sample3d plugin |
| `reports/` | recorded test results (see [Results on this host](#results-on-this-host)) |
| `logs/`, `work/`, `.cache/vllm` | build logs, scratch output of the commands below, vLLM's compile cache |

Everything in this table except `reports/` is fetched, built or produced on the host and is not in the git repository (see
`.gitignore`), and so is `.env` (copy `.env.example`). A fresh clone therefore has no avatar: open the Studio and create one
(see [Avatar Studio](#avatar-studio)).

## Start, stop, update

```bash
scripts/gpu_guard.sh && docker compose up -d   # vLLM first (about 105 s), then the renderer (about 20 s) and the gateway (about 36 s); the worker and coturn start alongside
docker compose ps                              # all "healthy" (coturn only with COMPOSE_PROFILES=network)
docker compose logs -f gateway                 # follow one service
docker compose stop                            # `down` also removes the containers; models, avatars and engines stay on the host
```

`up -d` waits for each health check before it starts the next service, so it does not return for 2.5 to 3 minutes (run it in the
background from a script). Health by hand:

```bash
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:18100/health   # vLLM: 200
curl -s http://127.0.0.1:18110/healthz                                   # renderer: {"status":"ready","avatars":[...],"loaded":[...],"session_active":false,...}
curl -sk https://10.25.73.101:8282/api/health                            # gateway and Studio (-k: self-signed certificate; plain http://127.0.0.1:8282 without GATEWAY_TLS): the renderer, the worker and the free disk
docker inspect kka-worker --format '{{.State.Health.Status}}'            # worker: healthy
```

* **First start on a GPU model the renderer has not seen:** it builds the TensorRT engines into `engines/` (20 to 25 minutes, once),
  answers 503 on `/healthz` meanwhile, and the health check allows 35 minutes. Watch `docker logs -f kka-renderer`.
* **Changing settings:** `.env` changes apply with `docker compose up -d`. `services/gateway/config/kka.yaml`, the gateway handlers,
  the Studio (its Python and its page) and the renderer and worker code are baked into the images:
  `docker compose build gateway renderer worker` (a few minutes, mostly the renderer), then `up -d`. A rebuilt renderer or gateway
  interrupts a running conversation. A rebuilt worker interrupts a running avatar job: it goes back into the queue and starts
  again, and after two interruptions it is shown as failed (Try again).
* **Versions are pinned** (vLLM digest in `compose.yaml`, sources and weights by commit in `scripts/fetch_*`). Do not move to `latest`.
* **After a host reboot** Docker restarts the containers itself (`restart: unless-stopped`) but does not honour the start order.
  This was not tested (the host is shared). If something is unhealthy: `docker compose stop && scripts/gpu_guard.sh && docker compose up -d`.

## GPU placement and the vGPU licence

* Every service gets `device_ids: ["${GPU_ID}"]` in `compose.yaml`, so a container sees one GPU and nothing else.
  `scripts/gpu_guard.sh` refuses to continue unless `GPU_ID` in `.env` is the card whose PCI address is `GPU_PCI_BUS`. The stack ran
  on GPU 0 until that GPU stopped accepting compute (below) and has run on **GPU 1** (`GPU_ID=1`, `GPU_PCI_BUS=00000000:04:00.0`)
  since 2026-11-26, 12:45 UTC; [Moving to another GPU](#moving-to-another-gpu) has the steps.
* Check the pinning (each line must end in `[1]`, the `GPU_ID`) and that the card inside a container is the one you meant. A
  container always sees its one GPU as index 0, so compare the UUID with the host's:

  ```bash
  docker inspect kka-vllm kka-renderer kka-gateway kka-worker --format '{{.Name}} {{range .HostConfig.DeviceRequests}}{{.DeviceIDs}}{{end}}'
  docker exec kka-renderer nvidia-smi --query-gpu=uuid --format=csv,noheader
  nvidia-smi -i 1 --query-gpu=uuid --format=csv,noheader          # the same UUID as the line above
  ```

* **Both GPUs of this VM are unlicensed vGPUs** (`NVIDIA H100L-94C`, NVIDIA Virtual Compute Server):

  ```bash
  nvidia-smi -i 1 -q | grep -i -E "licens|vGPU Software|Product Name"     # License Status : Unlicensed
  ```

  NVIDIA documents that a vGPU for Compute without a licence loses compute performance, down to an idle level, until one is acquired
  ([NVIDIA vGPU for Compute licensing](https://docs.nvidia.com/ai-enterprise/release-7/latest/infra-software/vgpu/licensing.html),
  [NVIDIA AI Enterprise licensing](https://docs.nvidia.com/ai-enterprise/release-8/latest/infra-software/vgpu/licensing.html),
  which gives 20 minutes as the grace period of an unlicensed VM). Fixing that needs a client configuration token from the
  organisation's NVIDIA License System administrator; the stack does nothing about it, and nobody should try to bypass licensing.
  What the two GPUs did on 2026-11-26 (UTC) did not follow the licence state, which was `Unlicensed` on both throughout:
  * GPU 0 ran the stack, throttled, for about two days (the VM booted on 2026-11-24 at 07:37). The quick GPU check below printed
    93 TFLOPS and 0.65 to 0.77 TB/s on it, and a frame took 42 to 46 ms to render.
  * From about 11:13 GPU 0 refused all CUDA work (it worked up to about 10:35). It still does.
  * At 12:39, with three processes of other workloads running on it, GPU 1 refused too: a one-off attempt (with the owner's
    permission) to create a CUDA context on it from the host returned the same error 801.
  * Those three processes were stopped at about 12:41 on the owner's instruction (SIGTERM; they exited within seconds and nothing
    restarted them). At 12:43:33 GPU 1 accepted CUDA contexts, from the host and from a container, and the quick GPU check at 12:44:48
    printed 515 TFLOPS and 3,296 GB/s.
  * The stack was started on GPU 1 at 12:45:44 and was healthy at 12:47:20 (the TensorRT engines were reused: both GPUs are sm90).
    It was checked end to end there while GPU 1 was fast, see [Results on GPU 1](#results-on-gpu-1).
  * **GPU 1 then slowed down to GPU 0's speed.** Its ASR decodes took 50 to 83 ms up to 13:03:39 and 241 to 244 ms at 13:11:36, in
    real conversations, and the end-to-end run at 13:30 matched what GPU 0 gave: 43 to 44 ms to render a frame, a fifth of the
    frames repeated. So it was fast for 20 to 28 minutes (from 12:43:33 until some time between 13:03:39 and 13:11:36) and has been
    slow since, see [Later the same day: GPU 1 slow](#later-the-same-day-gpu-1-slow). The Stage A job that ran on it from 13:14:51 to
    13:29:15 did not cause this: the slowdown began before the job started. GPU 1 still accepts CUDA, and no log shows an error.
  * **Not known:** why GPU 0 ran for two days before it refused; why GPU 1 refused at 12:39 and not at 12:43 (the other processes
    had stopped in between, which is not shown to be the cause; restarting them to find out was not tried, as it could take the
    running stack down); why GPU 1 was fast at first, and why it slowed down between 13:04 and 13:11. The hypervisor is not visible
    from the VM. So **GPU 1 is throttled now, and may refuse as GPU 0 did**. No GPU was reset, no driver reloaded and nothing was
    done to the licence.
* **If a GPU refuses CUDA work again** (as GPU 0 did from 11:13) this is what it looks like:
  * `cuCtxCreate` returns error 801 (`operation not supported`) from the host itself and from every container, with the host's
    550 driver library and with the 570 compatibility library in the images alike, so it is not an image or library problem.
  * The renderer stops at its first CUDA call and `/healthz` reports `status: error` (`CUDA error: operation not supported`). It does
    not retry.
  * The vLLM engine stops with `CUDA error: device doesn't have valid Grid license`, and `restart: unless-stopped` then puts it in a
    restart loop (each attempt fails the same way).
  * The gateway cannot start either, so the Studio page is down as well. Its Faster-Whisper handler loads the model on the GPU and
    fails with `RuntimeError: CUDA failed with error operation not supported` (shown by loading the same model by hand in the
    gateway image; the container log itself prints no error). The container exits a few seconds after it starts, before the
    Whisper, LLM or TTS handler reports that it has loaded, and `restart: unless-stopped` starts it again in a loop.
  * coturn keeps running, and so does the worker: it reports healthy because it uses the GPU only while a job runs (a job would
    fail the same way).

  `docker compose stop` ends the restart loops and removes nothing. The VM has no licence client: `/etc/nvidia/ClientConfigToken/`
  is empty, there is no `/etc/nvidia/gridd.conf`, no `nvidia-gridd` service or process, no token file under `/home`, `/root`,
  `/opt`, `/srv`, `/mnt` or `/tmp`, and none of the system log files (they go back to 2026-10-25) has a `nvidia-gridd` line.
  **The way out** is to license the VM: an NVIDIA License System administrator issues a client configuration token for it (a vGPU for
  Compute licence) and the VM is set up as
  [NVIDIA's guide](https://docs.nvidia.com/vgpu/latest/grid-licensing-user-guide/configuring-nls-licensed-client.html) describes:
  `/etc/nvidia/gridd.conf` (copied from the `gridd.conf.template` that is already there) with `FeatureType="1"`, the token in
  `/etc/nvidia/ClientConfigToken/`, the `nvidia-gridd` service restarted, ports 80 and 443 open to the licence service. This VM has
  `/usr/bin/nvidia-gridd` but no systemd service for it. The other way is a GPU that computes: a licensed vGPU or a physical GPU on
  another VM or host, or the other GPU of this VM if the quick GPU check below passes on it. Then run `docker compose up -d`: vLLM,
  the gateway and the renderer start again, with the avatar and models as they were (a renderer that is already running in the
  `error` state needs `docker compose restart renderer`, because it does not retry a failed start by itself). Nothing in this
  repository can work around a refusal, and nothing here tries to.
* **Quick GPU check** (10 seconds; run it on any new GPU and compare). It creates a CUDA context and loads the GPU, so on the GPU
  the stack is using the avatar may stutter for those seconds. It printed 515 TFLOPS and 3,296 GB/s on idle GPU 1 (2026-11-26,
  12:44 UTC, while it was fast) and 93 TFLOPS and 0.65 to 0.77 TB/s on GPU 0 earlier that day, against ratings of several hundred
  TFLOPS and 2 to 4 TB/s for an H100. It was not run again after GPU 1 slowed down, because that would have opened another CUDA
  context on a GPU that may refuse; the slowdown showed in the stack's own numbers instead. Without a new CUDA context, the ASR
  decode time in the gateway log is a cheap signal (50 to 83 ms while GPU 1 was fast, 225 to 315 ms since; the times in that log are
  UTC):

  ```bash
  docker logs --since 30m kka-gateway 2>&1 | grep -o 'asr_done.*decode_ms=[0-9]*'
  ```

  Change the device index to test the other GPU:

  ```bash
  docker run --rm --gpus '"device=1"' --entrypoint python kka-renderer:latest -c "
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
| `GPU_ID`, `GPU_PCI_BUS` | `1`, `00000000:04:00.0` | the one GPU; checked by `scripts/gpu_guard.sh` (GPU 0 is `00000000:03:00.0`) |
| `BIND_ADDR` | `127.0.0.1` | address the vLLM and renderer ports are published on (and the gateway's, unless `GATEWAY_BIND_ADDR` is set). Those two services have no login: leave them on `127.0.0.1` |
| `GATEWAY_BIND_ADDR` | empty (= `BIND_ADDR`) | address of the gateway web UI (the Studio) and the TURN relay only. Set to this host's LAN IP (here `10.25.73.101`) to serve other machines; talking to an avatar needs no login, and `STUDIO_KEY` can protect changes (see [Network](#network-security-and-privacy)) |
| `GATEWAY_PORT`, `LLM_PORT`, `RENDERER_PORT` | `8282`, `18100`, `18110` | host ports; pick free ones (`ss -ltn`). Other services on this host use, among others, 5432, 6333, 8001, 8060, 8888, 9000 to 9106 (in part), 11434, 19530 and 27017 |
| `LLM_GPU_MEM_UTIL`, `LLM_MAX_MODEL_LEN` | `0.22`, `8192` | vLLM's share of the card's memory and its context length |
| `GATEWAY_AVATAR` | `gulf-man-white` (compose's fallback when empty: the name of an avatar a fresh clone does not have) | seeds `run/avatar` the first time only; afterwards edit that file |
| `RENDERER_PRELOAD` | empty (`gulf-man-black` on this host) | avatar loaded at boot; empty loads none and each avatar loads when a session first asks for it. A name that is not in the library (an avatar deleted in the Studio, say) is skipped with a warning in the renderer's log. The renderer keeps one avatar in memory |
| `RENDERER_OVERLAP` | `70` | `70` = about 1.0 s of speech before the first frame; `75` = about 0.4 s at about twice the motion cost. Left at 70, the value measured on GPU 0, where the render stage was over budget. While GPU 1 was fast it rendered a frame in 9 ms, so `75` would have been worth trying; GPU 1 has since slowed to GPU 0's 43 ms, and `75` costs about twice the motion work, so it only makes sense on a fast GPU (**untested**) |
| `RENDERER_STEPS` | `10` | sampling steps of Ditto's motion model; not re-tuned |
| `GATEWAY_TLS`, `GATEWAY_TLS_CN`, `GATEWAY_TLS_SAN` | `0`, `localhost`, empty | HTTPS with a self-signed certificate. Browsers allow the microphone only on https or `localhost`, so another machine needs `GATEWAY_TLS=1`, CN set to the address it opens and SAN to `IP:<that address>` (here `10.25.73.101` for both). The certificate (365 days) is generated when the container is created and is not in a volume: stop and start keep it, but re-creating the container (a changed `.env` value, a rebuild, `--force-recreate`) makes a new one and the browser warns again (see [Network](#network-security-and-privacy)) |
| `COMPOSE_PROFILES` | empty | `network` starts the coturn relay. The gateway publishes only its web port, so without the relay only a browser on this Docker host can get media through |
| `TURN_HOST`, `TURN_USER`, `TURN_PASSWORD` | empty, `kka`, empty | where browsers reach coturn (the LAN IP), and its single credential. The credential is handed to every browser that loads the page: letters, digits and `. _ ~ -` only, and not one you use elsewhere |
| `GATEWAY_H264_KBPS`, `GATEWAY_H264_MAX_KBPS` | `4000`, `8000` | video bitrate the WebRTC encoder may use; the streams measured 0.9 to 1.4 Mbit/s |
| `GATEWAY_RECONNECT_TAKEOVER` | `1` | a new connection from the same client IP replaces a session that is still registered; `0` is strict (see [One session at a time](#one-session-at-a-time-reconnects-and-memory)) |
| `MODELS_DIR`, `AVATARS_DIR`, `IMAGES_DIR`, `LOOPS_DIR` | `./models` ... | host paths |
| `STUDIO_DIR` | `./studio` | host path of the Studio's uploads and job records, shared by the gateway, the worker and the renderer |
| `STUDIO_KEY` | empty | when set, creating, changing and deleting avatars in the Studio needs this key (the page asks for it once per browser tab and sends it as `X-Studio-Key`); empty means no key. Not a login: opening an avatar and talking to it never asks for it. Use letters and digits only (it travels in an HTTP header) and treat it like a password |

### `services/gateway/config/kka.yaml`

Handler option names are checked by pydantic, which **ignores unknown keys silently**: after an edit, rebuild the gateway and check
the `Registered handler ... with config:` lines of `docker logs kka-gateway` to see that the value arrived.

| Setting | Now | Effect |
|---|---|---|
| `logger.log_level` | `INFO` | INFO logs what was said and answered (`ASR: '...'`, `TTS: '...'`) and the timing trace that [merge_timeline.py](../services/gateway/tools/merge_timeline.py) needs. `WARNING` silences both |
| `BargeInVad.end_delay` | `11200` | silence that ends the user's turn, in samples at 16 kHz: 700 ms (it was `4000` = 250 ms in the first run). The longer wait delays every reply by 0.4 to 0.65 s (0.45 s of it is the wait; first sound 2.2 to 2.4 s then, 2.7 to 3.0 s with the new wait on GPU 0 and 2.1 to 2.3 s on GPU 1 while it was fast, 2.8 to 3.0 s after it slowed down) and keeps a thinking pause of up to 0.7 s from splitting one question into two. The stock endpointing also re-joins speech that restarts within 0.5 s after the turn ended |
| `BargeInVad.barge_in` | `true` | `false` is half-duplex: the user cannot interrupt. Needed on loudspeakers without echo cancellation |
| `BargeInVad.barge_in_window_ms`, `barge_in_min_speech_ms` | `512`, `256` | the user has to speak 256 ms within 512 ms to interrupt. Lower is quicker but coughs and clicks start to trigger it |
| `FasterWhisper.language`, `beam_size` | `en`, `1` | English only |
| `FasterWhisper.initial_prompt` | HPE product names | biases the transcript towards these terms; add new product names |
| `LLMOpenAICompatible.module` | `kka/llm/english_llm` | the stock OpenAI-compatible handler plus an English-only filter ([english_llm.py](../services/gateway/handlers/kka/llm/english_llm.py)): Chinese, Japanese and Korean characters are removed from the streamed text, and a reply that had nothing else, or a failed request, is answered with a short English sentence. It also closes a race in the stock handler: the voice-activity handler cancels a reply and restarts the turn whenever the person speaks again (it does so at every pause), but the LLM handler learns of the cancel from a signal that arrives some tens of milliseconds later, and a token sent in between opened a new stream that the next reply then joined, so a caption began with a stray word (` toSure! ...`) and the TTS probably spoke it (not listened for). The wrapper reads the reply stream's status before each chunk and ends the handler's use of a cancelled one. `llm/openai_compatible/llm_handler_openai_compatible` is the stock handler |
| `LLMOpenAICompatible.model_name` | `qwen3-8b` | must equal `LLM_SERVED_NAME` |
| `LLMOpenAICompatible.system_prompt` | two sentences, about 25 words | everything the avatar writes is spoken, so no markdown or lists. The model still sometimes writes more (four sentences for a long question were seen) |
| `KokoroTTS.voice` | `am_michael` | one voice for every avatar. American (`af_*`, `am_*`) or British (`bf_*`, `bm_*`) only, all of one kind; commas blend voices. The other files in `models/Kokoro-82M/voices` need espeak-ng (GPL-3) and are refused |
| `KokoroTTS.device` | `cpu` | on GPU 0 (throttled) Kokoro took 1.2 to 3 s for a clause while vLLM decoded; on the CPU 0.5 to 0.65 s. Not tried on GPU 1 (**untested**) |
| `KokoroTTS.pronunciations` | 26 words | see [Pronunciations](#pronunciations) |
| `RtcClient.output_video_fps`, `RendererAvatar.fps` | `25` | must be equal, and equal to the renderer's frame rate |
| `RtcClient.connection_ttl` | `3600` | the longest one conversation can last, in seconds: FastRTC closes the connection after that. Keep it long; a connection that ends earlier is released at once (see [One session at a time](#one-session-at-a-time-reconnects-and-memory)) |
| `chat_engine.concurrent_limit` | `1` | one conversation at a time; the renderer serves one session |

## Avatar Studio

The gateway serves the Studio, an English web page in a dark HPE-style theme. Its pages, styles, scripts and icons are files of this
repository (no CDN, web font or script from elsewhere; the content security policy allows only the page's own origin). Open
`https://10.25.73.101:8282` here (the gateway's address, `GATEWAY_BIND_ADDR`, and `GATEWAY_PORT`).

* **Library** (`/`). One card per avatar: thumbnail, name, description, and a badge for what it has: *Full motion* (idle and
  speaking body clips), *Partial motion* (one of them) or *Quick preview* (the photo and the live face only). A search box; the
  avatar that is *on stage* (the one the next conversation uses); a card per avatar that is being made, with its stage, progress,
  time left and *Cancel*, *Try again* and *Remove*; and the state of the services (renderer, language model, motion generator, free
  disk). Cards have *Open*, *Copy link*, *Settings* and *Delete*, and *Add full motion* when the avatar lacks the clips.
* **Stage page** (`/a/<avatar>`, a link that works for anyone on the network). *Start conversation* connects microphone and video,
  there is a box for typing instead, *Interrupt* stops the avatar, and the transcript is shown. Opening the page puts the avatar on
  stage (it writes `run/avatar`) and loads it into the renderer: 13 to 15 s the first time, and the page says so.
* **Settings** (a card or the stage page): name, description, *Facial expression* (0.2 to 1.2, default 0.6: lower is calmer lips,
  jaw and face) and *Head movement* (0 to 1, default 0.5; it only matters for an avatar without a speaking clip, because with one
  the head follows the generated body). Stored in `avatars/<name>/settings.json` and `profile.json`. A conversation that is running
  keeps its look; the next one loads the new values (the same 13 to 15 s).
* **New avatar.** Upload one photo (PNG, JPEG or WebP, up to 20 MB, short side at least 256 px, long side at most four times the
  short side; above 3072 px it is scaled down and above 60 megapixels it is refused), give a name (empty: the file name) and a
  short English description of the person (clothing and setting; it goes into the motion model's prompt). Chinese, Japanese and
  Korean text in either is refused. Then choose:

  | Choice | What the job does | Estimate shown by the page |
  |---|---|---|
  | **Full motion**, HD (default) | Stage A makes the idle and speaking clips at `--sample-area 960` (704 x 1280 for a portrait), then the renderer registers the avatar | about 13 min |
  | **Full motion**, Standard | the same at `--sample-area 768` (about 560 x 1024) | about 8 min |
  | **Quick preview** | no clips: the renderer registers the photo and the live face | about 1 min |

  A quick preview becomes a full avatar with *Add full motion* (the same quality choice); it stays usable while that runs. The
  estimates are fixed numbers in the page (`ESTIMATE_S` in `static/js/dialogs.js`), not measured live.
  The job card counts down from the worker's own estimate, which adds fixed figures per stage (`QUALITY`, `LOAD_S`, `STILL_S` and
  `LOOP_S` in `services/bodygen/worker.py`): 910 s, about 15 min, at HD and 580 s, about 10 min, at Standard. The one HD job that
  has run through the Studio on GPU 1 took 14.4 min, see below.

**What a job does.** Jobs run one at a time, in order; the page can be closed and the job goes on. The stages are: waiting in the
queue; waiting for GPU memory (HD needs 30 GB free, Standard 26 GB, and the live stack holds 29.5 GB of the 96 GB card, so it
normally starts at once; it waits up to 6 hours); loading the motion model (about 2 min); the idle clip and the speaking clip (about
5.5 min each at HD, 2.75 min at Standard); waiting for the renderer (it registers one avatar at a time and not during a
conversation; the job waits up to 6 hours); and preparing the live face (about 1 min). Making motion uses the GPU the conversations
use, so a conversation during a job is slower. Those times are the worker's estimates (and the first measurements, on GPU 0). The
one job that has run through the Studio on GPU 1 (`newavatar`, full motion, HD, 81 frames per clip, 13:14:51 to 13:29:15 UTC)
took 154 s to load the model, 322 s for the idle clip, 318 s for the speaking clip and 62 s to prepare the live face: 863 s in
all, one attempt, no error. During it the generator process used about 100 % of one CPU core while the GPU showed 0 to 8 %
utilization (two windows of 12 and 20 samples of the driver's counter); whether the clips are limited by the CPU or by the slowed
GPU (see [GPU placement](#gpu-placement-and-the-vgpu-licence)) was not profiled.

* **Cancel** works at every stage except the last minute, when the renderer is registering the avatar. **Try again** (failed or
  cancelled jobs) keeps the clips that were finished, unless the quality changes. **Remove** clears the card. A job that was running
  when the worker restarted is queued again, and fails after being interrupted twice. The records are `studio/jobs/*.json`.
* A failed job says why in one sentence and what to do (the GPU stayed busy, the renderer refused the avatar, the photo is gone).
* **Delete** removes the avatar's folder in `avatars/`, its clips in `loops/<name>/` and its upload in `studio/uploads/<name>/`.
  It is refused while a job for that avatar runs and while a conversation is going on, and it cannot be undone: the clips take
  minutes to make again. The avatar's id (its address) comes from its name: lower-case letters, digits and hyphens.
* The Studio **cannot** register an existing avatar's head again. That is the command-line route under
  [Registering an avatar again](#registering-an-avatar-again).

**Access key.** With `STUDIO_KEY` set (empty here), creating, changing and deleting avatars needs it: the page asks once per browser
tab (it is kept in that tab only) and sends it as the `X-Studio-Key` header. Ten wrong keys within a minute from one client address
lock that address out for a minute (HTTP 429). Browsers also refuse to let another website drive these actions: a state-changing
request that carries a foreign `Origin` or `Sec-Fetch-Site` gets 403. Opening an avatar and talking to it never needs the key. It
is not a login: see [Network, security and privacy](#network-security-and-privacy).

**The browser never names a path.** It sends the photo and the choices; the Studio decodes the photo, writes it again as a new PNG
under `studio/uploads/<avatar>/`, and tells the worker and the renderer where it is. The renderer's admin API accepts image and clip
paths only below `/studio` and `/loops`. For scripts the routes are (JSON, same origin):

| Route | Does |
|---|---|
| `GET /api/health`, `POST /api/unlock` | state of the services and the limits; checks a key |
| `GET /api/avatars`, `GET /api/avatars/{id}`, `GET /api/avatars/{id}/thumbnail` | the library |
| `POST /api/avatars` | create: multipart `photo`, `name`, `description`, `mode` (`full` or `quick`), `quality` (`hd` or `standard`); answers 202 with the job, 507 below 4 GiB of free disk |
| `POST /api/avatars/{id}/upgrade` | add full motion to a quick or partial avatar; 409 when it has both clips or its photo is gone |
| `PATCH /api/avatars/{id}`, `PUT /api/avatars/{id}/settings` | name and description; `expression` and `head_motion` |
| `DELETE /api/avatars/{id}` | delete; 409 while a job for it is active |
| `GET /api/stage`, `POST /api/stage` | the avatar on stage |
| `POST /api/avatars/{id}/warm`, `GET /api/avatars/{id}/status` | load an avatar into the renderer, and ask whether it is loaded |
| `GET /api/jobs`, `POST /api/jobs/{id}/cancel`, `POST /api/jobs/{id}/retry`, `DELETE /api/jobs/{id}` | the jobs |

## Avatars

An avatar is a folder `avatars/<name>/` that the renderer loads: the photo, the face data of the photo (`still/`) and, for an avatar
with motion, the face data of two body clips, `idle/` (breathing and posture while nobody is speaking) and `speaking/` (hand and body
gestures while it talks). The renderer plays the idle clip between replies and the speaking clip during one, and draws the live head
on both. The Studio makes avatars (above); the commands below do the same two steps by hand.

**Choosing a portrait.** One image is enough. Waist-up or full-body framing with visible hands gives the best gestures. The face
is rendered at up to 512 px, so a tight close-up looks soft and a very small face looks worse. Only onboard portraits you have the
right to animate (likeness and consent), especially for a customer-facing delivery.

**Stage A: the body clips** (once per portrait, offline). EchoMimicV3-Flash animates the portrait and gives two 3.2 s, 81-frame,
25 fps clips of body, hands and head movement: `idle`, driven by silence (breathing and a little sway; its last frame is pinned to
the photo, so it loops) and `speaking`, driven by a clip of natural speech (hand and body gestures at a speaking cadence; its last
frame is free, which allows larger gestures). The live head replaces the face at run time.

```bash
docker compose run --rm bodygen --job aisha=/images/aisha.png --subject "aisha=A woman in a navy blazer" \
    --states idle,speaking --speech /speech/sample_en.wav --sample-area 960
```

* About 2 minutes to load the models, then 318 to 337 s per clip (HD, `--sample-area 960`) with a 25.9 to 26.1 GB peak, two clips
  per avatar. Several `--job` arguments share the one load. The result is `loops/<name>/{idle,speaking}.{mp4,json}` and
  `{idle,speaking}_sheet.jpg`. Look at the sheets.
* `--speech` is any clip of natural speech of at least 3.3 s (`--speech-offset` picks the start). The gestures follow its rhythm,
  not the words of later replies.
* `--sample-area` caps the clip at that many pixels squared, aspect ratio kept, which gives the frame size. **The default is 768
  (about 560 x 1024 for a portrait photo); the avatars on this host were made with 960**, which gives 704 x 1280 for a portrait, 1280 x
  720 for 16:9 and 1216 x 752 for a wide stage shot, at the same frame rate and render time (see
  [Results on this host](#results-on-this-host)). Pass `--sample-area 960` every time, and use `--overwrite` to replace an existing
  clip. Larger values were not tried and the 960 run already peaks at 26 GB; if the card runs out of memory the job logs it and goes on
  to the next one. `--seed` varies the take. Keep `--teacache 0`: TeaCache is broken in this version.
* The renderer draws the live head on the `idle` clip too, so a calm body breathes and the eyes blink between replies, instead of a
  frozen photo with a face on it. `listening` and `thinking` play the idle clip unless the avatar has clips of its own (`--states`
  accepts them; none was generated). An avatar without clips (a quick preview) is the still photo with the live head, and its
  *Head movement* setting applies.
* While it runs the card is shared with the live stack and both slow down. On a 48 GB card stop `vllm` first (Stage A needs 26 GB
  next to the 8.2 GB of the renderer and the gateway).
* QA numbers are in `idle.json` and `speaking.json`. For `speaking`: `motion_mae_mean` of 10 to 19 is healthy; `step_mae_max` of 12
  or more means fast motion, so look at the sheet (`seated-man` has the highest, 13.2, and its sheet is clean); `first_vs_source_mae`
  of 2 to 6 is normal. The six `idle` clips here have `motion_mae_mean` 2.3 to 6.6 and `step_mae_max` 1.5 to 5.0: a much larger
  value means the body moves more than a calm idle should, so look at its sheet. Hands can still blur or vanish in a fast gesture:
  `neri` shows it, and in `gulf-man-white-2`, whose hands start at the bottom edge of the picture, they disappear while the cuffs
  flare (its QA numbers are healthy, so the numbers do not catch this): look at every new clip, and prefer a portrait that shows
  the hands well inside the frame. Five of the six clip avatars made here, `neri` and `gulf-man-white-2` among them, were deleted
  later and their clips with them (see [Avatars made on this host](#avatars-made-on-this-host)); their QA numbers and sheets are
  kept in `reports/stage_a/`.

**Onboarding** registers an avatar with the live renderer (about 1 minute for a photo and two clips, needs the engines; the Studio
does this itself):

```bash
docker compose run --rm onboard --image /images/aisha.png --name aisha \
    --loop idle=/loops/aisha/idle.mp4 --loop speaking=/loops/aisha/speaking.mp4
```

Without `--loop` the avatar is the portrait with the live head only. With clips, the clips' size becomes the frame size and pixels
that do not move in a clip are replaced by the photo's (this removes the generator's shimmer; `--no-pin` turns it off). Registering
renders every clip frame once with the live head, to match the head's colour to the body and to place it on each frame's real face
(see [Registering an avatar again](#registering-an-avatar-again) and `services/renderer/PATCHES.md`). `--add-loop state=clip` adds
one clip to an avatar that exists, `--images-dir` registers every image of a folder as a head-only avatar, and `--max-size`
(default 1280) caps the frame size. An avatar with two clips takes 440 to 450 MB on disk, a head-only one 5 to 15 MB. A
re-onboarded avatar is reloaded at the next session (the renderer watches `meta.json`).

**Switching.** `echo gulf-man-black > run/avatar` selects the avatar of the **next** session; no restart. The renderer holds one avatar, so
the first session on a different avatar spends about 13 s loading it (it renders the idle clip once), and speech during that time is
lost. The Studio's stage page does the switching itself: it selects the avatar, has the renderer load it and shows *Preparing* until
it is ready, so nothing is lost; later sessions on that avatar are immediate.

### Registering an avatar again

An avatar's head data is registered once and stored (`loop.json` has the cache `format`, `info.pkl` an `aligned` flag), so an avatar
keeps the behaviour of the renderer that registered it. The renderer loads old ones and says so in its log (`docker logs kka-renderer`):

| Cache format | Head on the clip | Log line when loaded |
|---|---|---|
| 4 (now) | placed on the real face of every clip frame at registration, so it keeps the size and place of the face under it | none |
| 3 | placed from the estimated pose of each frame, which makes it swell, slide and wobble a little (the reported problem) | `... registered by an older renderer (cache format 3, now 4): its head can wobble: it is not placed on the template's real face frame by frame` |
| 1, 2 | older still | the same line, ending `show the template's mouth` (1) or `swell when the template's mouth opens wide` (2) |

`gulf-man-black` and `newavatar`, the two avatars in the library now, are format 4 (their expressions are 0.8 and, since 13:33 on
2026-11-26, 1.2; `newavatar` was created through the Studio at 0.6). An avatar registered by an older build
(format 3 or below) keeps the behaviour of that build. To give it the placement, register it again from its photo and its clips, which
must still exist: deleting an avatar in the Studio deletes its clips. The Studio has no button for this, so use the command line, into
a staging folder, and swap the result in when no conversation is running:

```bash
curl -s http://127.0.0.1:18110/healthz        # "session_active": false
mkdir -p work/staging
docker compose run --rm -v $PWD/work/staging:/staging onboard --avatars /staging --image /images/<photo> --name <name> \
    --loop idle=/loops/<name>/idle.mp4 --loop speaking=/loops/<name>/speaking.mp4        # about 1 minute
for f in thumb.jpg settings.json profile.json; do [ -f avatars/<name>/$f ] && cp -p avatars/<name>/$f work/staging/<name>/; done
mv avatars/<name> work/<name>.old && mv work/staging/<name> avatars/<name>    # work/<name>.old is the backup
```

* `<photo>` is the image the clips were made from (`image` in `loops/<name>/idle.json`; the files are in `images/`). The photo is
  fitted to the clips' frame size, so a photo with another aspect ratio than the clips makes the registration fail with a "no face"
  error.
* The renderer sees that `meta.json` changed and loads the new folder at the next session; nothing is restarted. The copied
  `settings.json` keeps the avatar's sliders (`gulf-man-black`: `{"expression": 0.8, "head_motion": 0.5}`); an avatar without one
  uses the defaults, 0.6 and 0.5. `thumb.jpg` and `profile.json` (the Studio's name and description) are not written by `onboard`,
  hence the copy. Do not leave a copy of an avatar folder inside `avatars/`: every folder there that holds a `meta.json` is listed
  as an avatar.
* Keep `onboard`'s defaults, `--overlap 70 --steps 10` (the renderer's own). With Ditto's defaults (10 and 50) only about 30 of a
  clip's 81 silent frames come out, the placement cannot be measured, and `onboard` prints `head not placed on the template's real
  face (...)`: the avatar is saved but plays as a format 3 avatar (`aligned` false), and the renderer says nothing when it loads
  it. Read the command's output for that line.
* The avatar registered this way, `gulf-man-black`: idle clip 81 of 81 frames fitted, scale spread 0.39%, anchor error 2.0 px
  before and 0.8 px after; speaking clip 81 of 81, scale spread 1.5%, anchor error 4.0 px before and 2.6 px after (see
  `services/renderer/PATCHES.md`, "Head placement on the real face", for what this changed in the picture).

### Avatars made on this host

Seven avatars were made and measured here. **Six were deleted through the Studio on 2026-11-26, after those measurements, so
`gulf-man-black` is the only one in the library now** (it is the avatar on stage). The Studio's delete removes an avatar's folder
in `avatars/` and its clips in `loops/`, so the clips of the six are gone; their photos are still in `images/`. The table is what
was made and what the measurements in this runbook were taken on. Each clip has 81 frames (3.24 s at 25 fps). Stage A column:
generation time in seconds, `motion_mae_mean` and `step_mae_max` (the numbers in each `loops/<name>/<state>.json`; the speaking
clip is the livelier one, and for the idle clip a low motion is the aim).

| Avatar | Source | Frame (W x H) | In the library now | Idle clip (s / motion mean / step max) | Speaking clip (s / motion mean / step max) |
|---|---|---|---|---|---|
| `gulf-man-white` | `Gemini_Generated_Image_n3filtn3filtn3fi.png` | 704 x 1280 | deleted | 327 / 4.4 / 3.1 | 326 / 18.8 / 7.6 |
| `gulf-man-white-2` | `Gemini_Generated_Image_so828hso828hso82.png` | 704 x 1280 | deleted | 318 / 4.2 / 3.0 | 321 / 13.9 / 5.3 |
| `gulf-man-black` | `Gemini_Generated_Image_pmz50dpmz50dpmz5.png` | 704 x 1280 | **yes**: cache format 4, expression 0.8 | 323 / 4.3 / 3.4 | 320 / 10.0 / 6.5 |
| `gulf-man-black-closeup` | `Gemini_Generated_Image_o480yzo480yzo480.png` | 784 x 1168 | deleted | 329 / 6.1 / 5.0 | 329 / 14.6 / 11.8 |
| `seated-man` | `maxresdefault.jpg` | 1280 x 720 | deleted | 337 / 6.6 / 4.9 | 335 / 14.6 / 13.2 |
| `neri` | `Neri.png` | 1216 x 752 | deleted | 325 / 2.3 / 1.5 | 330 / 12.5 / 6.0 |
| `headshot-pn` | `pn.png` (300 x 300) | 300 x 300 | deleted (head only, it had no clips) | - | - |

`neri` was a blurry source: it animated with less detail, and in the photo the mouth is a little open. `headshot-pn` had no clips
(the photo is too small to animate), so only its face moved. A cartoon source (`Gemini_Generated_Image_uiorgduiorgduior.png`) is
also in `images/` but was **not** made an avatar: it was dropped and its loop deleted.

Bringing one of the six back means making its clips again (the Studio's *Create*, about 13 minutes at HD, or `docker compose run
bodygen`). It is then registered with the head placement of format 4, as `gulf-man-black` was. Anyone who can reach the Studio can
delete avatars while `STUDIO_KEY` is empty (see [Network](#network-security-and-privacy)), and a deleted avatar cannot be restored
from the Studio. The photo of `gulf-man-black` is a smile with teeth, and the head takes its resting face from it, so that
avatar's mouth never closes fully or purses; that is the price of keeping its own smile.

The four `gulf-man-*` avatars come from Gemini images and show Gemini's sparkle watermark in the corner. `seated-man`, `neri` and
`headshot-pn` are photographs of real people: get their consent before showing them to anyone (their test recordings and contact
sheets are in `reports/`). The Studio's library shows every avatar in a portrait (4:5) window, from the top centre, so a landscape
or square avatar (like the deleted `seated-man` and `neri`) is cropped there; the stage page uses the avatar's own proportions.

## Measuring latency

Four tools, from narrow to wide. All need a quiet GPU: nobody talking to the avatar, no Stage A run, nothing else on the card.

**1. The renderer alone** (head, compositor, player; no ASR, LLM or TTS). Plays a 10.8 s spoken reply into the renderer and records
what a viewer sees and hears. The audio arrives three times faster than real time (`--tts-speed 3`), as a TTS would deliver it.

```bash
docker run --rm --network host --entrypoint python -v $PWD/work:/work -v $PWD/services/renderer/tools:/tools:ro kka-renderer:latest \
    /tools/ws_client.py --url ws://127.0.0.1:18110/ws --avatar gulf-man-black --audio /tools/samples/reply_clean.wav \
    --out /work/runs/head_test.mp4 --tts-speed 3 --format raw
```

It prints the renderer's `speak_started` and `speak_done` events and a summary: time from the first audio to the first speaking
frame (601 ms in the saved run), frame arrival interval (mean 40.0 ms, p95 42.3 ms), and `distinct_frames_per_s` (19.6 there; that run
was of the first build on `gulf-man-white` at 704 x 1280, and its output is saved in
[reports/live_renderer/gulf-man-white.txt](../reports/live_renderer/gulf-man-white.txt)).
`speak_done` holds the frames sent, `repeated` (frames that repeat the previous one because the next was not rendered in time),
`audio_underruns` (must be 0) and `render_ms` (the time to produce a frame; a 25 fps clock allows 40). `--interrupt-after 2.0`
interrupts the reply 2 s after its first frame.

**2. The compositor alone.** [bench_compositor.py](../services/renderer/tools/bench_compositor.py) times what happens to a rendered face before it
is part of a frame, at each avatar's real frame size: the GPU blend into the face rectangle, the read-back to the CPU and the paste
into the body frame. The face is random pixels (the cost does not depend on them), so it needs no engine, only the GPU. The target
is under 5 ms per frame.

```bash
docker run --rm --gpus '"device=1"' --entrypoint python -v $PWD/avatars:/avatars:ro -v $PWD/services/renderer/tools:/tools:ro \
    kka-renderer:latest /tools/bench_compositor.py gulf-man-black --frames 600
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
needs `log_level: INFO`. Use `--lead-in 18` for the first session after switching avatars (the renderer loads the avatar first, about
13 s here, and the session is deaf meanwhile). `--save-frames N`, `--no-mp4`, `--offer-codec`, `--level-db` and `--hold` are described by `--help`. The questions in
`services/gateway/tools/samples/` (`q_greenlake`, `q_servers`, `q_short`, `q_help`, `q_interrupt`, `q_long`) were synthesised by
`bench_asr_tts.py`, which also measures ASR decode time and TTS first-audio time on their own. Steps: `say:<clip>`, `reply[:timeout]`,
`noreply:<s>`, `sleep:<s>`, `say_on_audio:<delay>:<clip>`, `say_after_end:<delay>:<clip>`. A real run (the first build, on
`gulf-man-white`, excerpt):

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

**Repeated frames.** On GPU 0, which was throttled, 18 to 23% of frames repeated the previous one: the render stage needed 42 to 46 ms
for a frame that a 25 fps clock allows 40 ms for, so the face updated about 20 times a second while the clock, the sound and the
body stayed at 25. On GPU 1, while it was fast, the render stage needed 8.5 to 9.7 ms and no frame repeated; after it slowed down it needed
43 to 44 ms again and 19 to 22% of the frames repeated (13:30 UTC, see [Later the same day](#later-the-same-day-gpu-1-slow)). H.264 re-encodes identical frames slightly
differently, so only the renderer's own count (the `renderer:` line) shows it reliably.

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

**Head steadiness.** The head is a separate picture pasted on the body clip, so it can swell, slide or turn against the body (the
wobble). `motion_probe.py` reports the mouth opening, the head size, the speed of the face against its surroundings, the body motion
and every change of state; `wobble_probe.py` compares the live face's landmarks with those of the template frame it was pasted on,
frame by frame. Both read a renderer-alone recording made with `--meta` (the state of every frame) and silent frames around the reply
(`--lead-in 4 --tail 4`), and run in the renderer image, which has MediaPipe (its `gl_context` errors on stderr are harmless: it runs
on the CPU). The avatar must be loaded already, so that the recording starts at once:

```bash
docker run --rm --network host --entrypoint python -v $PWD/work/runs:/work -v $PWD/services/renderer/tools:/tools:ro kka-renderer:latest \
    /tools/ws_client.py --url ws://127.0.0.1:18110/ws --avatar gulf-man-black --audio /tools/samples/reply_clean.wav \
    --out /work/reply.mp4 --meta /work/reply.json --tts-speed 3 --lead-in 4 --tail 4 --format raw
probe() { docker run --rm --entrypoint python -v $PWD/work/runs:/work -v $PWD/avatars:/avatars:ro \
    -v $PWD/services/renderer/tools:/tools:ro -v $PWD/models/ditto-talkinghead:/models/ditto-talkinghead:ro kka-renderer:latest "$@"; }
probe /tools/motion_probe.py /work/reply.mp4 --meta /work/reply.json --json /work/motion.json
probe /tools/wobble_probe.py /work/reply.mp4 --meta /work/reply.json --template /avatars/gulf-man-black/speaking/frames.npy \
    --json /work/wobble.json
```

The probes take about 40 s and 12 s. Read the face width against the template's first (`width_pct` in `wobble_probe.py`: `std`, and
`hf_p95` for the fast part): it is the head swelling and shrinking on its body. `head_size_pulse_pct` in `motion_probe.py` (the
distance between the outer eye corners against its own median over +-0.3 s) is not a wobble figure on its own, because blinks move
those points too: the template's own pulse over the frames played in these recordings is 3 to 4% at p95 and up to 8% at the maximum,
and the first build, which copies the template's face, shows the same. The groups `eyes`, `nose`, `brows` and `cheeks` are points
that the mouth does not move; about 1.5 px or less at `hf_p95` is steady. One reply of `gulf-man-black`, the 10.8 s one above (the
first three columns were recorded side by side on one renderer GPU, the last on the live renderer):

| | first build (`eff36fd`) | second build, no placement, expression 0.4 | placed, expression 0.4 | placed, expression 0.8 (live) |
|---|---|---|---|---|
| face width against the template's, std / HF95, % | 0.72 / 1.29 | 1.86 / 1.90 | 0.74 / 1.21 | 1.10 / 1.84 |
| eyes / nose / brows / cheeks, HF95, px | 0.65 / 1.58 / 0.95 / 0.86 | 1.07 / 1.71 / 2.03 / 1.66 | 0.69 / 1.38 / 1.19 / 1.24 | 0.76 / 1.66 / 1.05 / 1.38 |
| head-size pulse, p95 / max, % | 3.33 / 7.25 | 2.74 / 3.79 | 1.22 / 3.21 | 1.27 / 2.68 |
| jaw opening (`jawOpen`), p95 | 0.649 | 0.244 | 0.204 | 0.321 |

The first build's head kept its place and size on the template's face, but its mouth opened wide (the template's talking mouth plus
the audio's); the second build had a calm mouth and a head that swelled and slid (width std 1.86%, brows and cheeks at 2.0 and 1.7
px); the placed head keeps the calm mouth and comes close to the first build's steadiness (a little less of it at 0.8). The face
width includes the jaw line, so it
rises with the expression: in a sweep of the slider on one renderer the jaw p95 was 0.20, 0.26, 0.33 and 0.36 at 0.4, 0.6, 0.8 and
1.0 (the live recording above: 0.32 at 0.8) and the width std 0.74, 0.87, 1.02 and 1.22%. Frames that repeat because the renderer was
late raise every figure of every recording alike, so compare recordings made on the same GPU in the same state. The full table is in
`services/renderer/PATCHES.md`, "Head placement on the real face".

### Results on this host

All seven avatars made here at the time (six of them were deleted from the library afterwards, see
[Avatars made on this host](#avatars-made-on-this-host)), 2026-11-26, GPU 0 (unlicensed vGPU), HD frames, 700 ms end-of-speech wait,
through the full gateway: the question
"Hi there, can you tell me what HPE GreenLake is?", warm, nobody else on the GPU. All were heard correctly; video arrived as H.264 at
25.0 fps with a mean interval of 40.0 ms (p95 41.7 to 49.0 ms, longest gap 58 to 92 ms) after WebRTC. These runs are of the first build
(`eff36fd`), which had the speaking clip only: they predate the idle clips, the neutral rest face and the head placement. The bullet
"Current build" below repeats the renderer-alone and browser checks for `gulf-man-black`.

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
* **Current build, `gulf-man-black`** (idle and speaking clips, head placed, expression 0.8, the live renderer; the same 10.8 s reply
  as in the renderer-alone run): the first speaking frame came 574 ms after the first audio, 312 frames were sent, 66 of them repeated
  (21%, about 19.7 distinct frames per second), 43.0 ms to render a frame, no audio underruns: within the ranges of the first build,
  as the placement is computed when the avatar is registered, not per frame. Headless Chromium through the gateway
  (`https://10.25.73.101:8282/a/gulf-man-black`, direct path, not relay-only): 704 x 1280 H.264, 24.9 fps decoded during the reply, no
  dropped frames, no freezes. The page's captions show its question in two turns (a shorter one first), so its 5.05 s from the end of
  the speech to the reply sound is not comparable with the 2.5 to 2.65 s above. The other six avatars were not measured on the
  current build (they were deleted from the library afterwards); the steadiness of the head is under **Head steadiness** above.
* **Looks:** the contact sheets show sharp faces, varied mouth shapes and gestures through H.264. Hands are the weak spot of the
  generated loops: in `neri` they blur or vanish in fast gestures, and in `gulf-man-white-2` they start at the bottom edge of the
  picture and disappear while the empty cuffs flare. The other avatars' sheets show clean hands.
* **Files:** `reports/live_renderer/` (per avatar: the renderer-only recording, its contact sheet and the output text, which is from a
  repeat run made after the recording the lip-sync was measured on; `compositor_bench_hd.txt`), `reports/e2e_gateway/<avatar>/`
  (`timeline.txt`, `metrics.json`, `session.mp4`, `session_sheet.jpg`) with `bargein/` and `bargein_run2/`, `reports/lipsync/` (one
  JSON per avatar, `validation_gulf-man-white_audio_delayed_200ms.json`, `gateway_hd_gulf-man-white.json`), `reports/stage_a/<avatar>/`
  (QA numbers and sheet of the loop, six avatars), `reports/network/` (the two relay runs), `reports/reconnect/` (`before_fix`,
  `after_fix`, `final_image`), `reports/earlier_768px_250ms/` (the run before HD and the longer wait), `reports/bench_asr_tts.json`,
  `reports/bench_trt.json`, `reports/bench_flashhead_native_nocompile*` (SoulX-FlashHead Pro, which is not used: see the README),
  `reports/gpu1_spot_check/` (`timeline.txt` and `metrics.json` of the fast-period run below; no recording) and
  `reports/gpu1_slowdown/` (`timeline.txt` and `metrics.json` of the run at 13:30, and `asr_decode_since_1245.txt`; no recording).

### Results on GPU 1

2026-11-26, 12:48 to 13:05 UTC, **while GPU 1 was fast** (it slowed down soon after, see
[Later the same day: GPU 1 slow](#later-the-same-day-gpu-1-slow)): the current build (idle and speaking clips, head placed,
expression 0.8, the end-of-reply and cancel fixes) on `gulf-man-black`, then the only avatar in the library, with nobody else on the
GPU. Three questions through the full
gateway with `e2e_client.py` (item 3 of [Measuring latency](#measuring-latency)), 700 ms end-of-speech wait, warm. The GPU 0
column is the first build (`eff36fd`, the table above), so the two differ in the build as well as the GPU.

| | GPU 1 (3 questions) | GPU 0 (7 avatars) |
|---|---|---|
| Quick GPU check | 515 TFLOPS, 3,296 GB/s | 93 TFLOPS, 0.65-0.77 TB/s |
| Render per frame, compositing included | 8.8-9.7 ms | 42-46 ms |
| Repeated frames | 0 of 767 | 18-23% |
| ASR decode | 53-73 ms | 248-281 ms |
| ASR text after the end of speech (includes the 0.7 s wait) | 0.88-1.02 s | 1.11-1.19 s |
| First clause's speech synthesis (Kokoro, CPU) | 0.74-0.87 s | 0.53-0.77 s |
| First frame after the first audio | 106-136 ms | 487-523 ms |
| First reply sound after the end of speech | 2.07-2.25 s | 2.68-2.96 s |
| Video at the client | 25.0 fps, mean interval 40.0 ms, p95 44.2 ms, longest gap 77 ms | 25.0 fps, mean 40.0 ms, p95 41.7-49.0 ms, longest gap 58-92 ms |

Against the brief, in this fast period: 25 fps with no repeated frame and a render stage at 9 ms of the 40 ms a frame may take is **met**; ASR under
120 ms is **met** (the 0.7 s wait comes on top); TTS first audio under 150 ms is **not met**: Kokoro runs on the CPU and needs
0.74 to 0.87 s for the first clause (0.53 to 0.77 s in the GPU 0 runs; the replies differ from run to run and the CPUs are shared
with the other workloads of this host, so the difference was not investigated), and Kokoro on the GPU, which was too slow next to
vLLM on GPU 0, was not tried on GPU 1; the whole turn under 500 ms is **not met**, at 2.1 to 2.3 s, of which 0.7 s is the
end-of-speech wait and about 0.8 s the first clause's speech.
**Not measured on GPU 1:** lip-sync offset, barge-in timing and the compositor alone. Stage A ran once, later, and `newavatar`
did not exist yet (see [Avatar Studio](#avatar-studio)).

* **Browser** (headless Chromium, `browser_e2e.py`, direct path, before and after the cancel fix): connected in 150 and 235 ms, 704 x
  1280 H.264 at 25.0 and 24.9 fps decoded, no dropped frames, no freezes, no console errors. Its `reply_sound_after_speech_end_s`
  is not comparable between runs (2.85 s, then 4.35 s): `q_help.wav` has a 1.1 s pause after "summary", the tool takes the end of
  the first segment as the end of the speech, and whether the first reply answers the question's first half or the whole varies
  from run to run (the VAD restarts the turn when speech resumes: `Early VAD end` / `POST_END` in the gateway log).
* **Cancel fix.** Before it, a capture of the page's data channel (a throw-away script, not in the repository) showed a word of a
  cancelled reply at the start of the next caption in one of the three avatar streams (` toSure! I can help ...`), and the first
  browser run's captions had ` moreSure!` and ` assistSure!`. After it, none of 10 avatar streams in four captures started with a
  fragment (one capture had five restarted turns), the gateway log has no error, and the three-question budget above is the same as
  before the fix (first reply sound 2.05 to 2.21 s, then 2.07 to 2.25 s). The unit tests cover the guard (117 gateway tests).

### Later the same day: GPU 1 slow

GPU 1 did not stay at the speed above. The ASR decode times in the gateway log
(`reports/gpu1_slowdown/asr_decode_since_1245.txt`: all 30 decodes from 12:59:31, when the gateway container was re-created and its
log starts, to 13:34:39) were 50 to 83 ms for the 19 decodes up to 13:03:39 and 225 to 315 ms for the 11 decodes from 13:11:36 on
(no decode was logged in between). So GPU 1 was fast for 20 to 28 minutes after it first accepted CUDA at 12:43:33, and
slowed down between 13:03:39 and 13:11:36. That is before the Stage A job began (13:14:51), so the job did not cause it, and the
run at 13:30, after the job had ended, shows the slow numbers.

At 13:30 UTC, with only this stack on GPU 1 (vLLM, renderer and gateway; about 29 GB), the same three questions were run again through
the gateway on `gulf-man-black`, the avatar of the fast run, with `e2e_client.py`, the same 700 ms wait, warm
(`reports/gpu1_slowdown/`; the raw run with its recording is in `work/e2e/gpu1_c/` on this host, not in the repository):

| | GPU 1 slow, 13:30 (3 questions) | GPU 1 fast (above) | GPU 0 (7 avatars) |
|---|---|---|---|
| Render per frame, compositing included | 42.8-44.3 ms | 8.8-9.7 ms | 42-46 ms |
| Repeated frames | 179 of 840 (19-22%) | 0 of 767 | 18-23% |
| ASR decode | 233-289 ms | 53-73 ms | 248-281 ms |
| ASR text after the end of speech (includes the 0.7 s wait) | 1.09-1.28 s | 0.88-1.02 s | 1.11-1.19 s |
| First clause's speech synthesis (Kokoro, CPU) | 0.65-0.84 s | 0.74-0.87 s | 0.53-0.77 s |
| First frame after the first audio | 463-527 ms | 106-136 ms | 487-523 ms |
| First reply sound after the end of speech | 2.79-3.02 s | 2.07-2.25 s | 2.68-2.96 s |
| Video at the client | 25.0 fps, mean interval 40.0 ms, p95 43.7 ms, longest gap 60 ms | 25.0 fps, mean 40.0 ms, p95 44.2 ms, longest gap 77 ms | 25.0 fps, mean 40.0 ms, p95 41.7-49.0 ms, longest gap 58-92 ms |

* This is GPU 0's profile: the render stage is over its 40 ms budget again, so a fifth of the frames are repeats of the one before
  them (the video still arrives at 25 fps), and the first frame comes about half a second after the first audio. The brief's
  25 fps without repeated frames was met on GPU 1 only in the fast period.
* Both GPUs report `License Status : Unlicensed`. GPU 1 still accepts CUDA, the stack's logs since 13:00 show no error (checked at
  about 13:35), and no other process is on it. Why it was fast at first and why it slowed down is not known, and the quick GPU
  check was not run again, so as not to open another CUDA context on a GPU that may refuse (see
  [GPU placement](#gpu-placement-and-the-vgpu-licence)).
* The way to a GPU that stays fast is the one under [GPU placement](#gpu-placement-and-the-vgpu-licence): a licence for the vGPU, or
  a licensed GPU.

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
# gateway: text normalisation, the pronunciation table against Kokoro's vocabulary, the timeline merger, the TURN configuration, the
# session handling (reconnect takeover, time-limit release, idle trim), the English-only LLM guard and the Studio backend
# (settings, photos, access key, routes): 117 tests
docker run --rm --entrypoint sh -e KOKORO_CONFIG=/models/Kokoro-82M/config.json -v $PWD/services/gateway:/opt/kka-gw:ro \
    -v $PWD/models/Kokoro-82M:/models/Kokoro-82M:ro -w /opt/oac kka-gateway:latest \
    -c "pip install -q pytest; python -m pytest -q -p no:cacheprovider /opt/kka-gw/tests"

# renderer: background pinning, the steady crop, avatar storage, settings and reload, the admin API, the end-of-reply walk and
# blink check, the rest-face options, the head placement (calibrate.py) and the avatars loaded at boot: 57 tests
docker run --rm --entrypoint sh -e PYTHONPATH=/r:/r/ditto -v $PWD/services/renderer:/r:ro -w /r kka-renderer:latest \
    -c "pip install -q pytest httpx; python -m pytest -q -p no:cacheprovider tests" 2>&1 | grep -v "^W0000\|^I0000"

# generation worker: progress parsing, job validation and flow against a fake generator and renderer, cancel, retry, restart
# recovery and the HTTP API: 15 tests
docker run --rm --entrypoint sh -v $PWD/services/bodygen:/b:ro -w /b kka-bodygen:latest \
    -c "pip install -q pytest; python -m pytest -q -p no:cacheprovider tests"
```

The tests need no GPU (the admin API's tests fake the registrar; the worker's use `tests/fake_generator.py` instead of the motion
model) and install pytest into a throw-away container, so they need the network for that one `pip install`. What they cannot
show is covered by the measurements under [Measuring latency](#measuring-latency): the tools there are
`services/gateway/tools/{e2e_client,browser_e2e,merge_timeline}.py` and, in the renderer image, `ws_client.py`, `lipsync_probe.py`,
`motion_probe.py`, `wobble_probe.py` and `bench_compositor.py`. The others under `services/renderer/tools/`: `onboard.py` (see
[Avatars](#avatars)), `test_images.py` (every image of a folder through the live head path, with `clip.mp4`, `contact.jpg`,
`face_strip.jpg`), `check_registration.py` (registration debug images), `run_head_clip.py`, `bench_trt.py` (TensorRT engines against
PyTorch), `bench_graph.py`, `build_engines.py`.

## Network, security and privacy

What this host publishes (`GATEWAY_BIND_ADDR=10.25.73.101`, `COMPOSE_PROFILES=network`):

| Port | What | Listens on |
|---|---|---|
| 8282/tcp | gateway: the Studio (library, avatar pages, creation API), the WebRTC signalling, HTTPS with a self-signed certificate | `10.25.73.101` |
| 3478/udp and 3478/tcp | coturn, the TURN relay that carries the media | `10.25.73.101` |
| 18100/tcp | vLLM (`/v1`, no key) | `127.0.0.1` |
| 18110/tcp | the renderer's WebSocket | `127.0.0.1` |

Nothing else is published.

* **No login anywhere**, by decision, for a trusted network. Anyone who can reach port 8282 can talk to the avatar and use the GPU,
  one session at a time. The renderer's WebSocket and vLLM's API accept anyone who can reach them, which is why they stay on
  `127.0.0.1` (`BIND_ADDR`). Docker publishes ports around the host firewall (ufw), so `GATEWAY_BIND_ADDR` and `BIND_ADDR` are what
  limit where the services listen, not ufw rules. A network that is not trusted needs authentication in front of port 8282, which
  is not built.
* **The Studio can change things, not only talk.** Anyone who can reach port 8282 can also upload a photo (up to 20 MiB), start a
  generation job that holds the GPU for minutes (the page estimates 13 for a full HD avatar), rename avatars and delete them.
  `STUDIO_KEY` in `.env` puts a key in front of those actions only; it is **not set on this host**, so they are open to everyone who
  can talk to the avatar. The key is a shared secret checked by the gateway and sent in a header (so use it over HTTPS), not an
  account; the details and the other protections (rate limit, refusal of cross-site requests) are under
  [Avatar Studio](#avatar-studio). An uploaded photo stays on this host (`studio/uploads/<avatar>/` and the avatar's
  `source.png`) until the avatar is deleted in the Studio.
* **Microphone and HTTPS.** Browsers allow the microphone only on `localhost` or over HTTPS, so `GATEWAY_TLS=1` makes the gateway
  serve HTTPS. The entrypoint creates a self-signed certificate when the container is created (365 days; CN `GATEWAY_TLS_CN`, SAN
  `localhost`, `127.0.0.1` and `GATEWAY_TLS_SAN`, here `IP:10.25.73.101`). It lives in the container, not in a volume: stop and start
  keep it, re-creating the container (an image rebuild, `--force-recreate`) makes a new one. Every browser shows a certificate
  warning on the first visit, which the user has to accept. Checked: a certificate with the requested CN and SAN, `https://.../`
  serves the Studio (`/ui/index.html` is the same page), plain HTTP on that port is refused. For anything beyond a demo, mount a
  certificate of the organisation's CA over `/opt/kka/tls` as `cert.pem` and `key.pem`.
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

The move from GPU 0 to GPU 1 of this host was done on 2026-11-26 (steps 1 and 2 below, 12:45 to 12:47 UTC, then the measurements of
[Results on GPU 1](#results-on-gpu-1)); GPU 1 slowed down to GPU 0's speed within the hour
([Later the same day](#later-the-same-day-gpu-1-slow)). Nothing below was run on an L40S or an H100 outside this VM.

1. Set `GPU_ID` and `GPU_PCI_BUS` in `.env` (`nvidia-smi --query-gpu=index,pci.bus_id --format=csv`) and run `scripts/gpu_guard.sh`.
   `docker compose up -d` then re-creates `vllm`, `renderer`, `gateway` and `worker` on the new GPU (no image build is needed).
   Re-creating the worker ends a Studio job that is running, so check first that `active` is 0:

   ```bash
   docker exec kka-worker python3 -c "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:8300/healthz').read().decode())"
   # {"status": "ok", "active": 0, "current": null}
   ```

   The gateway gets a new self-signed certificate, so browsers warn again, and the renderer loads its avatar again (about 13 s).
2. The first start on a GPU architecture builds the TensorRT engines (20 to 25 minutes); an architecture that already has
   `engines/sm<cc>_trt<version>/` reuses them (GPU 0 to GPU 1: both sm90, so the stack was healthy 95 s after `up -d`). The
   grid-sample3d plugin is compiled for sm_89 and sm_90; an L40S is sm_89.
3. `LLM_GPU_MEM_UTIL` is a fraction of the card: 0.22 of 94 GB is 21 GB. On a 48 GB card start from about 0.40 (estimate) and lower it
   if the renderer or gateway run out of memory.
4. Stage A (26 GB at `--sample-area 960`, 22.7 GB at the default 768) next to the live stack (28.8 GB on GPU 0: 29,457 MiB; 28.3 GB on
   GPU 1: 29,025 MiB) does not fit in 48 GB: stop `vllm` first. The Studio's worker does not stop anything: it waits until the card
   has 30 GB free (HD) or 26 GB (Standard) and fails the job after 6 hours, so on a 48 GB card stop `vllm` before creating a
   full-motion avatar there.
5. Run the [quick GPU check](#gpu-placement-and-the-vgpu-licence) and the measurements above, and repeat them after a while: GPU 0
   gave 93 TFLOPS, a 42 to 46 ms render stage and repeated frames; GPU 1 gave 515 TFLOPS, 9 ms and none for its first 20 to 28
   minutes and then slowed to GPU 0's numbers, so what the same card does can differ a great deal, and can change within the hour.
   With the render stage far below 40 ms per frame, `RENDERER_OVERLAP=75` and `KokoroTTS.device: cuda` (rebuild the gateway) are
   worth trying to shorten the first reply; neither was tried on GPU 1.

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
| the renderer's `/healthz` says `status: error` with `CUDA error: operation not supported`, vLLM dies with `device doesn't have valid Grid license`, or the quick GPU check fails the same way | the vGPU has no NVIDIA licence and the driver refuses CUDA work, see [GPU placement and the vGPU licence](#gpu-placement-and-the-vgpu-licence) | `docker compose stop` (ends the restart loops); have the VM licensed (or use a GPU that computes); then `docker compose up -d` and, for a renderer left in `error`, `docker compose restart renderer`. State of the licence: `nvidia-smi -i <GPU_ID> -q \| grep -i -E "licens\|Product Name"` |
| `gl_context ... GPU suport is not available` in the renderer log | MediaPipe falls back to the CPU; harmless | none |
| `gpu_guard: GPU_ID=... is at PCI ...` | the GPU order on the host changed | correct `GPU_ID` in `.env` so that it is the GPU at `GPU_PCI_BUS` (`nvidia-smi --query-gpu=index,pci.bus_id --format=csv`) |
| vLLM refuses to start, "free memory ... less than desired" | something else holds memory on the GPU | `nvidia-smi -i <GPU_ID>`; start vLLM first and the rest after it |
| page loads but there is no microphone prompt | not a secure context: plain HTTP on any name but `localhost` | use `https://10.25.73.101:8282` (`GATEWAY_TLS=1`), or `http://localhost:8282` on this host |
| the browser warns about the certificate | it is self-signed | accept it once per browser, or mount the organisation's certificate, see [Network](#network-security-and-privacy) |
| the page connects to nothing, or "start" does nothing, for up to 30 s | another session is still active (a browser that left without a goodbye is noticed after about 30 s) and the page does not show the refusal | wait; `docker logs kka-gateway` shows `Offer from ... refused`. The same machine reloading replaces its own session at once, see [One session at a time](#one-session-at-a-time-reconnects-and-memory) |
| the page loads from another machine but there is no video or sound | the media cannot reach the TURN relay: coturn is not running (`COMPOSE_PROFILES=network`), `TURN_HOST` is not the address the browser uses, or 3478 is blocked | `docker ps` shows `kka-coturn` healthy; `docker logs kka-coturn`; test with `e2e_client.py --relay-only` from that machine |
| `kka-coturn` exits at start | `TURN_HOST`, `TURN_USER` or `TURN_PASSWORD` is missing or holds a character other than letters, digits and `. _ ~ -` | fix `.env`, then `docker compose up -d --no-deps --force-recreate coturn`; if `TURN_*` changed, recreate `gateway` too (it also makes a new certificate) |
| the gateway's memory keeps climbing between conversations | the idle trim did not run, or something holds sessions | look for the `Idle:` lines in `docker logs kka-gateway`; see [Memory](#one-session-at-a-time-reconnects-and-memory) |
| black or frozen video | the renderer is down, loading, or the avatar name is wrong | `curl http://127.0.0.1:18110/healthz`; `cat run/avatar`; the name must be in `avatars` of that reply. An unknown name is ignored with a warning in the gateway log and the default is used |
| speech right after switching avatar is ignored | the new avatar loads for about 13 s while the session starts (the Studio's stage page waits for this itself and shows *Preparing*) | wait about 15 s after connecting; the next session is immediate |
| the head swells, slides or wobbles while the avatar talks | the avatar is cache format 3 or older: the renderer's log says `registered by an older renderer (cache format 3, now 4): its head can wobble ...` when it loads it. A high *Expression* also opens the jaw wider and moves the jaw line | register the avatar again, see [Registering an avatar again](#registering-an-avatar-again) (here only `gulf-man-black` is format 4); lower *Expression* in the avatar's Settings |
| `onboard` prints `head not placed on the template's real face (...)` | fewer than half of the clip's silent frames could be fitted, usually because the engine ran with Ditto's defaults (overlap 10, steps 50) instead of 70 and 10; the avatar is saved but is format 3 in effect | register it again without `--overlap` / `--steps` (or with `KKA_OVERLAP=70`, `KKA_STEPS=10`) |
| a Studio job stays at *Waiting for GPU memory (N of 30 GB free)* | the card has less free memory than the job needs (30 GB for HD, 26 for Standard); the live stack takes 29 GB of 94 | free memory on the GPU (`nvidia-smi -i <GPU_ID>`, stop what else holds it); the job goes on by itself and fails after 6 hours |
| the Studio answers 401 or 429 on create, change or delete | `STUDIO_KEY` is set and the key is missing or wrong (401), or ten wrong keys within a minute locked that address out for a minute (429) | enter the key from `.env` (the page asks once per tab); wait a minute |
| the avatar interrupts itself or answers its own voice | loudspeakers feed back into the microphone with barge-in on | use a headset, or `barge_in: false` in `kka.yaml` and rebuild |
| one long question is answered in pieces | a pause of more than 0.7 s ended the turn | raise `end_delay` (`11200` = 700 ms, `14400` = 900 ms); every reply is delayed by the same amount |
| `asr_skip reason=cancelled_during_decode` in the log | normal: a newer speech segment replaced one that was still being decoded | none |
| replies take 2 to 3 s | expected with the 700 ms end-of-speech wait: first sound 2.1 to 2.3 s on GPU 1 while it was fast, and 2.7 to 3.0 s on GPU 0 and on GPU 1 since it slowed down, see [Results on GPU 1](#results-on-gpu-1) and [Later the same day](#later-the-same-day-gpu-1-slow) | check that nothing else uses the GPU (`nvidia-smi -i <GPU_ID>`), run the quick GPU check, and look at the licence state |
| 18 to 23% of frames repeat | the render stage takes 42 to 46 ms against a 40 ms budget, as it did on GPU 0 while it was throttled and on GPU 1 from some time between 13:04 and 13:11 UTC on 2026-11-26 (19 to 22% at 13:30; GPU 1 had none while it was fast); sound and body stay in step. The ASR decode times in the gateway log show it too: 225 to 315 ms instead of 50 to 83 | run the quick GPU check: a throttled GPU needs a licence or a faster GPU |
| a word is mispronounced or missing | Kokoro skips words that are not in its dictionaries | add an entry, see [Pronunciations](#pronunciations) |
| answers are too long or contain lists | the model ignored the prompt | tighten `system_prompt`; `normalise` already strips markdown |
| Stage A runs out of memory | its peak is 26 GB at `--sample-area 960` and another process holds memory | stop `vllm` (or the other user) or use a lower `--sample-area` (the default 768 peaked at 22.7 GB); the Studio's worker waits for free memory instead, see above |
| onboarding prints `[fail] <name>` with a face error on a Stage A loop | the generated clip has a frame without a detectable face, or the photo and the clips have different proportions (the photo is fitted to the clips' frame size) | regenerate with `--seed 1` (or another) and `--overwrite`; give `--image` the photo the clips were made from |
| the loop's hands move too fast | `step_mae_max` of 12 or more in `speaking.json` | regenerate with another `--seed` and compare the sheets |
| a port is already in use | another service on this host | change it in `.env` |

## Not verified

* **A second machine, and real audio hardware.** Headless Chromium (fake microphone and camera, certificate errors ignored) and an
  aiortc client, both made from this host to its own LAN address and both through the TURN relay, passed. Not exercised: a browser
  on another machine, Wi-Fi, a network that blocks UDP, a person accepting the certificate warning and the microphone permission
  prompt, and a real microphone and real speakers (see [Network](#network-security-and-privacy)).
* **The Studio with the real motion model.** One full-motion HD job ran through the Studio on the real model: `newavatar`, started
  from the page by someone else at 13:14:51 UTC on 2026-11-26 and ready at 13:29:15 (863 s, one attempt, see
  [Avatar Studio](#avatar-studio)), and the renderer placed its head. Its clips were checked by numbers (the first frame against
  the photo, the motion from frame to frame) and not by eye, and no conversation with it was measured. Standard quality, cancel,
  retry and *Add full motion* were not run on the real model: they were run end to end against a real renderer with a stand-in
  generator (`services/bodygen/tests/fake_generator.py`, which copies existing clips), and in the worker's and the gateway's tests.
  The clips of `gulf-man-black` were made with `docker compose run bodygen` (the same generator) rather than through the worker.
  The page's time estimates (13, 8 and 1 minutes) were checked once, at HD (14.4 minutes). The Studio's pages were exercised in
  headless Chromium, and by whoever created `newavatar`.
* **Head placement on other avatars.** `gulf-man-black` was registered with it (cache format 4) and its steadiness was measured.
  `newavatar` was registered with it too (81 of 81 frames fitted in both clips) but its steadiness was not measured. The six others
  made here (format 3, or head only) were deleted in the Studio, so the placement was never run on them.
  An avatar from an older build can still wobble until it is registered again from its clips, which the Studio cannot do (the
  command line does, in about a minute). The steadiness was judged from landmark measurements, contact sheets and a headless
  browser run, not by a person watching it live. The `--lead-in 18` suggested for `e2e_client.py` after switching avatars was not
  run; 13 s is the load time in the renderer's log.
* **How long GPU 1 keeps working, and at what speed.** It is an unlicensed vGPU like GPU 0, which ran for two days at about a fifth
  of an H100's speed and then refused all CUDA work. GPU 1 was fast for 20 to 28 minutes (from 12:43:33 UTC until some time between
  13:03:39 and 13:11:36) and has computed at GPU 0's slow speed since, as the run at 13:30 and the ASR decode times up to 13:34 show.
  Why it was fast at first and why it slowed down are not known (see
  [GPU placement and the vGPU licence](#gpu-placement-and-the-vgpu-licence) and
  [Later the same day: GPU 1 slow](#later-the-same-day-gpu-1-slow)). It was watched until about 13:35 UTC and still accepted CUDA
  then. Check `curl -s http://127.0.0.1:18110/healthz` (`ready`, and `loaded` lists the avatar in use) and have a conversation before
  a demo. The three processes of other workloads that were on GPU 1 and were stopped for this were not started again, so how the
  stack behaves next to them is not known either.
* **Other GPUs.** An L40S (sm_89) engine build and a GPU with a valid licence were not run. GPU 1's figures while it was fast
  (515 TFLOPS, 3.3 TB/s) were in the range of an H100's ratings, and it then slowed down, see above. Even while it was fast the
  brief's 500 ms whole turn was not met (2.1 to 2.3 s; see [Results on GPU 1](#results-on-gpu-1)): the 0.7 s end-of-speech wait and
  Kokoro on the CPU (0.74 to 0.87 s for the first clause) make up about 1.5 s of it, and a faster GPU shortens neither.
* **Loudspeakers.** Barge-in was measured with a clean line-in; with speakers it needs echo cancellation or a headset.
* **A reboot of the host** (the containers restart by policy, which was not tried), outbound network traffic, Stage A above
  `--sample-area 960`, and long runs: the memory behaviour was checked over series of sessions, not over days.
* **Scale.** One conversation at a time; the renderer, the VAD configuration and the avatar selection are single-session.
* **Content.** Each avatar's motion is two generic clips (idle and speaking, 3.2 s each, played back and forth, so a movement repeats
  about every 6.5 s), the TTS is English only, Arabic names are approximate, and one voice serves every avatar.
