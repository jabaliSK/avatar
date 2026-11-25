"use strict";

const $ = (id) => document.getElementById(id);
const holoboxMode = new URLSearchParams(location.search).get("display") === "holobox";
document.body.classList.toggle("holobox", holoboxMode);
let avatars = [];
let selected = null;
let sessionId = null;
let busy = false;
let switching = false;
let creatingAvatar = false;
let modelsReady = false;
let recording = null;
let microphoneStream = null;
let recordTimer = null;
let previewUrl = null;
let toastTimer = null;
let liveMode = false;
let liveReady = false;
let liveSocket = null;
let livePhase = "connecting";
let lastLatency = "";
let liveTurn = false;

function showIdle() {
  $("live-canvas").hidden = true;
  $("avatar-video").pause();
  $("avatar-video").hidden = true;
  resumeIdle();
}

function resumeIdle() {
  const idle = $("idle-video");
  const src = idle.getAttribute("src");
  idle.hidden = !src || document.hidden;
  $("portrait").hidden = !idle.hidden;
  if (idle.hidden) return;
  idle.play().catch((error) => {
    if (idle.getAttribute("src") !== src || !$("live-canvas").hidden || !$("avatar-video").hidden) return;
    idle.hidden = true;
    $("portrait").hidden = false;
    if (!document.hidden && error.name !== "AbortError") {
      notify(`Idle animation could not play: ${error.message}`);
    }
  });
}

function decodeBase64(encoded) {
  const raw = atob(encoded);
  const bytes = new Uint8Array(raw.length);
  for (let i = 0; i < bytes.length; i++) bytes[i] = raw.charCodeAt(i);
  return bytes;
}

class LivePlayback {
  constructor() {
    this.context = null;
    this.sources = new Set();
    this.frames = [];
    this.epoch = 0;
    this.animation = null;
    this.next = 0;
    this.done = false;
    this.stats = {};
    this.initialBufferSeconds = 1.3;
    this.pending = [];
    this.pendingSeconds = 0;
    this.bufferTarget = this.initialBufferSeconds;
  }

  reset() {
    this.epoch++;
    cancelAnimationFrame(this.animation);
    for (const source of this.sources) source.stop();
    this.sources.clear();
    for (const frame of this.frames) frame.image.close();
    this.frames = [];
    for (const packet of this.pending) {
      for (const image of packet.images) image.close();
    }
    this.pending = [];
    this.pendingSeconds = 0;
    this.bufferTarget = this.initialBufferSeconds;
    this.next = 0;
    this.done = false;
  }

  async start() {
    this.reset();
    const Audio = window.AudioContext || window.webkitAudioContext;
    if (!Audio) throw new Error("Live audio requires a browser supporting Web Audio.");
    if (!this.context) this.context = new Audio();
    await this.context.resume();
    if (this.context.state !== "running") throw new Error("Enable browser audio playback and retry.");
    this.started = performance.now();
    this.stats = { firstPlaybackSeconds: null, underruns: 0, maxQueuedFrames: 0, segments: 0 };
    this.tick();
  }

  async enqueue(segment) {
    const epoch = this.epoch;
    while (this.next - this.context.currentTime > 2.4) {
      await new Promise((resolve) => setTimeout(resolve, 25));
      if (epoch !== this.epoch) return false;
    }
    const images = [];
    try {
      for (const encoded of segment.frames) {
        const bytes = decodeBase64(encoded);
        const image = await createImageBitmap(new Blob([bytes], { type: "image/jpeg" }));
        images.push(image);
        if (epoch !== this.epoch) {
          for (const item of images) item.close();
          return false;
        }
      }
    } catch (error) {
      for (const image of images) image.close();
      throw error;
    }
    const bytes = decodeBase64(segment.audio);
    const data = new DataView(bytes.buffer);
    const buffer = this.context.createBuffer(1, bytes.length / 2, segment.rate);
    const samples = buffer.getChannelData(0);
    for (let i = 0; i < samples.length; i++) samples[i] = data.getInt16(i * 2, true) / 32768;
    this.pending.push({ images, buffer, fps: segment.fps });
    this.pendingSeconds += buffer.duration;
    this.stats.segments++;
    this.schedule();
    return true;
  }

  schedule() {
    if (!this.pending.length) return;
    const now = this.context.currentTime;
    if (this.next && this.next < now + 0.025) {
      this.stats.underruns++;
      this.next = 0;
      this.bufferTarget = Math.min(2.4, this.bufferTarget + 0.5);
    }
    if (!this.next && !this.done && this.pendingSeconds < this.bufferTarget) {
      livePhase = "buffering";
      return;
    }
    // Keep on-time chunks contiguous; add scheduling headroom only on start or restart.
    let start = this.next || now + 0.05;
    for (const packet of this.pending) {
      this.schedulePacket(packet, start);
      start += packet.buffer.duration;
    }
    this.next = start;
    this.pending = [];
    this.pendingSeconds = 0;
    this.stats.maxQueuedFrames = Math.max(this.stats.maxQueuedFrames, this.frames.length);
  }

  schedulePacket({ images, buffer, fps }, start) {
    const source = this.context.createBufferSource();
    source.buffer = buffer;
    source.connect(this.context.destination);
    source.onended = () => { source.disconnect(); this.sources.delete(source); };
    this.sources.add(source);
    source.start(start);
    images.forEach((image, index) => this.frames.push({ at: start + index / fps, image }));
  }

  tick() {
    const now = this.context.currentTime;
    let newest = null;
    while (this.frames.length && this.frames[0].at <= now) {
      if (newest) newest.image.close();
      newest = this.frames.shift();
    }
    if (newest) {
      const canvas = $("live-canvas");
      if (canvas.width !== newest.image.width || canvas.height !== newest.image.height) {
        canvas.width = newest.image.width;
        canvas.height = newest.image.height;
      }
      canvas.getContext("2d").drawImage(newest.image, 0, 0);
      newest.image.close();
      $("live-canvas").hidden = false;
      $("portrait").hidden = true;
      if (!$("idle-video").hidden) $("idle-video").pause();
      $("idle-video").hidden = true;
      if (this.stats.firstPlaybackSeconds === null) {
        this.stats.firstPlaybackSeconds = (performance.now() - this.started) / 1000;
        lastLatency = `First playback: ${this.stats.firstPlaybackSeconds.toFixed(2)}s`;
      }
      if (livePhase !== "streaming" && !this.pending.length) {
        livePhase = "streaming";
        updateControls();
      }
    }
    if (!this.done && this.next && now >= this.next + 0.2 && livePhase === "streaming") {
      livePhase = "buffering";
      updateControls();
    }
    if (this.done && now >= this.next) {
      finishLive();
      return;
    }
    this.animation = requestAnimationFrame(() => this.tick());
  }
}

const playback = new LivePlayback();

function finishLive() {
  playback.reset();
  busy = false;
  liveTurn = false;
  livePhase = "ready";
  showIdle();
  updateControls();
  renderLibrary();
}

function disconnectLive() {
  const socket = liveSocket;
  liveSocket = null;
  liveReady = false;
  playback.reset();
  if (socket) socket.close();
  $("idle-video").pause();
  $("idle-video").removeAttribute("src");
  $("idle-video").hidden = true;
  $("live-canvas").hidden = true;
}

function connectLive() {
  disconnectLive();
  if (!liveMode || !sessionId) return;
  livePhase = "connecting";
  const socket = new WebSocket(`${location.protocol === "https:" ? "wss:" : "ws:"}//${location.host}/api/sessions/${sessionId}/live`);
  liveSocket = socket;
  socket.onmessage = async (message) => {
    if (socket !== liveSocket) return;
    try {
      const event = JSON.parse(message.data);
      if (event.event === "prepared") {
        $("idle-video").src = event.idle;
        liveReady = true;
        livePhase = "ready";
        document.querySelector('[data-provider="renderer"] small').textContent = "WORKER WARM";
        showIdle();
      } else if (event.event === "stage") {
        livePhase = event.stage;
      } else if (event.event === "segment") {
        if (!liveTurn || livePhase === "stopping") return;
        if (await playback.enqueue(event) && socket === liveSocket) {
          socket.send(JSON.stringify({ type: "ack", index: event.index }));
        }
      } else if (event.event === "complete") {
        appendTurn(event.turn);
        $("message").value = "";
        playback.stats.generationCompleteAt = performance.now();
        playback.done = true;
        playback.schedule();
        if (playback.stats.firstPlaybackSeconds !== null) {
          lastLatency = `First playback: ${playback.stats.firstPlaybackSeconds.toFixed(2)}s`;
        }
      } else if (event.event === "cancelled") {
        finishLive();
      } else if (event.event === "error") {
        finishLive();
        notify(event.message);
      }
      updateControls();
    } catch (error) {
      if (socket === liveSocket) {
        socket.send(JSON.stringify({ type: "cancel" }));
        finishLive();
        notify(`Live playback failed: ${error.message}`);
      }
    }
  };
  socket.onerror = () => {
    if (socket === liveSocket) notify("Live connection failed. Check the server and refresh model status to reconnect.");
  };
  socket.onclose = () => {
    if (socket !== liveSocket) return;
    liveReady = false;
    const interrupted = liveTurn;
    finishLive();
    livePhase = "disconnected";
    updateControls();
    if (interrupted) notify("Live connection closed before the reply finished. Refresh model status to reconnect.");
  };
  updateControls();
}

async function sendLive(text) {
  if (busy || !liveReady || !liveSocket || liveSocket.readyState !== WebSocket.OPEN) return;
  busy = true;
  liveTurn = true;
  lastLatency = "";
  livePhase = "thinking";
  showIdle();
  updateControls();
  renderLibrary();
  try {
    await playback.start();
    liveSocket.send(JSON.stringify({ type: "message", text }));
  } catch (error) {
    finishLive();
    notify(error.message);
  }
}

function notify(message, success = false) {
  const toast = $("notification");
  toast.textContent = message;
  toast.classList.toggle("success", success);
  toast.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { toast.hidden = true; }, success ? 4500 : 12000);
}

async function api(url, options = {}) {
  const response = await fetch(url, options);
  if (!response.ok) {
    let detail = `Request failed (${response.status}).`;
    try {
      const body = await response.json();
      detail = typeof body.detail === "string" ? body.detail : "Please check the input and try again.";
    } catch { /* Keep the HTTP status when the server does not return JSON. */ }
    throw new Error(detail);
  }
  return response.json();
}

function updateControls() {
  const ready = Boolean(selected && sessionId && modelsReady && (!liveMode || liveReady) && !busy && !switching && !creatingAvatar);
  $("message").disabled = !ready || Boolean(recording);
  $("send").disabled = !ready || Boolean(recording);
  $("record").disabled = !ready;
  $("new-chat").disabled = !selected || busy || switching || creatingAvatar || Boolean(recording);
  $("share").disabled = !selected || busy || switching || creatingAvatar;
  $("holobox-view").disabled = !selected || busy || switching || creatingAvatar;
  $("create-button").disabled = busy || switching || creatingAvatar || Boolean(recording);
  $("stage-state").textContent = busy ? "GENERATING REPLY" : recording ? "LISTENING" : selected ? "READY TO CONVERSE" : "AWAITING PORTRAIT";
  $("voice-status").textContent = recording ? "Recording · click Stop to send" : busy ? "Generating a synchronized reply..." : modelsReady ? "Ready when you are" : "Model setup required";
  $("stop-reply").hidden = !liveTurn;
  if (liveMode && selected) {
    const phases = {
      connecting: "Connecting live session...", warming: "One-time model warmup...",
      preparing: "Preparing portrait and idle animation...", ready: "Listening",
      thinking: "Thinking...", speaking: "Synthesizing speech...", streaming: "Speaking live",
      buffering: "Buffering reply...",
      stopping: "Stopping reply...", disconnected: "Disconnected - refresh model status",
    };
    $("stage-state").textContent = livePhase.toUpperCase();
    $("voice-status").textContent = recording ? "Recording - click Stop to send" : `${phases[livePhase] || livePhase}${lastLatency ? ` / ${lastLatency}` : ""}`;
  }
}

async function refreshStatus() {
  const status = await api("/api/status");
  liveMode = status.mode === "live";
  modelsReady = Object.values(status.pipeline).every((provider) => provider.configured);
  for (const [name, provider] of Object.entries(status.pipeline)) {
    const node = document.querySelector(`[data-provider="${name}"]`);
    node.classList.toggle("ready", provider.configured);
    node.querySelector("small").textContent = provider.configured
      ? provider.mode === "endpoint" ? "ENDPOINT CONFIGURED · NOT PROBED" : "MODEL FILES PRESENT"
      : "SETUP REQUIRED";
  }
  if (liveMode) {
    $("mode-note").textContent = "LIVE TEXT / Warm GPU, synchronized streamed speech and video. Mic uses recorded turns and requires HTTPS.";
    const rendererStatus = document.querySelector('[data-provider="renderer"] small');
    rendererStatus.textContent = status.live.error ? "WORKER FAILED" : status.live.ready ? "WORKER WARM" : "WORKER WARMING";
    if (status.live.error) notify(status.live.error);
    if (sessionId && (!liveSocket || liveSocket.readyState === WebSocket.CLOSED)) connectLive();
  }
  const renderer = status.pipeline.renderer;
  playback.initialBufferSeconds = Math.max(1.3, renderer.initial_buffer_seconds || 0);
  if (renderer.label) {
    $("renderer-label").textContent = renderer.label;
    $("renderer-caption").textContent = renderer.label.toUpperCase();
    $("renderer-description").textContent = renderer.description;
  }
  updateControls();
}

function renderLibrary() {
  $("library").replaceChildren();
  $("avatar-count").textContent = `${avatars.length} avatar${avatars.length === 1 ? "" : "s"}`;
  if (!avatars.length) {
    const empty = document.createElement("p");
    empty.className = "empty-small";
    empty.textContent = "Your saved avatars will appear here.";
    $("library").append(empty);
  }
  for (const avatar of avatars) {
    const button = document.createElement("button");
    button.className = `library-item${selected && selected.id === avatar.id ? " active" : ""}`;
    button.disabled = busy || switching || creatingAvatar || Boolean(recording);
    const image = document.createElement("img");
    image.src = avatar.image;
    image.alt = "";
    const label = document.createElement("span");
    label.textContent = avatar.name;
    const sub = document.createElement("small");
    sub.textContent = "Photo avatar";
    label.append(sub);
    button.append(image, label);
    button.addEventListener("click", () => selectAvatar(avatar).catch((error) => notify(error.message)));
    $("library").append(button);
  }
}

async function newConversation() {
  const session = await api("/api/sessions", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ avatar_id: selected.id }),
  });
  sessionId = session.id;
  disconnectLive();
  localStorage.setItem(`avatar-session:${selected.id}`, sessionId);
  $("transcript").replaceChildren();
  $("avatar-video").pause();
  $("avatar-video").hidden = true;
  $("portrait").hidden = false;
  connectLive();
  updateControls();
}

async function selectAvatar(avatar) {
  if (busy || recording || switching) return;
  switching = true;
  try {
    disconnectLive();
    sessionId = null;
    selected = avatar;
    history.replaceState(null, "", avatar.url + (holoboxMode ? "?display=holobox" : ""));
    $("exit-holobox").href = avatar.url;
    $("stage-title").textContent = avatar.name;
    $("stage-placeholder").hidden = true;
    $("portrait").src = avatar.image;
    $("source-photo").href = avatar.image;
    $("source-photo").hidden = false;
    $("portrait").hidden = false;
    $("avatar-video").pause();
    $("avatar-video").hidden = true;
    $("transcript").replaceChildren();
    renderLibrary();
    updateControls();
    const savedId = localStorage.getItem(`avatar-session:${avatar.id}`);
    if (savedId) {
      let session;
      try {
        session = await api(`/api/sessions/${savedId}`);
      } catch (error) {
        // An expired local reference is recoverable, but still notify the user.
        notify(`Could not restore the previous conversation: ${error.message}`);
        localStorage.removeItem(`avatar-session:${avatar.id}`);
      }
      if (session && session.avatar_id === avatar.id) {
        sessionId = session.id;
        for (const turn of session.turns) appendTurn(turn);
        connectLive();
        updateControls();
        return;
      }
    }
    await newConversation();
  } finally {
    switching = false;
    updateControls();
    renderLibrary();
  }
}

function addMessage(role, text, elapsed = null) {
  const entry = document.createElement("div");
  entry.className = `message ${role}`;
  const name = document.createElement("strong");
  name.textContent = role === "user" ? "YOU" : selected.name.toUpperCase();
  const content = document.createElement("p");
  content.textContent = text;
  entry.append(name, content);
  if (elapsed !== null) {
    const time = document.createElement("small");
    time.textContent = `${elapsed}s to generate`;
    entry.append(time);
  }
  $("transcript").append(entry);
  $("transcript").scrollTop = $("transcript").scrollHeight;
}

function appendTurn(turn) {
  addMessage("user", turn.user_text);
  addMessage("assistant", turn.assistant_text, turn.elapsed);
}

async function sendTurn(form) {
  if (busy || !sessionId) return;
  busy = true;
  playback.reset();
  $("idle-video").pause();
  $("avatar-video").pause();
  $("stage-loading").hidden = false;
  updateControls();
  renderLibrary();
  try {
    const turn = await api(`/api/sessions/${sessionId}/turn`, { method: "POST", body: form });
    appendTurn(turn);
    $("message").value = "";
    $("avatar-video").src = turn.video;
    $("avatar-video").hidden = false;
    $("portrait").hidden = true;
    try {
      await $("avatar-video").play();
    } catch {
      notify("Your reply is ready. Press Play on the video to hear it.", true);
    }
  } catch (error) {
    notify(error.message);
  } finally {
    busy = false;
    $("stage-loading").hidden = true;
    updateControls();
    renderLibrary();
  }
}

$("image").addEventListener("change", () => {
  const file = $("image").files[0];
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = file ? URL.createObjectURL(file) : null;
  $("preview").hidden = !previewUrl;
  $("upload-copy").hidden = Boolean(previewUrl);
  if (previewUrl) $("preview").src = previewUrl;
});

$("avatar-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (busy || recording || switching || creatingAvatar) return;
  creatingAvatar = true;
  const form = new FormData();
  form.append("name", $("name").value.trim());
  form.append("consent", String($("consent").checked));
  form.append("image", $("image").files[0]);
  updateControls();
  renderLibrary();
  try {
    const created = await api("/api/avatars", { method: "POST", body: form });
    avatars = await api("/api/avatars");
    await selectAvatar(avatars.find((avatar) => avatar.id === created.id));
    notify("Avatar saved. Your browser URL is ready to copy.", true);
  } catch (error) {
    notify(error.message);
  } finally {
    creatingAvatar = false;
    updateControls();
    renderLibrary();
  }
});

$("message-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const text = $("message").value.trim();
  if (!text || recording) return;
  if (liveMode) {
    await sendLive(text);
    return;
  }
  const form = new FormData();
  form.append("text", text);
  await sendTurn(form);
});

$("record").addEventListener("click", async () => {
  if (recording) {
    if (recording.state === "recording") recording.stop();
    return;
  }
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia || !window.MediaRecorder) {
    notify("Microphone recording requires a supported browser on localhost or HTTPS. You can still type.");
    return;
  }
  switching = true;
  updateControls();
  renderLibrary();
  try {
    microphoneStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const mimeType = ["audio/webm;codecs=opus", "audio/mp4", "audio/ogg;codecs=opus"]
      .find((type) => MediaRecorder.isTypeSupported(type));
    if (!mimeType) throw new Error("This browser does not support a compatible recording format.");
    const chunks = [];
    recording = new MediaRecorder(microphoneStream, { mimeType });
    let failed = false;
    recording.addEventListener("dataavailable", (event) => {
      if (event.data.size) chunks.push(event.data);
    });
    recording.addEventListener("stop", async () => {
      clearTimeout(recordTimer);
      if (microphoneStream) microphoneStream.getTracks().forEach((track) => track.stop());
      const blob = new Blob(chunks, { type: mimeType });
      recording = null;
      microphoneStream = null;
      $("record").classList.remove("recording");
      $("record").textContent = "Mic";
      $("record").setAttribute("aria-label", "Record a voice message");
      if (failed || !blob.size) {
        updateControls();
        renderLibrary();
        if (!failed) notify("No audio was recorded. Please try again.");
        return;
      }
      const form = new FormData();
      const extension = mimeType.startsWith("audio/mp4") ? "mp4" : mimeType.startsWith("audio/ogg") ? "ogg" : "webm";
      form.append("audio", blob, `recording.${extension}`);
      await sendTurn(form);
    });
    recording.addEventListener("error", () => {
      failed = true;
      notify("Microphone recording failed. Please try again or type your message.");
    });
    recording.start();
    switching = false;
    $("record").classList.add("recording");
    $("record").textContent = "Stop";
    $("record").setAttribute("aria-label", "Stop recording and send voice message");
    recordTimer = setTimeout(() => { if (recording && recording.state === "recording") recording.stop(); }, 59000);
    updateControls();
    renderLibrary();
  } catch (error) {
    if (microphoneStream) microphoneStream.getTracks().forEach((track) => track.stop());
    microphoneStream = null;
    recording = null;
    switching = false;
    notify(`Could not record: ${error.message}`);
    updateControls();
    renderLibrary();
  }
});

$("share").addEventListener("click", async () => {
  const url = new URL(selected.url + (holoboxMode ? "?display=holobox" : ""), location.origin).href;
  try {
    await navigator.clipboard.writeText(url);
    notify("Avatar URL copied. It works wherever this server is reachable.", true);
  } catch {
    notify(`Copy this avatar URL: ${url}`, true);
  }
});
$("holobox-view").addEventListener("click", () => {
  if (selected) location.assign(`${selected.url}?display=holobox`);
});
$("new-chat").addEventListener("click", async () => {
  if (switching || busy || creatingAvatar || recording) return;
  switching = true;
  updateControls();
  renderLibrary();
  try {
    await newConversation();
  } catch (error) {
    notify(error.message);
  } finally {
    switching = false;
    updateControls();
    renderLibrary();
  }
});
$("refresh-status").addEventListener("click", () => refreshStatus().catch((error) => notify(error.message)));
$("stop-reply").addEventListener("click", () => {
  playback.reset();
  showIdle();
  livePhase = "stopping";
  updateControls();
  if (liveSocket && liveSocket.readyState === WebSocket.OPEN) {
    liveSocket.send(JSON.stringify({ type: "cancel" }));
  } else {
    finishLive();
  }
});
$("avatar-video").addEventListener("ended", () => { if (liveMode) showIdle(); });
document.addEventListener("visibilitychange", () => {
  if (!document.hidden && selected && $("live-canvas").hidden && $("avatar-video").hidden) resumeIdle();
});
window.addEventListener("beforeunload", () => {
  disconnectLive();
  if (microphoneStream) microphoneStream.getTracks().forEach((track) => track.stop());
});

async function init() {
  await refreshStatus();
  avatars = await api("/api/avatars");
  renderLibrary();
  const id = location.pathname.startsWith("/avatar/") ? location.pathname.split("/")[2] : null;
  const avatar = avatars.find((item) => item.id === id);
  if (avatar) await selectAvatar(avatar);
}
init().catch((error) => notify(`Unable to load the workspace: ${error.message}`));
