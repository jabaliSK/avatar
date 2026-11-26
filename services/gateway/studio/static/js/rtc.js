// One live conversation with an avatar: a WebRTC connection to the gateway that carries the person's voice up and the avatar's video and
// voice down, plus a data channel that carries the text of the conversation. This module reports what happens; the page decides what to show.
//
// The protocol is the one the gateway's own page uses: GET /openavatarchat/initconfig for the ICE servers and microphone constraints,
// POST /webrtc/offer for the SDP exchange and for trickled ICE candidates, and a data channel named "text" for JSON messages.

const CONFIG_URL = "/openavatarchat/initconfig";
const OFFER_URL = "/webrtc/offer";
const CONNECT_TIMEOUT_MS = 25000;
const LOST_TIMEOUT_MS = 10000;
const THINKING_TIMEOUT_MS = 40000;
const TEXT_READY_DELAY_MS = 1200;  // the gateway drops messages that arrive in the first half second of a stream
const STATS_INTERVAL_MS = 1000;
const MAX_RECORDS = 200;

export class ConversationError extends Error {}

const SERVICE_UNREACHABLE = "Cannot reach the conversation service. Check your network connection and try again.";
const NOT_STARTED = "The conversation could not be started. Try again in a moment.";
const LOST = "The connection was lost. Start the conversation again.";
const TIMED_OUT = "The connection could not be established. Check your network connection and try again.";

const timeoutSignal = (ms) => (typeof AbortSignal.timeout === "function" ? AbortSignal.timeout(ms) : undefined);

function randomId() {
  return Array.from(crypto.getRandomValues(new Uint8Array(6)), (byte) => byte.toString(16).padStart(2, "0")).join("");
}

async function postOffer(body) {
  let response;
  try {
    response = await fetch(OFFER_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      cache: "no-store",
      signal: timeoutSignal(20000),
    });
  } catch {
    throw new ConversationError(SERVICE_UNREACHABLE);
  }
  if (!response.ok) throw new ConversationError(NOT_STARTED);
  return response.json().catch(() => ({}));
}

async function loadConfig() {
  let response;
  try {
    response = await fetch(CONFIG_URL, { cache: "no-store", signal: timeoutSignal(8000) });
  } catch {
    throw new ConversationError(SERVICE_UNREACHABLE);
  }
  if (!response.ok) throw new ConversationError("The conversation service is not ready yet. Try again in a moment.");
  return response.json().catch(() => ({}));
}

/** The microphone, or the reason there is none. The conversation still works by typing, so this never throws. */
async function openMicrophone(constraints) {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    return { problem: "The microphone needs a secure (HTTPS) connection. You can type to the avatar instead." };
  }
  try {
    return { stream: await navigator.mediaDevices.getUserMedia({ audio: constraints || true }) };
  } catch (error) {
    const name = error && error.name;
    if (name === "OverconstrainedError") {
      try {
        return { stream: await navigator.mediaDevices.getUserMedia({ audio: true }) };
      } catch {
        return { problem: "The microphone could not be started. You can type to the avatar instead." };
      }
    }
    if (name === "NotAllowedError" || name === "SecurityError") {
      return { problem: "Microphone access is blocked. Allow it in your browser's site settings to talk, or type to the avatar instead." };
    }
    if (name === "NotFoundError") return { problem: "No microphone was found. You can type to the avatar instead." };
    return { problem: "The microphone could not be started. You can type to the avatar instead." };
  }
}

/** The gateway only sends video back when the offer carries a video track, so a tiny changing picture is sent up. */
function placeholderVideo() {
  const canvas = document.createElement("canvas");
  canvas.width = 64;
  canvas.height = 48;
  if (typeof canvas.captureStream !== "function") {
    throw new ConversationError("This browser cannot start a video conversation. Use a current version of Chrome, Edge, Firefox or Safari.");
  }
  const ctx = canvas.getContext("2d");
  let tick = 0;
  const paint = () => {
    ctx.fillStyle = tick++ % 2 ? "#000000" : "#010101";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
  };
  paint();
  const stream = canvas.captureStream(5);
  const timer = setInterval(paint, 200);
  return {
    track: stream.getVideoTracks()[0],
    stop() {
      clearInterval(timer);
      stream.getTracks().forEach((track) => track.stop());
    },
  };
}

/** Reports how loud the microphone is, about fifteen times a second. Nothing is played back. */
function watchLevel(audioContext, stream, report) {
  const source = audioContext.createMediaStreamSource(stream);
  const analyser = audioContext.createAnalyser();
  analyser.fftSize = 512;
  source.connect(analyser);  // not connected any further, so nothing is played back
  const samples = new Uint8Array(analyser.fftSize);
  let frame = 0;
  let last = 0;
  const tick = (now) => {
    frame = requestAnimationFrame(tick);
    if (now - last < 66) return;
    last = now;
    analyser.getByteTimeDomainData(samples);
    let sum = 0;
    for (const sample of samples) {
      const centred = (sample - 128) / 128;
      sum += centred * centred;
    }
    report(Math.min(1, Math.sqrt(sum / samples.length) * 4));
  };
  frame = requestAnimationFrame(tick);
  return () => {
    cancelAnimationFrame(frame);
    source.disconnect();
  };
}

function offerProblem(answer) {
  const code = answer && answer.meta && answer.meta.error;
  if (code === "concurrency_limit_reached") return "Another conversation is in progress. Please wait a moment and try again.";
  return NOT_STARTED;
}

/**
 * Starts a conversation and resolves once the offer has been answered. Handlers (all optional):
 *   onStream(MediaStream)   the avatar's video and voice
 *   onStatus(status)        "connected" (media is flowing and the person can type) | "reconnecting"
 *   onTurn(turn)            "listening" | "thinking" | "speaking"
 *   onCaptions(records)     [{id, role: "human" | "avatar", text, final, cancelled}]
 *   onLevel(0..1)           how loud the person is speaking
 *   onStats({rttMs, fps, width, height, route})
 *   onEnded(message)        the conversation is over; message is "" when the person ended it
 * Rejects with a ConversationError whose message is ready to show; onEnded is not called for a conversation that never started.
 */
export async function startConversation(handlers = {}) {
  const on = (name, ...args) => {
    try {
      if (handlers[name]) handlers[name](...args);
    } catch (error) {
      console.error(error);
    }
  };

  const config = await loadConfig();
  const constraints = config.track_constraints && config.track_constraints.audio;

  // Everything below is declared before the connection exists, because events can arrive as soon as it does.
  const parts = { mic: null, video: null, audioContext: null, stopLevel: null };
  const timers = { connect: 0, lost: 0, thinking: 0, stats: 0, ready: 0 };
  const records = [];
  let pc = null;
  let channel = null;
  let established = false;  // until this function returns, a failure rejects instead of calling onEnded
  let finished = false;
  let failure = "";
  let muted = false;
  let turn = "listening";
  let anonymous = 0;
  let lastVideo = null;

  const clearTimers = () => Object.values(timers).forEach((timer) => { clearTimeout(timer); clearInterval(timer); });

  function finish(message = "") {
    if (finished) return;
    finished = true;
    clearTimers();
    if (parts.stopLevel) parts.stopLevel();
    try { if (channel) channel.close(); } catch { /* already closed */ }
    if (pc) {
      pc.getTransceivers().forEach((transceiver) => { try { transceiver.stop(); } catch { /* already stopped */ } });
      pc.getSenders().forEach((sender) => { if (sender.track) sender.track.stop(); });
      const connection = pc;
      setTimeout(() => connection.close(), 400);
    }
    if (parts.mic) parts.mic.getTracks().forEach((track) => track.stop());
    if (parts.video) parts.video.stop();
    if (parts.audioContext) parts.audioContext.close().catch(() => {});
    if (established) on("onEnded", message);
    else failure = failure || message;
  }

  // ---------------------------------------------------------------------------------------------------------------- the conversation
  function setTurn(next) {
    if (turn === next) return;
    turn = next;
    clearTimeout(timers.thinking);
    if (next === "thinking") timers.thinking = setTimeout(() => setTurn("listening"), THINKING_TIMEOUT_MS);  // a reply that never starts must not leave "Thinking" on
    on("onTurn", next);
  }

  function emitCaptions() {
    on("onCaptions", records.map((record) => ({ ...record })));
  }

  function caption(role, payload) {
    if (typeof payload.text !== "string") return;
    const key = payload.stream_key || payload.request_id || `anonymous-${++anonymous}`;
    const id = `${role}-${key}`;
    const continues = payload.metadata && payload.metadata.continue_from_stream;
    if (continues !== undefined && continues !== null) {
      const index = records.findLastIndex((record) => record.role === role && record.id !== id);
      if (index >= 0) records.splice(index, 1);  // this turn replaces the unfinished one before it
    }
    let record = records.find((item) => item.id === id);
    if (!record) {
      if (!payload.text) return;
      record = { id, role, key, text: "", final: false, cancelled: false };
      records.push(record);
      if (records.length > MAX_RECORDS) records.shift();
    }
    record.text = payload.mode === "increment" ? record.text + payload.text : payload.text;
    record.final = Boolean(payload.end_of_speech);
    emitCaptions();
    const spoken = payload.text.trim() !== "";
    if (role === "human" && payload.end_of_speech && spoken) setTurn("thinking");
    else if (role === "avatar" && spoken && turn !== "speaking") setTurn("thinking");
  }

  function signal(payload) {
    if (payload.type === "stream_cancel") {
      const keys = new Set([payload.stream_key, ...(payload.parent_stream_keys || [])].filter(Boolean));
      let changed = false;
      for (const record of records) {
        if (keys.has(record.key) && !record.cancelled) {
          record.cancelled = true;
          changed = true;
        }
      }
      if (changed) emitCaptions();
    }
    if (payload.stream_type !== "client_playback") return;
    if (payload.type === "stream_begin") setTurn("speaking");
    else if (payload.type === "stream_end" || payload.type === "stream_cancel") setTurn("listening");
  }

  function wireChannel(target) {
    target.addEventListener("message", (event) => {
      let data;
      try {
        data = JSON.parse(event.data);
      } catch {
        return;
      }
      const name = data && data.header && data.header.name;
      const payload = (data && data.payload) || {};
      if (name === "EchoHumanText") caption("human", payload);
      else if (name === "EchoAvatarText") caption("avatar", payload);
      else if (name === "InterruptNotification" || name === "EndSpeech") setTurn("listening");
      else if (name === "ChatSignal") signal(payload);
    });
    target.addEventListener("close", () => finish("The conversation ended."));
  }

  // ---------------------------------------------------------------------------------------------------------------- measurements
  async function readStats() {
    if (finished || !pc) return;
    let report;
    try {
      report = await pc.getStats();
    } catch {
      return;
    }
    let transport = null;
    let video = null;
    const pairs = new Map();
    const locals = new Map();
    report.forEach((stat) => {
      if (stat.type === "transport") transport = stat;
      else if (stat.type === "candidate-pair") pairs.set(stat.id, stat);
      else if (stat.type === "local-candidate") locals.set(stat.id, stat);
      else if (stat.type === "inbound-rtp" && (stat.kind || stat.mediaType) === "video") video = stat;
    });
    let pair = transport && pairs.get(transport.selectedCandidatePairId);
    if (!pair) pair = [...pairs.values()].find((item) => item.state === "succeeded" && (item.nominated || item.selected));
    const stats = {};
    if (pair) {
      if (typeof pair.currentRoundTripTime === "number") stats.rttMs = Math.round(pair.currentRoundTripTime * 1000);
      const local = locals.get(pair.localCandidateId);
      if (local) stats.route = local.candidateType === "relay" ? "relay" : "direct";
    }
    if (video) {
      stats.width = video.frameWidth;
      stats.height = video.frameHeight;
      if (lastVideo && video.timestamp > lastVideo.timestamp) {
        stats.fps = Math.round(((video.framesDecoded - lastVideo.framesDecoded) * 1000) / (video.timestamp - lastVideo.timestamp));
      }
      lastVideo = { timestamp: video.timestamp, framesDecoded: video.framesDecoded };
    }
    on("onStats", stats);
  }

  function onConnectionState() {
    const state = pc.connectionState;
    if (state === "connected") {
      clearTimeout(timers.connect);
      clearTimeout(timers.lost);
      clearTimeout(timers.ready);
      timers.ready = setTimeout(() => { if (!finished) on("onStatus", "connected"); }, TEXT_READY_DELAY_MS);
      if (!timers.stats) timers.stats = setInterval(readStats, STATS_INTERVAL_MS);
    } else if (state === "disconnected") {
      on("onStatus", "reconnecting");
      clearTimeout(timers.lost);
      timers.lost = setTimeout(() => finish(LOST), LOST_TIMEOUT_MS);
    } else if (state === "failed") {
      finish(LOST);
    } else if (state === "closed") {
      finish("");
    }
  }

  // ---------------------------------------------------------------------------------------------------------------- connecting
  try {
    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
    parts.audioContext = AudioContextClass ? new AudioContextClass() : null;
    if (parts.audioContext && parts.audioContext.state === "suspended") parts.audioContext.resume().catch(() => {});

    const mic = await openMicrophone(constraints);
    let audioTrack;
    let notice = "";
    if (mic.stream) {
      parts.mic = mic.stream;
      audioTrack = mic.stream.getAudioTracks()[0];
    } else {
      notice = mic.problem;
      if (!parts.audioContext) throw new ConversationError("This browser cannot start a conversation. Use a current version of Chrome, Edge, Firefox or Safari.");
      audioTrack = parts.audioContext.createMediaStreamDestination().stream.getAudioTracks()[0];  // silence, so the gateway still gets an audio track
    }
    parts.video = placeholderVideo();

    pc = new RTCPeerConnection(config.rtc_configuration);
    const upstream = new MediaStream([audioTrack, parts.video.track]);
    pc.addTrack(audioTrack, upstream);
    pc.addTrack(parts.video.track, upstream);

    const remote = new MediaStream();
    pc.addEventListener("track", (event) => {
      if (!remote.getTracks().includes(event.track)) remote.addTrack(event.track);
      on("onStream", remote);
    });

    channel = pc.createDataChannel("text");
    wireChannel(channel);

    const webrtcId = randomId();
    const pending = [];  // candidates found before the gateway knows this connection
    let answered = false;
    pc.addEventListener("icecandidate", ({ candidate }) => {
      if (!candidate) return;
      const message = { candidate: candidate.toJSON(), webrtc_id: webrtcId, type: "ice-candidate" };
      if (answered) postOffer(message).catch(() => {});
      else pending.push(message);
    });
    pc.addEventListener("connectionstatechange", onConnectionState);

    await pc.setLocalDescription(await pc.createOffer());
    timers.connect = setTimeout(() => finish(TIMED_OUT), CONNECT_TIMEOUT_MS);
    const answer = await postOffer({ sdp: pc.localDescription.sdp, type: pc.localDescription.type, webrtc_id: webrtcId });
    if (finished) throw new ConversationError(failure || NOT_STARTED);
    if (answer.status === "failed" || !answer.sdp) throw new ConversationError(offerProblem(answer));
    await pc.setRemoteDescription({ type: answer.type, sdp: answer.sdp });
    answered = true;
    for (const message of pending.splice(0)) postOffer(message).catch(() => {});
    if (finished) throw new ConversationError(failure || NOT_STARTED);

    established = true;
    if (parts.mic && parts.audioContext) parts.stopLevel = watchLevel(parts.audioContext, parts.mic, (level) => on("onLevel", level));

    return {
      /** A problem with the microphone the person should be told about; the conversation still works by typing. */
      notice,
      micAvailable: Boolean(parts.mic),
      get muted() {
        return muted;
      },
      setMuted(value) {
        muted = Boolean(value);
        if (parts.mic) parts.mic.getAudioTracks().forEach((track) => { track.enabled = !muted; });
      },
      sendText(text) {
        const value = String(text).trim();
        if (!value || finished || !channel || channel.readyState !== "open") return false;
        channel.send(JSON.stringify({
          header: { name: "SendHumanText", request_id: randomId() },
          payload: { request_id: randomId(), stream_key: randomId(), mode: "full_text", text: value, end_of_speech: true },
        }));
        return true;
      },
      interrupt() {
        if (finished || !channel || channel.readyState !== "open") return false;
        channel.send(JSON.stringify({ header: { name: "Interrupt", request_id: randomId() }, payload: {} }));
        return true;
      },
      stop() {
        finish("");
      },
    };
  } catch (error) {
    if (!(error instanceof ConversationError)) console.error(error);
    finish();
    throw new ConversationError(error instanceof ConversationError ? error.message : failure || NOT_STARTED);
  }
}
