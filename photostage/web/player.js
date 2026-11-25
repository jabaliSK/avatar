"use strict";
// The player page of one avatar: /a/<key>. The avatar fills the screen; a dock below holds the reply text, a status
// line and the chat bar. Each of those is an option (see options.js) that the avatar is saved with and the page's
// address can override, so that a screen which should show nothing but the avatar can be set up by its address alone.
const $ = (id) => document.getElementById(id);
const key = location.pathname.split("/")[2];
const playback = new LivePlayback();
const CROSSFADE_SECONDS = 0.25;
const HEAD_FADE_SECONDS = 0.5;
const wait = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const DATA_IMAGE = { colorSpaceConversion: "none", premultiplyAlpha: "none" };
const address = new URLSearchParams(location.search);
let options = {}, compositor = null, idle = null, turn = null, history = [], recorder = null, motion = null, clips = null;
// Whether the service pictures speech whole (MuseTalk dubs it into the clips): the gestures are then in its pictures.
let dubbed = false;
let mix = 0, head = 0, shownLive = false, shownIdle = null, lastTick = performance.now(), lastPhase = "";
// The filler phrases said while a reply is being made ("Let me check that."): {box, fps, fillers, buffer}, and the next.
let fillers = null, nextFiller = 0;
const seconds = () => performance.now() / 1000;
// IDLE, LISTENING, THINKING or SPEAKING (conversation.js). A page that embeds or drives the player can read it from the
// player's data-state, or follow the "photostage:state" events.
const conversation = new Conversation((state, from, cause) => {
  $("player").dataset.state = state;
  withClips((stage) => stage.setState(state, seconds()));
  window.dispatchEvent(new CustomEvent("photostage:state", { detail: { state, from, cause } }));
});
$("player").dataset.state = conversation.state;
window.photoStageConversation = conversation;

// Progress messages follow the "status" option. Errors always show: a viewer must be able to tell why nothing happens.
function setStatus(text, kind) {
  const line = $("status");
  line.classList.toggle("error", kind === "error");
  line.textContent = kind === "error" || options.status ? text : "";
}

const megabytes = (bytes) => (bytes / 1048576).toFixed(1);
let shownFraction = 0;
// The total grows as the responses arrive, so the bar is kept from moving backwards.
const progress = new Progress(({ loaded, total, fraction }) => {
  shownFraction = Math.max(shownFraction, fraction);
  $("loading-bar").hidden = false;
  $("loading-fill").style.width = `${Math.round(shownFraction * 100)}%`;
  $("loading-text").textContent = `Loading the avatar… ${megabytes(loaded)} of ${megabytes(total)} MB`;
});

let failed = false;
function failLoading(message, withHome) {
  failed = true;
  $("loading").hidden = false;
  $("loading").classList.add("failed");
  $("loading-text").textContent = message;
  $("loading-bar").hidden = true;
  $("loading-home").hidden = !withHome;
}

// The body is an extra: whatever goes wrong with its motion must not freeze the avatar or stop the speech.
function stopMotion(error) {
  console.error("Body motion stopped:", error);
  motion = null;
  if (compositor) compositor.setPose(null);
}

// The body clips are an extra too: whatever goes wrong with them gives the avatar its procedural body back.
function stopClips(error) {
  console.error("Body clips stopped:", error);
  const stage = clips;
  clips = null;
  window.photoStageClips = null;
  try {
    if (stage) stage.dispose();
  } catch (inner) {
    console.error(inner);
  }
}

function withClips(action) {
  if (!clips) return;
  try {
    action(clips);
  } catch (error) {
    stopClips(error);
  }
}

// While a clip is drawn the body is the clip's. The photo's hands stay as they are (every clip begins and ends in that pose,
// so a hand that has moved would not match it) and the sway gives way to the clip by the clip's weight.
function underClips(pose, weight) {
  if (!pose) return null;
  const rest = 1 - weight;
  return { sway: [pose.sway[0] * rest, pose.sway[1] * rest], hands: [] };
}

// The player hands over each stretch of speech it schedules, so that the clips can time their gestures to it and the hands can
// follow its stresses. A clip is timed on the page's clock: the speech begins at the audio clock's `start`, here.
playback.onAudio = (samples, rate, start) => {
  withClips((stage) => stage.speechStart(performance.now() / 1000 + start - playback.context.currentTime));
  if (!motion) return;
  try {
    motion.hear(samples, rate, start);
  } catch (error) {
    stopMotion(error);
  }
};

async function getJson(url, init) {
  const response = await fetch(url, init);
  if (!response.ok) throw new DownloadError(`Could not load ${url} (${response.status}).`, response.status);
  return response.json();
}

async function bitmap(url, required = true) {
  let blob;
  try {
    blob = await download(url, { onProgress: (loaded, total) => progress.update(url, loaded, total) });
  } catch (error) {
    if (!required && error.status === 404) return null;
    throw error;
  }
  return createImageBitmap(blob, DATA_IMAGE);
}

// The idle loop is a pack of JPEG frames (and detail guides) played forward then
// backward, so it loops without a jump and decodes exactly like the live frames.
async function loadIdle(base) {
  const [info, buffer] = await Promise.all([
    getJson(`${base}idle.json`),
    download(`${base}idle.bin`).then((blob) => new Response(blob).arrayBuffer()),
  ]);
  const decode = (offset, length, type) => createImageBitmap(
    new Blob([new Uint8Array(buffer, offset, length)], { type }), DATA_IMAGE);
  const frames = await Promise.all(info.frames.map(async ([at, size, guideAt, guideSize]) => ({
    image: await decode(at, size, "image/jpeg"),
    guide: guideSize ? await decode(guideAt, guideSize, "image/png") : null,
  })));
  const period = 2 * frames.length - 2;
  return {
    at(seconds) {
      const step = Math.floor(seconds * info.fps) % period;
      return frames[step < frames.length ? step : period - step];
    },
  };
}

// The body rig is an extra: an avatar made before it existed, or one whose body could not be built, still works as the
// photo with the animated head.
async function loadBody(base) {
  try {
    const rig = await getJson(`${base}body.json`, { cache: "no-cache" });
    const [field, ...hands] = await Promise.all([
      rig.field ? bitmap(`${base}${rig.field}`) : null,
      ...rig.hands.map(async (hand) => ({
        alpha: await bitmap(`${base}${hand.alpha}`), plate: await bitmap(`${base}${hand.plate}`),
      })),
    ]);
    return { rig, field, hands };
  } catch (error) {
    if (error.status !== 404) console.warn("The body could not be loaded, so it will not move:", error);
    return null;
  }
}

const bodyImages = (body) => (body ? [body.field, ...body.hands.flatMap((hand) => [hand.alpha, hand.plate])] : []);

// The body clips are an extra like the body rig: an avatar without a clip pack, one whose pack this browser cannot use, or one
// with the clips switched off goes on with the procedural body. (Switched off is the option, or no body motion at all.)
const clipsWanted = () => options.clips !== false && options.motion > 0;

async function loadClipPack(base) {
  if (!clipsWanted()) return null;
  try {
    return await getJson(`${base}clips.json`, { cache: "no-cache" });
  } catch (error) {
    if (error.status !== 404) console.warn("The body clips could not be loaded, so the body moves as it does without them:", error);
    return null;
  }
}

// A <video> for the pool of clips. It sits in the page, too small to see and fully transparent, because some browsers do not
// decode the video of an element that is not in it, or that is not drawn at all.
function clipVideo() {
  const video = document.createElement("video");
  video.muted = true;
  video.playsInline = true;
  video.preload = "auto";
  video.disablePictureInPicture = true;
  video.setAttribute("muted", "");
  video.setAttribute("playsinline", "");
  video.setAttribute("aria-hidden", "true");
  video.tabIndex = -1;
  video.style.cssText = "position:fixed;left:0;top:0;width:2px;height:2px;opacity:0;pointer-events:none";
  document.body.appendChild(video);
  return video;
}

const clipCanvases = new WeakMap();
function clipSnapshot(video, time) {
  if (typeof VideoFrame === "function") return new VideoFrame(video, { timestamp: Math.round(time * 1e6) });
  let canvas = clipCanvases.get(video);
  if (!canvas) {
    canvas = document.createElement("canvas");
    clipCanvases.set(video, canvas);
  }
  if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
  }
  canvas.getContext("2d").drawImage(video, 0, 0);
  return canvas;
}

// A small picture of `region` ([x, y, width, height]) of a photo or a video frame, as rows of [r, g, b] in 0-1: what the
// colours of the clips are fitted on (see clipTone).
function colourSample(source, region) {
  const [x, y, width, height] = region;
  const across = CLIP_TONE_SIDE, down = Math.max(8, Math.round(CLIP_TONE_SIDE * height / width));
  const canvas = document.createElement("canvas");
  canvas.width = across;
  canvas.height = down;
  const context = canvas.getContext("2d", { willReadFrequently: true });
  context.imageSmoothingQuality = "high";
  context.drawImage(source, x, y, width, height, 0, 0, across, down);
  const { data } = context.getImageData(0, 0, across, down);
  const rows = [];
  for (let i = 0; i < data.length; i += 4) rows.push([data[i] / 255, data[i + 1] / 255, data[i + 2] / 255]);
  return rows;
}

// The colours that turn the first frame of a thinking clip, as this browser shows it, into the photo's own; null where there is
// nothing to do (the fit is the identity, or the clip did not load in time).
async function fitClipTone(reels, manifest, photoRows) {
  const names = Object.keys(manifest.clips);
  const name = names.find((n) => manifest.clips[n].role === "think") ||
    names.find((n) => manifest.clips[n].role !== "talk");
  if (!name || reels.failed(name)) return null;
  reels.prepare([name]);
  for (let i = 0; i < 40 && !reels.ready(name) && !reels.failed(name); i++) await wait(100);
  const video = reels.video(name);
  if (!video || !reels.ready(name) || !(video.videoWidth > 0)) return null;
  const fit = clipTone(colourSample(video, [0, 0, video.videoWidth, video.videoHeight]), photoRows);
  return fit.fitted ? fit : null;
}

async function loadQuietBank(base, spec, fps, count = null) {
  const [info, blob] = await Promise.all([getJson(base + spec.index), download(base + spec.frames)]);
  const buffer = await blob.arrayBuffer();
  if (info.fps !== fps || !Array.isArray(info.frames) ||
      (count !== null && info.frames.length !== count) || info.frames.length < 2 || info.frames.length > 200 ||
      !info.frames.every((row) => clipNumbers(row, 2) && row.every(Number.isInteger) &&
        row[0] >= 0 && row[1] > 0 && row[0] + row[1] <= buffer.byteLength)) {
    throw new Error("The silent frame bank does not match the motion loop.");
  }
  const stride = spec.anchorEvery;
  if (stride != null && (info.frames.length % stride !== 0 ||
      info.frames.some(([at, size], i) => i % stride === 0 &&
        (at !== info.frames[0][0] || size !== info.frames[0][1])))) {
    throw new Error("Moving idle exits must reference the same anchor picture.");
  }
  if (spec.randomBlink) {
    const { every, offset, length } = spec.randomBlink;
    if (info.frames.length !== 2 * stride || info.frames.slice(stride).some(([at, size], i) => {
      const variant = i % every >= offset && i % every < offset + length;
      const same = at === info.frames[i][0] && size === info.frames[i][1];
      return variant === same;
    })) throw new Error("Idle blink variants must match their body frames outside the blink slots.");
  }
  const unique = new Map(info.frames.map(([at, size]) => [`${at}:${size}`, [at, size]]));
  const entries = [...unique.entries()];
  const settled = await Promise.allSettled(entries.map(([, [at, size]]) => createImageBitmap(
    new Blob([new Uint8Array(buffer, at, size)], { type: "image/jpeg" }), DATA_IMAGE)));
  const decoded = new Map();
  for (let i = 0; i < entries.length; i++) {
    if (settled[i].status === "fulfilled") decoded.set(entries[i][0], settled[i].value);
  }
  const frames = info.frames.map(([at, size]) => decoded.get(`${at}:${size}`));
  const failure = settled.find((item) => item.status === "rejected");
  if (failure) {
    closeAll(decoded.values());
    throw failure.reason;
  }
  return frames;
}

async function loadFillers(base, spec, size, loopCount, loopFps, restCount) {
  const [info, blob] = await Promise.all([getJson(base + spec.index), download(base + spec.frames)]);
  const buffer = await blob.arrayBuffer();
  const found = checkFillers(info, buffer.byteLength, size, loopCount, loopFps, restCount);
  if (!found) throw new Error("the filler phrases do not match the speaking frames");
  return Object.assign(found, { buffer });
}

let fillerCanvas = null;
// One picture of a filler: the speaking-loop frame with the part the mouth changes drawn over it, and the rest frame
// the filler began on over that at 1 - `fade`.
function composeFiller(frame, crop, box, rest, fade) {
  if (!fillerCanvas || fillerCanvas.width !== frame.width || fillerCanvas.height !== frame.height) {
    fillerCanvas = document.createElement("canvas");
    fillerCanvas.width = frame.width;
    fillerCanvas.height = frame.height;
  }
  const context = fillerCanvas.getContext("2d");
  context.globalAlpha = 1;
  context.drawImage(frame, 0, 0);
  context.drawImage(crop, box[0], box[1]);
  if (rest && fade < 1) {
    context.globalAlpha = 1 - fade;
    context.drawImage(rest, 0, 0);
    context.globalAlpha = 1;
  }
  return fillerCanvas;
}

// Says the next filler phrase while the reply is made, lip-synced in the speaking loop, starting on a rest frame it was
// made for.
async function playFiller(t) {
  if (!fillers || !clips || !clips.restFrames || !playback.context) return;
  const filler = fillers.fillers[nextFiller++ % fillers.fillers.length];
  const length = Math.max(...filler.takes.map((take) => take.rows.length)) / fillers.fps;
  const plan = clips.planFiller(seconds(), 0.45, filler.takes, length);
  if (!plan) return;
  const rows = plan.take.rows;
  let images;
  try {
    images = await Promise.all(rows.map(([, at, size]) => createImageBitmap(
      new Blob([new Uint8Array(fillers.buffer, at, size)], { type: "image/jpeg" }), DATA_IMAGE)));
  } catch (error) {
    console.warn("A filler phrase could not be decoded:", error);
    clips.stopFiller();
    return;
  }
  // Too late to start in step with its rest frame, or the turn is over: the reply comes without it.
  if (turn !== t || !clips || clips.filler === null || seconds() > plan.at - 0.04) {
    closeAll(images);
    if (clips && turn === t) clips.stopFiller();
    return;
  }
  const pcm = new Int16Array(fillers.buffer.slice(filler.audio[0], filler.audio[0] + filler.audio[1]));
  const buffer = playback.context.createBuffer(1, pcm.length, filler.rate);
  const channel = buffer.getChannelData(0);
  for (let i = 0; i < pcm.length; i++) channel[i] = pcm[i] / 32768;
  const source = playback.context.createBufferSource();
  source.buffer = buffer;
  source.connect(playback.context.destination);
  source.start(Math.max(playback.context.currentTime, playback.audibleTime + plan.at - seconds()));
  source.onended = () => source.disconnect();
  t.filler = source;
  clips.startFiller({ rows, images, box: fillers.box, fps: fillers.fps, fade: fillers.fade });
}

function stopFiller(t) {
  if (t.filler) {
    try { t.filler.stop(); } catch { /* already ended */ }
    t.filler = null;
  }
  if (clips && clips.filler) clips.stopFiller();
}

// Ordinary clip failures retain the legacy fallback; a continuous bank failure must be visible.
async function startClips(base, manifest, photoRows) {
  let reels = null;
  try {
    if (manifest.continuous) {
      if (!dubbed) throw new Error("Continuous motion requires MuseTalk.");
      const clip = Object.values(manifest.clips)[0];
      const settled = await Promise.allSettled([
        loadQuietBank(base, manifest.continuous, clip.fps, Math.round(clip.fps * clip.duration)),
        manifest.continuous.rest ? loadQuietBank(base, manifest.continuous.rest, clip.fps) : Promise.resolve(null),
      ]);
      const failure = settled.find((item) => item.status === "rejected");
      if (failure) {
        for (const item of settled) if (item.status === "fulfilled" && item.value) closeAll(item.value);
        throw failure.reason;
      }
      const [frames, restFrames] = settled.map((item) => item.value);
      if (restFrames && manifest.continuous.rest.blink &&
          (manifest.continuous.rest.blink.length >= restFrames.length ||
           manifest.continuous.rest.blink.some((frame) => frame >= restFrames.length))) {
        closeAll([...frames, ...restFrames]);
        throw new Error("The idle blink frames do not match the silent frame bank.");
      }
      if (restFrames && restFrames.some((frame) => frame.width !== frames[0].width || frame.height !== frames[0].height)) {
        closeAll([...frames, ...restFrames]);
        throw new Error("The idle pictures must match the speaking frame size.");
      }
      compositor.loadClips(manifest);
      clips = new ClipStage({ manifest, compositor, frames, restFrames, now: seconds(), compose: composeFiller });
      window.photoStageClips = clips;
      if (restFrames && manifest.continuous.fillers) {
        loadFillers(base, manifest.continuous.fillers, [frames[0].width, frames[0].height], frames.length, clip.fps,
          manifest.continuous.rest.anchorEvery)
          .then((found) => { fillers = found; })
          .catch((error) => console.warn("The filler phrases could not be loaded, so replies start without them:", error));
      }
      return;
    }
    if (dubbed && !(manifest = quietClipManifest(manifest))) throw new Error("the pack has no clip to play around the speech");
    reels = new ClipReels(manifest, {
      base,
      download: (url) => download(url),
      makeVideo: clipVideo,
      captureFrame: clipSnapshot,
      url: URL,
      canPlay: (type) => document.createElement("video").canPlayType(type),
      makeBlob: (blob, type) => new Blob([blob], { type }),
    });
    await reels.load();
    const director = new ClipDirector(manifest.clips, {
      ready: (name) => reels.ready(name), failed: (name) => reels.failed(name), motion: options.motion,
      idle: manifest.mode === "avatar",
    });
    if (!director.usable) throw new Error("none of the clips could be loaded");
    compositor.loadClips(manifest);
    const tone = photoRows ? await fitClipTone(reels, manifest, photoRows) : null;
    if (tone) compositor.setClipTone(tone.matrix, tone.offset);
    clips = new ClipStage({ manifest, reels, director, compositor });
    clips.setState(conversation.state, seconds());
    reels.background.then(() => {
      if (clips && clips.director === director) director.hint(seconds());
    });
    window.photoStageClips = clips;
  } catch (error) {
    if (manifest.continuous) throw error;
    console.warn("The body clips could not be started, so the body moves as it does without them:", error);
    if (reels) reels.dispose();
    compositor.clearClips();
    clips = null;
  }
}

function buildText(build) {
  const step = build.step && build.steps ? ` Step ${build.step} of ${build.steps}: ${build.label || "working"}.` : "";
  const minutes = Number.isFinite(build.eta) && build.eta > 0 ? ` About ${Math.max(1, Math.round(build.eta / 60))} min left.` : "";
  return `Avatar creation in progress…${step}${minutes}`;
}

// A new avatar's animations take a while to make: the page shows how far they are, and goes on once they are there.
async function waitUntilBuilt(avatar) {
  $("loading-spinner").hidden = false;
  while (avatar.build && avatar.build.state === "building") {
    $("loading-text").textContent = buildText(avatar.build);
    $("loading-bar").hidden = false;
    $("loading-fill").style.width = `${Math.round(avatar.build.percent || 0)}%`;
    await wait(5000);
    try {
      avatar = await getJson(`/api/avatars/${key}`);
    } catch (error) {
      if (error.status === 404) throw new Error("This avatar was deleted while it was being created.");
    }
  }
  $("loading-bar").hidden = true;
  $("loading-fill").style.width = "0%";
  $("loading-text").textContent = "Loading the avatar…";
  if (avatar.build && avatar.build.state === "failed") {
    console.warn("The avatar's animations could not be made:", avatar.build.error);
  }
  return avatar;
}

async function waitUntilReady() {
  while (!failed) {
    try {
      const status = await (await fetch("/status")).json();
      if (status.ready) return;
      setStatus(status.message || "Warming up…");
    } catch {
      setStatus("Waiting for the avatar service…");
    }
    await wait(3000);
  }
}

async function loadAvatar() {
  const base = `/assets/${key}/`;
  const [rig, photo, ref, detail, bias, body, pack] = await Promise.all([
    getJson(`${base}rig.json`),
    bitmap(`${base}photo.png`), bitmap(`${base}ref.png`), bitmap(`${base}detail.png`, false), bitmap(`${base}bias.png`),
    loadBody(base),
    loadClipPack(base),
  ]);
  const manifest = pack && parseClipManifest(pack, [photo.width, photo.height]);
  // The part of the photo that the clips show, kept small, to fit the clips' colours to before the photo is let go.
  let photoRows = null;
  if (manifest) {
    try {
      const [x1, y1, x2, y2] = manifest.box;
      photoRows = colourSample(photo, [x1, y1, x2 - x1, y2 - y1]);
    } catch (error) {
      console.warn("The colours of the body clips will not be fitted to the photo:", error);
    }
  }
  const canvas = $("stage-canvas");
  compositor = new Compositor(canvas, address, dubbed);
  compositor.load({ photo, rig, ref, detail, bias, body });
  if (dubbed) compositor.setPicture("photo");
  for (const image of [photo, ref, detail, bias, ...bodyImages(body)]) if (image) image.close();
  motion = new BodyMotion(body && body.rig, { intensity: options.motion, seed: Date.now() });
  canvas.hidden = false;
  $("loading").hidden = true;
  requestAnimationFrame(render);
  // The still photo is already on screen; the animated head fades in once its frames are decoded.
  setStatus("Preparing the animated head…");
  if (!(dubbed && manifest && manifest.continuous)) idle = await loadIdle(base);
  if (manifest) {
    setStatus("Preparing the body clips…");
    await startClips(base, manifest, photoRows);
  } else if (pack) {
    console.warn("The body clips do not fit this avatar's photo, so the body moves as it does without them.");
  }
}

function render(now) {
  requestAnimationFrame(render);
  const dt = Math.min(0.1, (now - lastTick) / 1000);
  lastTick = now;
  let wanted = 0;
  if (turn) {
    const { image, guide, finished } = playback.poll();
    if (image) {
      compositor.setFrame("live", image, guide);
      image.close();
      if (guide) guide.close();
      // The first live frame is due with the first sound: playback schedules both on the audio clock.
      if (!shownLive) conversation.speaking(now / 1000);
      shownLive = true;
      if (turn.reply && !turn.revealed) reveal(turn);
    }
    wanted = shownLive ? 1 : 0;
    if (playback.phase !== lastPhase) {
      lastPhase = playback.phase;
      if (playback.phase === "streaming") setStatus("Speaking.");
      else if (playback.phase === "buffering" && shownLive) setStatus("Buffering…");
    }
    if (finished) {
      finish(turn, "done", "Listening.");
      wanted = 0;
    }
  }
  const step = dt / CROSSFADE_SECONDS;
  mix = clips && clips.frames ? wanted : mix + Math.max(-step, Math.min(step, wanted - mix));
  if (idle || shownLive) head = Math.min(1, head + dt / HEAD_FADE_SECONDS);
  if (idle && mix < 1) {
    const frame = idle.at(now / 1000);
    if (frame !== shownIdle) {
      compositor.setFrame("idle", frame.image, frame.guide);
      shownIdle = frame;
    }
  }
  let weight = 0;
  withClips((stage) => {
    if (playback.total > 0) stage.length(playback.total);
    if (playback.active) stage.progress(playback.position);
    weight = stage.frames && mix === 1 ? stage.weight : stage.update(now / 1000);
  });
  if (motion) {
    try {
      const pose = motion.tick(dt, playback.context ? playback.context.currentTime : 0, Boolean(turn && shownLive));
      compositor.setPose(clips && clips.manifest.mode === "avatar" ? null : clips ? underClips(pose, weight) : pose);
    } catch (error) {
      stopMotion(error);
    }
  }
  compositor.draw(mix, head);
}

// What was asked and what the avatar answers, under the avatar. The answer appears when the avatar starts to say it.
function showAsked(t) {
  if (!options.reply) return;
  $("exchange").hidden = false;
  $("you").textContent = t.text;
  $("reply").textContent = "";
  $("reply").classList.add("thinking");
}

function reveal(t) {
  t.revealed = true;
  if (!options.reply) return;
  $("reply").textContent = t.reply;
  $("reply").classList.remove("thinking");
}

function fitMessage() {
  const box = $("message");
  box.style.height = "auto";
  box.style.height = `${Math.min(box.scrollHeight + 2, 132)}px`;
}

function finish(t, outcome, message) {
  if (turn !== t) return;
  if (t.loop && shownLive && clips && clips.frames) {
    clips.resume(t.loop.frame + Math.floor(playback.position * t.loop.fps + 1e-6), seconds());
  }
  turn = null;
  lastPhase = "";
  if (outcome !== "done") stopFiller(t);
  if (t.ws && t.ws.readyState < 2) t.ws.close();
  playback.reset();
  if (motion) motion.hush();
  withClips((stage) => (outcome === "done" ? stage.finish() : stage.stop()));
  // A failed message goes back in the box, and the viewer is then taken to be still typing it.
  if (outcome === "error" && !$("message").value && t.text) {
    $("message").value = t.text;
    fitMessage();
    conversation.input("typing", true, seconds());
  }
  conversation.end(seconds(), outcome);
  if (t.reply && !t.revealed) reveal(t);
  $("reply").classList.remove("thinking");
  if (outcome === "done" && t.mode === "chat" && t.reply) {
    history = [...history, { role: "user", content: t.text }, { role: "assistant", content: t.reply }].slice(-8);
  }
  $("send").disabled = false;
  $("stop").disabled = true;
  setStatus(message, outcome === "error" ? "error" : "info");
}

async function receive(t, message) {
  if (turn !== t) return;
  if (message.kind === "error") throw new Error(message.message);
  if (message.kind === "stage") {
    setStatus("Thinking…");
  } else if (message.kind === "speech") {
    t.reply = message.reply;
    compositor.setPicture(message.picture || null);
    withClips((stage) => stage.speak(message.reply));
    setStatus("Preparing voice…");
  } else if (message.kind === "motion") {
    if (!clips || !clips.frames || !clips.manifest.clips[message.clip] ||
        message.frames !== clips.frames.length || message.fps !== clips.loopClip.fps ||
        !Number.isFinite(message.seconds) || message.seconds <= 0) {
      throw new Error("The live motion pack changed. Please reload the player.");
    }
    const start = clips.arm(seconds(), playback.preparationDelay(message.seconds));
    const deadline = playback.audibleTime + start.at - seconds();
    const period = start.period;
    playback.alignStart = (earliest) => {
      const at = deadline + Math.max(0, Math.ceil((earliest - deadline) / period)) * period;
      playback.stats.phaseWaitSeconds = at - earliest;
      return at;
    };
    t.loop = { frame: start.frame, fps: message.fps };
    t.ws.send(JSON.stringify({ type: "phase", frame: start.frame }));
  } else if (message.kind === "segment") {
    if (await playback.enqueue(message) && turn === t) {
      t.ws.send(JSON.stringify({ type: "ack", index: message.index }));
    }
  } else if (message.kind === "done") {
    t.metrics = message;
    playback.done = true;
    playback.schedule();
  }
}

// Starts a turn (THINKING). The thinking clip covers everything before the reply's first sound, a recording's
// transcription included, so that no wait shows as a still avatar. Resolves to the turn, or null if it cannot start.
async function beginTurn(text) {
  try {
    await playback.start();
  } catch (error) {
    setStatus(error.message, "error");
    return null;
  }
  window.photoStageStats = playback.stats;
  if (motion) motion.hush();
  const now = seconds();
  conversation.ask(now);
  withClips((stage) => stage.ask(now));
  const t = { ws: null, text, mode: $("mode").value, reply: "", revealed: false, filler: null };
  turn = t;
  if (t.mode === "chat") playFiller(t).catch((error) => console.warn("The filler phrase failed:", error));
  shownLive = false;
  $("send").disabled = true;
  $("stop").disabled = false;
  setStatus("Thinking…");
  return t;
}

async function send(event) {
  event.preventDefault();
  const text = $("message").value.trim();
  if (!text || turn || !compositor) return;
  const t = await beginTurn(text);
  if (!t) return;
  $("message").value = "";
  fitMessage();
  conversation.input("typing", false, seconds());
  converse(t);
}

// Sends what was asked and plays the reply as it streams in.
function converse(t) {
  const ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/turn/${key}`);
  t.ws = ws;
  showAsked(t);
  ws.onopen = () => ws.send(JSON.stringify({ text: t.text, mode: t.mode, history, clips: Boolean(clips),
    continuous: Boolean(clips && clips.frames), motion: options.motion }));
  let chain = Promise.resolve();
  ws.onmessage = (e) => {
    chain = chain.then(() => receive(t, JSON.parse(e.data))).catch((error) => finish(t, "error", error.message));
  };
  ws.onclose = () => {
    chain.then(() => {
      if (turn === t && !playback.done) finish(t, "error", "Connection closed before the reply was complete.");
    });
  };
  ws.onerror = () => finish(t, "error", "Cannot connect to the avatar service.");
}

$("compose").onsubmit = send;
$("message").addEventListener("input", () => {
  fitMessage();
  conversation.input("typing", $("message").value.trim() !== "", seconds());
});
$("message").addEventListener("keydown", (event) => {
  // Enter sends, Shift+Enter starts a new line; a key that only confirms an input method's text does neither.
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    $("send").click();
  }
});
$("stop").onclick = () => turn && finish(turn, "stopped", "Stopped. Listening.");

const fullscreenRoot = $("player");
const requestFullscreen = fullscreenRoot.requestFullscreen || fullscreenRoot.webkitRequestFullscreen;
$("fullscreen").onclick = () => {
  const open = document.fullscreenElement || document.webkitFullscreenElement;
  const exit = document.exitFullscreen || document.webkitExitFullscreen;
  const result = open ? exit.call(document) : requestFullscreen.call(fullscreenRoot);
  if (result && result.catch) result.catch((error) => setStatus(error.message, "error"));
};

// A recording that ends is answered at once: the avatar thinks while it is transcribed, and the transcript is sent. While
// the avatar cannot take a message (still loading, or answering one typed meanwhile), the transcript is left in the box.
async function answerRecording(blob) {
  const ready = Boolean(compositor) && !turn && !$("send").disabled;
  const t = ready ? await beginTurn("") : null;
  conversation.input("voice", false, seconds());
  if (ready && !t) return;
  try {
    const form = new FormData();
    form.append("audio", blob, "recording");
    const response = await fetch("/transcribe", { method: "POST", body: form }), body = await response.json();
    if (!response.ok) throw new Error(body.detail);
    const text = String(body.text || "").trim();
    if (!t) {
      $("message").value = text;
      fitMessage();
      conversation.input("typing", text !== "", seconds());
      setStatus("Transcript ready. Press Send.");
      return;
    }
    if (turn !== t) return;
    if (!text) {
      finish(t, "error", "No speech was heard. Try again.");
      return;
    }
    t.text = text;
    converse(t);
  } catch (error) {
    if (t) finish(t, "error", error.message);
    else setStatus(error.message, "error");
  }
}

$("mic").onclick = async () => {
  if (!navigator.mediaDevices || !window.MediaRecorder) {
    setStatus("Microphone recording requires HTTPS or localhost. Text works over LAN HTTP.", "error");
    return;
  }
  if (recorder) { recorder.stop(); return; }
  // Speaking to the avatar while it answers interrupts it, as it would a person.
  if (turn) finish(turn, "stopped", "Stopped.");
  playback.unlock();
  let stream = null;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const rec = new MediaRecorder(stream), chunks = [];
    recorder = rec;
    rec.ondataavailable = (e) => chunks.push(e.data);
    const timer = setTimeout(() => { if (rec.state === "recording") rec.stop(); }, 45000);
    rec.onstop = () => {
      clearTimeout(timer);
      stream.getTracks().forEach((track) => track.stop());
      recorder = null;
      $("mic").textContent = "Record";
      answerRecording(new Blob(chunks, { type: rec.mimeType }));
    };
    rec.start();
    conversation.input("voice", true, seconds());
    $("mic").textContent = "End recording";
    setStatus("Recording…");
  } catch (error) {
    if (stream) stream.getTracks().forEach((track) => track.stop());
    conversation.input("voice", false, seconds());
    setStatus(error.message, "error");
  }
};

// Show or hide each part of the page as the options say, and fix the mode that new messages start in.
function applyOptions() {
  $("dock").hidden = !(options.chat || options.status);
  $("compose").hidden = !options.chat;
  $("mic").hidden = !options.mic;
  $("mode").hidden = !options.modes;
  $("stop").hidden = !options.stop;
  $("home").hidden = !options.home;
  $("fullscreen").hidden = !(options.fullscreen && requestFullscreen);
  $("mode").value = options.mode;
}

(async () => {
  let avatar, schema, service;
  try {
    [schema, avatar, service] = await Promise.all([getJson("/api/options").then((body) => body.options), getJson(`/api/avatars/${key}`),
      getJson("/status")]);
  } catch (error) {
    failLoading(error.status === 404 ? "This avatar does not exist. It may have been deleted."
      : error.status ? error.message : "Cannot reach the avatar service. Check the network and reload the page.", true);
    return;
  }
  document.title = `${avatar.name} | PhotoStage`;
  if (avatar.build && avatar.build.state === "building") {
    try {
      avatar = await waitUntilBuilt(avatar);
    } catch (error) {
      failLoading(error.message, true);
      return;
    }
  }
  dubbed = service.renderer === "musetalk";
  options = activeOptions(schema, resolveOptions(schema, avatar.options, address));
  applyOptions();
  try {
    await Promise.all([loadAvatar(), waitUntilReady()]);
    $("send").disabled = false;
    setStatus("Listening.");
  } catch (error) {
    if (compositor) setStatus(error.message, "error");
    else failLoading(error.message, false);
  }
})();
