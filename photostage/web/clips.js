"use strict";
// Body clips. A clip is a few seconds of the avatar's body, from under the chin down, gesturing: video that a model made
// from the photo and that sits in the avatar's folder (clips.json and clip-*.mp4/.webm, see photostage/clippack.py). While
// a reply is being prepared a clip of the avatar thinking plays on a loop, so that nothing ever stands still; while the
// reply is spoken, gestures that suit what is being said follow one another. The live head is drawn over all of it and moves
// with the clip's neck and shoulders (each clip carries that motion as a track), so that head and body read as one person.
// Full-avatar packs instead track the face and supply the whole person, including the idle face and blinks. The compositor
// reenacts only the facial interior during speech; listening clips continue between turns.
//
// This file holds the director (which clip plays when: plain logic that needs no browser and is tested on its own), the
// pool of <video> elements that plays the clips, and the stage that ties them to the compositor. Everything is inert for an
// avatar without a clip pack, and an avatar whose clips fail goes on with the procedural body (see body.js).

const CLIP_FADE = 0.3; // a clip fades in and out over this long, from and to the photo: both are the same pose
const CLIP_FAST_FADE = 0.15; // ... and over this long when it has to go at once
const CLIP_MAX_WAIT = 2; // a clip that is let go waits for its next exit if that is no later than this
const CLIP_HANDOFF_WAIT = 0.6; // a thinking clip gives way to speech within this long
const CLIP_START_TIMEOUT = 2.5; // a video that has shown no frame this long after it was started is given up
const CLIP_STALL = 1; // and one that has shown no new frame for this long
const CLIP_GAP = [0.25, 1.2]; // the body rests this long between two gestures at the normal amount of motion
const CLIP_MIN_TAIL = 1.2; // subsequent gestures need this much speech left
const CLIP_FIRST_TAIL = 0.35; // a short reply can still open with one short gesture
const CLIP_OVERRUN = 1.5; // a gesture may last this much longer than the speech that is left
const CLIP_RECENT = 2; // the latest gestures are not repeated
const CLIP_THINK_SWAP = 7; // a thinking clip that has played this long gives way to another one, if there is one
const CLIP_LOOKAHEAD = 0.4; // a gesture suits what is said this long after it begins
const CLIP_SLOTS = 3; // video elements in the pool: one plays, the others hold the clips to play next
const CLIP_TONE_SIDE = 64; // the pictures the tone of a clip is fitted on are this many pixels wide
const CLIP_ROLES = ["idle", "listen", "think", "talk"];
const CLIP_QUIET = {
  idle: ["idle", "think", "listen"],
  listen: ["listen", "think", "idle"],
  think: ["think", "listen", "idle"],
};

const clipClamp = (x, low, high) => Math.min(high, Math.max(low, x));
const clipSmooth = (x) => {
  const t = clipClamp(x, 0, 1);
  return t * t * (3 - 2 * t);
};
const clipNumbers = (list, count) => Array.isArray(list) && list.length === count && list.every((x) => typeof x === "number" && isFinite(x));
const clipAffine = (pose) => clipNumbers(pose, 6) && pose[0] * pose[4] - pose[1] * pose[3] > 0.1;

// One clip of a manifest, checked and completed; null where it cannot be played. `exits` are the times (in seconds) at which
// the pose is close to rest, where the clip can end without a visible change: the end of the clip is always one. A clip that
// does not loop never shows its last frame twice, so it ends one frame before its duration.
function checkClip(spec) {
  if (!spec || CLIP_ROLES.indexOf(spec.role) < 0 || typeof spec.mood !== "string") return null;
  const { duration, fps } = spec;
  if (!(duration > 0) || !(fps > 0) || !spec.files || typeof spec.files !== "object") return null;
  const loop = Boolean(spec.loop);
  const last = loop ? duration : duration - 1 / fps;
  if (!(last > 0)) return null;
  const frames = Math.round(duration * fps);
  const track = Array.isArray(spec.track) && spec.track.length ? spec.track : null;
  if (track && (track.length !== frames || !track.every((row) => clipNumbers(row, 4)))) return null;
  const faceTrack = spec.faceTrack || null;
  if (faceTrack && (!Array.isArray(faceTrack) || faceTrack.length !== frames || !faceTrack.every(clipAffine))) return null;
  const exits = (Array.isArray(spec.exits) ? spec.exits : []).filter((t) => typeof t === "number" && t > 0 && t < last);
  exits.push(last);
  exits.sort((a, b) => a - b);
  return {
    role: spec.role, mood: spec.mood, duration, fps, loop, last, track, faceTrack, files: spec.files,
    exits: exits.filter((t, i) => !i || t - exits[i - 1] > 1e-6),
  };
}

// clips.json, checked: null where the pack cannot be used (another picture size than the avatar's, nothing playable in it).
function parseClipManifest(spec, photoSize) {
  if (!spec || spec.version !== 1 || !spec.clips || typeof spec.clips !== "object") return null;
  if (!clipNumbers(spec.size, 2) || !clipNumbers(spec.box, 4) || !clipNumbers(spec.pivot, 2) || !clipNumbers(spec.follow, 2)) return null;
  if (photoSize && (spec.size[0] !== photoSize[0] || spec.size[1] !== photoSize[1])) return null;
  if (!(spec.box[2] > spec.box[0] && spec.box[3] > spec.box[1])) return null;
  const mode = spec.mode || "body";
  if (mode !== "body" && mode !== "avatar") return null;
  if (spec.facePose && !clipAffine(spec.facePose)) return null;
  const faceAnchors = spec.faceAnchors == null ? null : spec.faceAnchors;
  if (faceAnchors !== null && (mode !== "avatar" || !spec.facePose || !Array.isArray(faceAnchors) ||
      faceAnchors.length !== 13 || !faceAnchors.every((p) => clipNumbers(p, 2) && p.every((v) => v >= 0 && v <= 1)))) return null;
  const faceMode = spec.faceMode || "full", mouth = spec.mouth;
  if (!["full", "mouth", "head"].includes(faceMode)) return null;
  if (faceMode === "head" && (mode !== "avatar" || spec.facePose || faceAnchors !== null)) return null;
  if (faceMode === "mouth" && (mode !== "avatar" || !spec.facePose || !mouth ||
      !clipNumbers(mouth.box, 4) || !clipNumbers(mouth.center, 2) ||
      !(mouth.box[0] < mouth.center[0] && mouth.center[0] < mouth.box[2] &&
        mouth.box[1] < mouth.center[1] && mouth.center[1] < mouth.box[3]))) return null;
  if (mode === "avatar" && (!clipNumbers(spec.face, 4) || spec.box[0] !== 0 || spec.box[1] !== 0 ||
      spec.box[2] !== spec.size[0] || spec.box[3] !== spec.size[1] ||
      !(spec.face[2] > spec.face[0] && spec.face[3] > spec.face[1]))) return null;
  const clips = {};
  for (const name of Object.keys(spec.clips)) {
    const clip = checkClip(spec.clips[name]);
    if (clip && (!spec.facePose || clip.faceTrack) && (faceMode !== "head" || clip.track)) clips[name] = clip;
  }
  if (!Object.keys(clips).length) return null;
  const continuous = spec.continuous || null;
  const bankFiles = (bank) => bank && [bank.index, bank.frames].every((name) =>
    typeof name === "string" && /^[a-zA-Z0-9][a-zA-Z0-9_.-]*$/.test(name));
  if (continuous && (mode !== "avatar" || Object.keys(clips).length !== 1 ||
      !Object.values(clips).every((clip) => clip.role === "idle" && clip.loop) ||
      !bankFiles(continuous))) return null;
  if (continuous && continuous.rest != null) {
    const rest = continuous.rest, clip = Object.values(clips)[0];
    if (!bankFiles(rest) || !Number.isInteger(rest.anchor) || rest.anchor < 0 ||
        rest.anchor >= Math.round(clip.fps * clip.duration)) return null;
    if (rest.blink != null && (!Array.isArray(rest.blink) ||
        !rest.blink.every((frame) => Number.isInteger(frame) && frame >= 0 && frame < 200) ||
        new Set(rest.blink).size !== rest.blink.length)) return null;
    if (rest.anchorEvery != null && (rest.blink != null || !Number.isInteger(rest.anchorEvery) ||
        rest.anchorEvery < 1 || rest.anchorEvery > 200)) return null;
    if (rest.randomBlink != null) {
      const { every, offset, length } = rest.randomBlink;
      if (!rest.anchorEvery || ![every, offset, length].every(Number.isInteger) ||
          every < 2 || every > clip.fps * 6 || rest.anchorEvery % every !== 0 ||
          offset < 1 || length < 1 || offset + length > every) return null;
    }
  }
  // Filler phrases are an extra: a pack whose fillers do not fit plays on without them.
  let packContinuous = continuous;
  if (continuous && continuous.fillers != null) {
    const rest = continuous.rest;
    if (!bankFiles(continuous.fillers) || !rest || !rest.anchorEvery) {
      packContinuous = Object.assign({}, continuous);
      delete packContinuous.fillers;
    }
  }
  return { mode, face: spec.face, facePose: spec.facePose, faceAnchors, faceMode, mouth, continuous: packContinuous,
    size: spec.size, box: spec.box, pivot: spec.pivot, follow: spec.follow, types: spec.types || {}, clips };
}

// The pack without its gestures, for a renderer that pictures the speech itself (MuseTalk dubs the reply into the gesture
// clips and streams whole pictures): the player then plays only what comes before and after the speech. Null where nothing
// would be left.
function quietClipManifest(manifest) {
  const clips = {};
  for (const name of Object.keys(manifest.clips)) if (manifest.clips[name].role !== "talk") clips[name] = manifest.clips[name];
  return Object.keys(clips).length ? Object.assign({}, manifest, { clips }) : null;
}

function clipFace(clip, time) {
  if (!clip || !clip.faceTrack) return null;
  const at = clipClamp(time * clip.fps, 0, clip.faceTrack.length - 1);
  const i = Math.floor(at), j = Math.min(clip.faceTrack.length - 1, i + 1);
  return clip.faceTrack[i].map((v, k) => v + (clip.faceTrack[j][k] - v) * (at - i));
}

// How the clip's neck and shoulders are placed at media time `time`: {scale, turn (radians), x, y (photo pixels)} against the
// photo, from the clip's per-frame track.
function clipHead(clip, time) {
  const track = clip && clip.track;
  if (!track) return { scale: 1, turn: 0, x: 0, y: 0 };
  const at = clipClamp(time * clip.fps, 0, track.length - 1);
  const i = Math.floor(at), j = Math.min(track.length - 1, i + 1), k = at - i;
  const blend = (n) => track[i][n] + (track[j][n] - track[i][n]) * k;
  return { scale: blend(0), turn: blend(1), x: blend(2), y: blend(3) };
}

// The first exit at least `margin` seconds after `elapsed`, on the clip's whole timeline (a loop's exits repeat).
function clipNextExit(clip, elapsed, margin) {
  const turns = clip.loop ? Math.floor(elapsed / clip.duration) : 0;
  let best = clip.loop ? Infinity : clip.last;
  for (let k = turns; k <= (clip.loop ? turns + 1 : turns); k++) {
    for (const exit of clip.exits) {
      const t = k * clip.duration + exit;
      if (t >= elapsed + margin && t < best) best = t;
    }
  }
  return best;
}

// The reply as sentences with the share of the speech each takes, taken as the share of its words.
function clipSentences(text) {
  const parts = String(text || "").match(/[^.!?\u2026\n]+[.!?\u2026]*["')\]]*/g) || [];
  const sentences = [];
  let total = 0;
  for (const part of parts) {
    const words = part.trim().split(/\s+/).filter(Boolean).length;
    if (!words || !/[\w\u00c0-\uffff]/.test(part)) continue;
    sentences.push({ text: part.trim(), words, from: total, to: total + words });
    total += words;
  }
  for (const sentence of sentences) {
    sentence.from /= total;
    sentence.to /= total;
  }
  return sentences;
}

const CLIP_GREETING = /^\W*(hello|hi|hey|welcome|greetings|good (morning|afternoon|evening)|nice to meet)\b/i;
const CLIP_QUESTION = /^\W*(?:(?:could|would|can|will)\s+you|(?:do|does|did|have|has)\s+(?:you|we|they)|(?:what|where|when|why|who|which|how)\s+(?:is|are|was|were|do|does|did|can|could|would|should|will|have|has))\b/i;
// The kind of gesture a sentence calls for.
function clipMood(sentence, index) {
  const { text, words } = sentence;
  const normalized = text.toLowerCase().replace(/[’']/g, "'");
  if (index === 0 && CLIP_GREETING.test(text)) return "greet";
  if (/\?["')\]]*$/.test(text) || CLIP_QUESTION.test(text)) return "ask";
  if (/!["')\]]*$/.test(text)) return "emphasis";
  if (/\b(thank you|i appreciate|i'm sorry|i am sorry|my apologies|i understand|that sounds difficult)\b/.test(normalized)) return "calm";
  if (/\b(absolutely|congratulations|crucial|delighted|excited|great news|importantly|key point|wonderful)\b/.test(normalized)) return "emphasis";
  if (/\b(because|for example|for instance|first|finally|in summary|next|the reason|to begin with)\b/.test(normalized)) return "explain";
  if (words >= 18) return "explain";
  if (words <= 6) return "calm";
  return index % 2 ? "emphasis" : "explain";
}

// What to take where a pack has no clip of the mood wanted, best first.
const CLIP_SIMILAR = {
  greet: ["greet", "explain", "calm"], ask: ["ask", "explain", "calm"], emphasis: ["emphasis", "explain", "calm"],
  explain: ["explain", "emphasis", "calm"], calm: ["calm", "explain", "emphasis"],
};

// Decides which clip plays when, from what the player tells it (a message was sent, the reply's words and length are known,
// the speech has begun, the reply is complete, the user stopped it) and from what the video it commands is showing. It is
// told the time on every tick, in seconds, and answers {name, weight, time, commands}: the clip to draw, how much of it
// (0-1), the media time of the frame shown, and commands for the pool of videos ({type: "prepare" | "play" | "park"}).
//
// The clips of one body are strung together through the photo: every clip begins and ends in the pose of the photo, so
// a clip fades in from the photo and out into it, and the next one begins from there. A thinking clip plays from the moment
// the message is sent until the speech begins; it loops, and is let go at one of its exits (poses close to rest). Gestures
// follow, one at a time with a rest between them, each chosen for the sentence being said, and none begins when the speech
// is almost over. What a clip shows is the clock: the video's own time, not the page's, drives its fades, so a video that
// is late is still faded in step with its frames. Speech and gesture are only loosely tied (a gesture is a gesture whether
// it begins a little earlier or later); the head and the body are tied tightly, by the frame shown.
class ClipDirector {
  constructor(clips, { ready = () => true, failed = () => false, random = Math.random, motion = 1, idle = false } = {}) {
    this.clips = clips;
    this.ready = ready;
    this.failed = failed;
    this.random = random;
    this.idle = idle;
    this.state = "IDLE";
    this.pace = clipClamp(Number.isFinite(motion) ? motion : 1, 0.25, 2);
    this.broken = new Set();
    this.recent = [];
    this.lastThink = null;
    this.turn = null;
    this.cur = null;
    this.restUntil = 0;
    this.lastTick = null;
    this.queue = [];
    this.hint(0, true);
  }

  // Names of the clips that pass `test` and have not failed.
  names(test) {
    return Object.keys(this.clips).filter((name) => test(this.clips[name]) && !this.broken.has(name) && !this.failed(name));
  }

  get usable() {
    return this.names(() => true).length > 0;
  }

  choose(names) {
    return names.length ? names[Math.floor(this.random() * names.length) % names.length] : null;
  }

  setState(state, now) {
    if (!["IDLE", "LISTENING", "THINKING", "SPEAKING"].includes(state)) throw new Error(`Unknown conversation state: ${state}`);
    if (this.state === state) return;
    this.state = state;
    this.hint(now);
  }

  quietRole(now) {
    const turn = this.turn;
    if (turn && !turn.done) return turn.at === null || now < turn.at - 0.05 ? "think" : null;
    return this.idle ? (this.state === "LISTENING" ? "listen" : "idle") : null;
  }

  quietNames(role, readyOnly) {
    for (const candidate of CLIP_QUIET[role] || []) {
      const names = this.names((c) => c.role === candidate).filter((n) => !readyOnly || this.ready(n));
      if (names.length) return names;
    }
    return [];
  }

  // Which gestures suit `mood` with `left` seconds of speech to go, best first: those of the mood, then the similar ones;
  // the latest gestures are left out as long as something else is left, the very latest one longest.
  rank(mood, left, readyOnly) {
    const usable = this.names((c) => c.role === "talk" && c.duration - 1 / c.fps <= left + CLIP_OVERRUN).filter((n) => !readyOnly || this.ready(n));
    let pool = usable;
    for (let keep = CLIP_RECENT; keep > 0; keep--) {
      const fresh = usable.filter((n) => this.recent.slice(-keep).indexOf(n) < 0);
      if (fresh.length) {
        pool = fresh;
        break;
      }
    }
    for (const similar of CLIP_SIMILAR[mood] || []) {
      const group = pool.filter((n) => this.clips[n].mood === similar);
      if (group.length) return group;
    }
    return pool;
  }

  // The mood of what is said `seconds` into the speech.
  moodAt(seconds) {
    const { sentences, total } = this.turn;
    if (!sentences.length) return "explain";
    const share = total > 0 ? clipClamp(seconds / total, 0, 0.9999) : 0;
    const found = sentences.findIndex((sentence) => share < sentence.to);
    const index = found < 0 ? sentences.length - 1 : found;
    return clipMood(sentences[index], index);
  }

  // Tells the pool which clips are likely to play next, so that they are loaded in time.
  hint(now, initial) {
    const turn = this.turn;
    const active = turn && !turn.done;
    if (!initial && !active && !this.idle) return;
    const into = turn && turn.at !== null ? this.speechTime(now) : 0;
    const left = turn && turn.total > 0 ? Math.max(0, turn.total - into) : Infinity;
    const next = active ? this.rank(this.moodAt(into + CLIP_LOOKAHEAD), left, false) : [];
    const names = this.quietNames(this.quietRole(now) || "think", false)
      .slice(0, this.idle && !active ? 2 : 1).concat(next.slice(0, 2));
    if (this.idle && !active) {
      for (const role of ["think", "listen", "idle"]) names.push(...this.quietNames(role, false).slice(0, 1));
    }
    const unique = [...new Set(names)].slice(0, CLIP_SLOTS);
    if (unique.length) this.queue.push({ type: "prepare", names: unique });
  }

  // A message is sent: the avatar starts to think. A gesture that is still playing is let go first.
  ask(now) {
    this.turn = { asked: now, text: "", sentences: [], total: 0, at: null, position: null, gestures: 0, done: false };
    if (this.cur && this.cur.clip.role === "talk") this.release(false);
    this.hint(now);
  }

  // The words of the reply, which are known before its voice is.
  speak(text) {
    if (!this.turn) return;
    this.turn.text = String(text || "");
    this.turn.sentences = clipSentences(text);
    this.hint(this.lastTick || 0);
  }

  // How many seconds the speech lasts, once that is known.
  length(seconds) {
    if (this.turn && seconds > 0 && seconds !== this.turn.total) {
      this.turn.total = seconds;
      this.hint(this.lastTick || 0);
    }
  }

  progress(seconds) {
    if (!Number.isFinite(seconds) || seconds < 0) throw new Error("Speech position must be a non-negative number.");
    if (this.turn) this.turn.position = seconds;
  }

  speechTime(now) {
    return this.turn.position === null ? Math.max(0, now - this.turn.at) : this.turn.position;
  }

  // The first sound is to be heard at `at` (the clock of tick()).
  speechStart(at) {
    if (!this.turn || this.turn.at !== null) return;
    this.turn.at = at;
    this.hint(at);
  }

  // The reply is complete: what plays now is let go, and nothing new begins.
  finish() {
    if (this.turn) this.turn.done = true;
  }

  // The reply is cut short: everything fades quickly.
  stop() {
    if (this.turn) this.turn.done = true;
    if (this.cur) this.release(true);
  }

  gap() {
    return (CLIP_GAP[0] + this.random() * (CLIP_GAP[1] - CLIP_GAP[0])) / this.pace;
  }

  start(name, now) {
    const clip = this.clips[name];
    this.cur = {
      name, clip, began: now, moved: now, seen: false, time: 0, loops: 0, elapsed: 0, weight: 0,
      exit: clip.loop ? Infinity : clip.last, fadeOut: CLIP_FADE, released: false, ended: false,
    };
    this.queue.push({ type: "play", name });
  }

  // The playing clip is to end at its next exit (or at once, if `fast` or the exit is far off).
  release(fast, maxWait = CLIP_MAX_WAIT, fastFade = CLIP_FAST_FADE) {
    const cur = this.cur;
    if (!cur) return;
    if (!cur.seen) {
      this.queue.push({ type: "park", name: cur.name });
      this.cur = null;
      return;
    }
    let exit = clipNextExit(cur.clip, cur.elapsed, CLIP_FADE);
    cur.released = true;
    if (fast || exit - cur.elapsed > maxWait) {
      exit = cur.elapsed + fastFade;
      cur.fadeOut = fastFade;
    }
    cur.exit = Math.min(cur.exit, exit);
  }

  end(now) {
    const cur = this.cur;
    this.queue.push({ type: "park", name: cur.name });
    this.cur = null;
    this.restUntil = cur.clip.role === "talk" ? now + this.gap() : now;
    this.hint(now);
  }

  drop(now) {
    this.broken.add(this.cur.name);
    this.queue.push({ type: "park", name: this.cur.name });
    this.cur = null;
    this.restUntil = now;
  }

  // What the video says about the playing clip: where it is, and whether it is playing at all.
  observe(now, media, paused) {
    const cur = this.cur, clip = cur.clip;
    if (paused) {
      // A page that was not drawn (a hidden tab) is no fault of the video.
      cur.began = now;
      cur.moved = now;
    }
    if (media && media.name === cur.name) {
      if (cur.seen && clip.loop && media.time < cur.time - clip.duration / 2) cur.loops++;
      if (!cur.seen || media.time !== cur.time) cur.moved = now;
      cur.seen = true;
      cur.time = media.time;
      cur.ended = Boolean(media.ended);
    }
    cur.elapsed = cur.loops * clip.duration + cur.time;
    if (!cur.seen && now - cur.began > CLIP_START_TIMEOUT) this.drop(now);
    else if (cur.seen && !cur.ended && now - cur.moved > CLIP_STALL) this.drop(now);
  }

  // Lets go of what has done its part.
  steer(now) {
    const cur = this.cur, turn = this.turn;
    if (cur.released) return;
    const over = !turn || turn.done;
    if (cur.clip.role === "talk") {
      if (over) this.release(false);
      return;
    }
    const speaking = turn && !turn.done && turn.at !== null && now >= turn.at - 0.05;
    const role = this.quietRole(now), preferred = this.quietNames(role, false);
    const wanted = preferred.includes(cur.name) ? preferred.filter((n) => this.ready(n)) : this.quietNames(role, true);
    const others = wanted.filter((n) => n !== cur.name);
    if (speaking) this.release(false, CLIP_HANDOFF_WAIT, CLIP_FADE);
    else if (wanted.length && !wanted.includes(cur.name)) this.release(false, CLIP_HANDOFF_WAIT, CLIP_FADE);
    else if ((over && !this.idle) || (cur.elapsed >= CLIP_THINK_SWAP && others.length)) this.release(false);
  }

  weigh(now) {
    const cur = this.cur;
    const left = cur.exit - cur.elapsed;
    cur.weight = clipSmooth(cur.elapsed / CLIP_FADE) * clipSmooth(left / cur.fadeOut);
    if (cur.seen && (left <= 0 || (cur.ended && !cur.clip.loop))) this.end(now);
  }

  begin(now) {
    const turn = this.turn;
    const over = !turn || turn.done;
    if (over && !this.idle) return;
    if (over || turn.at === null || now < turn.at - 0.05) {
      const ready = this.quietNames(this.quietRole(now), true);
      const fresh = ready.filter((n) => n !== this.lastThink);
      const name = this.choose(fresh.length ? fresh : ready);
      if (name) {
        this.lastThink = name;
        this.start(name, now);
      }
      return;
    }
    if (now < this.restUntil) return;
    const into = this.speechTime(now);
    const left = turn.total > 0 ? turn.total - into : Infinity;
    if (left < (turn.gestures ? CLIP_MIN_TAIL : CLIP_FIRST_TAIL)) return;
    const name = this.choose(this.rank(this.moodAt(into + CLIP_LOOKAHEAD), left, true));
    if (!name) return;
    this.recent = this.recent.concat(name).slice(-CLIP_RECENT);
    turn.gestures++;
    this.start(name, now);
  }

  tick(now, media) {
    const paused = this.lastTick !== null && now - this.lastTick > 0.5;
    this.lastTick = now;
    if (this.cur) this.observe(now, media, paused);
    if (this.cur) this.steer(now);
    if (this.cur) this.weigh(now);
    if (!this.cur) this.begin(now);
    const cur = this.cur;
    const commands = this.queue;
    this.queue = [];
    return { name: cur ? cur.name : null, weight: cur ? cur.weight : 0, time: cur ? cur.time : 0, commands };
  }
}

// The format to take the clips in: the first of the manifest's that the browser says it can play (a "probably" before a
// "maybe"), or null. `canPlay` is the browser's canPlayType.
function clipFormat(types, canPlay) {
  let best = null, rank = 0;
  for (const format of Object.keys(types || {})) {
    const verdict = canPlay(types[format]);
    const score = verdict === "probably" ? 2 : verdict === "maybe" ? 1 : 0;
    if (score > rank) {
      best = format;
      rank = score;
    }
  }
  return best;
}

// Plays the clips in a small pool of <video> elements. All the clips are fetched into memory when the avatar loads (the
// non-speaking ones first, and the player waits only for those), but only a few can be decoded at a time, so the pool holds the
// clip that plays and the ones the director has said are likely to be next, and gives up the one used longest ago when it
// needs room, a non-speaking clip last. A video is asked for its frames with requestVideoFrameCallback where there is one, so that
// the picture and its media time belong together; otherwise currentTime has to do. The frame is captured in that
// callback: reading the still-playing video element later could upload newer pixels with the older pose timestamp.
// `env`: {base, download(url) -> Blob, makeVideo(), captureFrame(video, time), url, canPlay(type), makeBlob(blob, type)}.
class ClipReels {
  constructor(manifest, env) {
    this.clips = manifest.clips;
    this.env = env;
    this.format = clipFormat(manifest.types, env.canPlay);
    this.urls = new Map();
    this.lost = new Set();
    this.slots = [];
    this.playing = null;
    this.stamp = 0;
    this.background = Promise.resolve();
    if (!this.format) throw new Error("This browser plays none of the clip formats.");
  }

  // Fetches the clips; resolves when the non-speaking ones are in, and the rest follow in the background.
  async load() {
    const mime = this.format === "mp4" ? "video/mp4" : "video/webm";
    const fetchOne = async (name) => {
      const file = this.clips[name].files[this.format];
      try {
        if (!file) throw new Error(`${name} has no ${this.format} file`);
        const blob = await this.env.download(this.env.base + file);
        this.urls.set(name, this.env.url.createObjectURL(this.env.makeBlob(blob, mime)));
      } catch (error) {
        console.warn(`The clip ${name} could not be loaded:`, error);
        this.lost.add(name);
      }
    };
    const names = Object.keys(this.clips);
    const first = names.filter((name) => this.clips[name].role !== "talk");
    const rest = names.filter((name) => this.clips[name].role === "talk");
    await Promise.all(first.map(fetchOne));
    this.background = (async () => {
      for (let i = 0; i < rest.length; i += 2) await Promise.all(rest.slice(i, i + 2).map(fetchOne));
    })();
  }

  failed(name) {
    return this.lost.has(name);
  }

  ready(name) {
    const slot = this.slots.find((s) => s.name === name);
    return Boolean(slot) && slot.state !== "loading" && slot.video.readyState >= 2;
  }

  newSlot() {
    const video = this.env.makeVideo();
    const slot = { video, name: null, state: "empty", used: 0, shown: false, fresh: false, time: 0, uploaded: -1, image: null, playRequest: null };
    video.addEventListener("loadeddata", () => {
      if (slot.state === "loading") slot.state = "ready";
    });
    video.addEventListener("error", () => {
      if (slot.name) this.lost.add(slot.name);
      this.release(slot);
    });
    if (video.requestVideoFrameCallback) {
      const watch = (now, metadata) => {
        if (slot.disposed) return;
        if (slot.state === "playing") {
          try {
            this.capture(slot, metadata.mediaTime);
          } catch (error) {
            console.warn(`The clip ${slot.name} could not retain its video frame:`, error);
            this.lost.add(slot.name);
            video.pause();
            this.release(slot);
          }
        }
        slot.callback = video.requestVideoFrameCallback(watch);
      };
      slot.callback = video.requestVideoFrameCallback(watch);
      slot.framed = true;
    }
    this.slots.push(slot);
    return slot;
  }

  release(slot) {
    slot.playRequest = null;
    this.discardFrame(slot);
    if (this.playing === slot) this.playing = null;
    slot.name = null;
    slot.state = "empty";
    slot.shown = slot.fresh = false;
  }

  discardFrame(slot) {
    if (slot.image && typeof slot.image.close === "function") slot.image.close();
    slot.image = null;
  }

  capture(slot, time) {
    const image = this.env.captureFrame ? this.env.captureFrame(slot.video, time) : slot.video;
    if (slot.image !== image) this.discardFrame(slot);
    slot.image = image;
    slot.time = time;
    slot.shown = slot.fresh = true;
  }

  // A slot to load a clip into: a new one while the pool has room, else the one used longest ago that is neither playing nor
  // wanted, a thinking clip last.
  room(wanted) {
    const empty = this.slots.find((s) => s.state === "empty");
    if (empty) return empty;
    if (this.slots.length < CLIP_SLOTS) return this.newSlot();
    const free = this.slots.filter((s) => s !== this.playing && wanted.indexOf(s.name) < 0);
    const think = (s) => (this.clips[s.name].role !== "talk" ? 1 : 0);
    free.sort((a, b) => think(a) - think(b) || a.used - b.used);
    return free[0] || null;
  }

  // The clips likely to play next, most likely first: gets them into the pool.
  prepare(names) {
    const wanted = names.filter((name) => this.urls.has(name) && !this.lost.has(name)).slice(0, CLIP_SLOTS);
    for (const name of wanted) {
      const have = this.slots.find((s) => s.name === name);
      if (have) {
        have.used = ++this.stamp;
        continue;
      }
      const slot = this.room(wanted);
      if (!slot) break;
      slot.playRequest = null;
      this.discardFrame(slot);
      slot.name = name;
      slot.state = "loading";
      slot.used = ++this.stamp;
      slot.shown = slot.fresh = false;
      slot.time = 0;
      slot.video.loop = Boolean(this.clips[name].loop);
      slot.video.src = this.urls.get(name);
      slot.video.load();
    }
  }

  play(name) {
    const slot = this.slots.find((s) => s.name === name);
    if (!slot || slot.state === "loading") return;
    const video = slot.video;
    this.discardFrame(slot);
    if (video.currentTime !== 0) video.currentTime = 0;
    slot.state = "playing";
    slot.shown = slot.fresh = false;
    slot.time = 0;
    slot.uploaded = -1;
    slot.used = ++this.stamp;
    this.playing = slot;
    const request = {};
    slot.playRequest = request;
    const started = video.play();
    if (started && started.catch) {
      started.catch((error) => {
        // Parking or reusing a slot can reject its old play() promise after a new clip has started.
        if (slot.playRequest !== request) return;
        console.warn(`The clip ${name} would not play:`, error);
        this.lost.add(name);
        this.release(slot);
      });
    }
  }

  // The clip is over: back to its first frame, ready for another go.
  park(name) {
    const slot = this.slots.find((s) => s.name === name);
    if (!slot) return;
    slot.playRequest = null;
    this.discardFrame(slot);
    slot.video.pause();
    if (slot.video.currentTime !== 0) slot.video.currentTime = 0;
    if (this.playing === slot) this.playing = null;
    slot.state = "ready";
    slot.shown = slot.fresh = false;
  }

  // Where the playing video is, or null until it has shown its first frame.
  media() {
    const slot = this.playing;
    if (!slot) return null;
    const video = slot.video;
    if (!slot.framed) {
      slot.time = video.currentTime;
      slot.shown = slot.shown || video.currentTime > 0;
    }
    return slot.shown ? { name: slot.name, time: slot.time, ended: Boolean(video.ended) } : null;
  }

  // The playing video if it has shown a frame that was not taken yet, else null.
  frame() {
    const slot = this.playing;
    if (!slot || !slot.shown) return null;
    if (slot.framed) {
      if (!slot.fresh) return null;
      slot.fresh = false;
      return slot.image;
    }
    if (slot.video.currentTime === slot.uploaded) return null;
    slot.uploaded = slot.video.currentTime;
    return slot.video;
  }

  // The video of a clip that is in the pool, to look at its first frame.
  video(name) {
    const slot = this.slots.find((s) => s.name === name);
    return slot ? slot.video : null;
  }

  dispose() {
    for (const slot of this.slots) {
      slot.disposed = true;
      slot.playRequest = null;
      if (slot.callback && slot.video.cancelVideoFrameCallback) slot.video.cancelVideoFrameCallback(slot.callback);
      this.discardFrame(slot);
      slot.video.pause();
      slot.video.removeAttribute("src");
      slot.video.load();
      if (slot.video.remove) slot.video.remove();
    }
    this.slots = [];
    this.playing = null;
    for (const url of this.urls.values()) this.env.url.revokeObjectURL(url);
    this.urls.clear();
  }
}

// The picture that turns the clip's colours into the photo's: rows of [r, g, b] (0-1) for the clip's first frame and for the
// same part of the photo, as a 3x3 matrix (row major) and an offset with photo = matrix * clip + offset. The clip is the
// photo made over by a video codec, and a browser may read it a little differently from the way it was written, so the clip
// is fitted to the photo rather than trusted. A fit that asks for much is not trusted either: it is better to leave the
// colours as they are than to repaint the avatar, so such a fit gives the identity.
function clipTone(clip, photo) {
  const identity = { matrix: [1, 0, 0, 0, 1, 0, 0, 0, 1], offset: [0, 0, 0], fitted: false };
  const count = Math.min(clip.length, photo.length);
  if (count < 64) return identity;
  const solve = (weights) => {
    // Normal equations of photo[c] ~ clip . m + b, with a pull towards the identity so a flat picture stays well-posed.
    const a = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]];
    const rhs = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]];
    for (let i = 0; i < count; i++) {
      const w = weights ? weights[i] : 1;
      if (!w) continue;
      const x = [clip[i][0], clip[i][1], clip[i][2], 1];
      for (let r = 0; r < 4; r++) {
        for (let c = 0; c < 4; c++) a[r][c] += w * x[r] * x[c];
        for (let c = 0; c < 3; c++) rhs[r][c] += w * x[r] * photo[i][c];
      }
    }
    const ridge = 0.01 * count;
    for (let r = 0; r < 3; r++) {
      a[r][r] += ridge;
      for (let c = 0; c < 3; c++) rhs[r][c] += ridge * (r === c ? 1 : 0);
    }
    // Gauss-Jordan on the 4x4 system, three right-hand sides at once.
    const m = a.map((row, r) => row.concat(rhs[r]));
    for (let col = 0; col < 4; col++) {
      let pivot = col;
      for (let r = col + 1; r < 4; r++) if (Math.abs(m[r][col]) > Math.abs(m[pivot][col])) pivot = r;
      if (Math.abs(m[pivot][col]) < 1e-9) return null;
      [m[col], m[pivot]] = [m[pivot], m[col]];
      const d = m[col][col];
      for (let c = col; c < 7; c++) m[col][c] /= d;
      for (let r = 0; r < 4; r++) {
        if (r === col) continue;
        const f = m[r][col];
        for (let c = col; c < 7; c++) m[r][c] -= f * m[col][c];
      }
    }
    // m[r][4 + c] is the coefficient of input r for output c.
    return {
      matrix: [0, 1, 2].flatMap((out) => [0, 1, 2].map((input) => m[input][4 + out])),
      offset: [m[3][4], m[3][5], m[3][6]],
    };
  };
  const apply = (fit, i) => [0, 1, 2].map((out) => fit.matrix[out * 3] * clip[i][0] + fit.matrix[out * 3 + 1] * clip[i][1]
    + fit.matrix[out * 3 + 2] * clip[i][2] + fit.offset[out]);
  let fit = solve(null);
  if (!fit) return identity;
  // One more pass without the pixels that do not match (the head, which the live head replaces anyway, or a hand).
  const miss = [];
  for (let i = 0; i < count; i++) {
    const got = apply(fit, i);
    miss.push(Math.max(Math.abs(got[0] - photo[i][0]), Math.abs(got[1] - photo[i][1]), Math.abs(got[2] - photo[i][2])));
  }
  const median = miss.slice().sort((a, b) => a - b)[count >> 1];
  const keep = miss.map((x) => (x <= 3 * median + 0.01 ? 1 : 0));
  fit = solve(keep) || fit;
  const far = fit.matrix.some((x, i) => Math.abs(x - (i % 4 ? 0 : 1)) > 0.25) || fit.offset.some((x) => Math.abs(x) > 0.1);
  return far ? identity : { matrix: fit.matrix, offset: fit.offset, fitted: true };
}

const FILLER_FPS = 25;
const FILLER_VERSION = 2;
// The filler phrases' index (clip-fillers.json) checked against its data (`byteLength`), the pictures' size [w, h], the
// speaking loop's frame count and fps (the fillers are said in it: its mouth is at ease, the rest's smile is not) and the
// rest bank's frames per period; null if anything does not fit. Each phrase is {text, rate, audio: [offset, length] of
// 16-bit PCM, takes: [{phase: rest frame it starts on, loop: loop frame it starts on, rows: [[loop frame, offset,
// length]...]}]}; `fade`: pictures over which the rest dissolves into the loop.
function checkFillers(info, byteLength, size, loopCount, loopFps, restCount) {
  const span = (at, length) => Number.isInteger(at) && Number.isInteger(length) && at >= 0 && length > 0 &&
    at + length <= byteLength;
  if (!info || info.version !== FILLER_VERSION || info.fps !== FILLER_FPS || info.loopFps !== loopFps ||
      info.restFps !== loopFps || !Number.isInteger(info.fade) || info.fade < 1 || info.fade > FILLER_FPS ||
      !Array.isArray(info.fillers) || !info.fillers.length || !clipNumbers(info.box, 4) ||
      !info.box.every(Number.isInteger)) return null;
  const [x, y, w, h] = info.box;
  if (x < 0 || y < 0 || w < 2 || h < 2 || x + w > size[0] || y + h > size[1]) return null;
  const fillers = [];
  for (const filler of info.fillers) {
    if (!filler || typeof filler.text !== "string" || !Number.isInteger(filler.rate) || filler.rate < 8000 ||
        !Array.isArray(filler.audio) || !span(...filler.audio) || filler.audio[1] % 2 || !Array.isArray(filler.takes) ||
        !filler.takes.length) return null;
    const takes = [];
    for (const take of filler.takes) {
      if (!take || !Number.isInteger(take.phase) || take.phase < 0 || take.phase >= restCount ||
          !Number.isInteger(take.loop) || take.loop < 0 || take.loop >= loopCount || !Array.isArray(take.rows) ||
          take.rows.length < 2 || take.rows.length > FILLER_FPS * 10 || takes.some((t) => t.phase === take.phase)) {
        return null;
      }
      for (const [frame, at, length] of take.rows) {
        if (!Number.isInteger(frame) || frame < 0 || frame >= loopCount || !span(at, length)) return null;
      }
      if (take.rows[0][0] !== take.loop) return null;
      takes.push({ phase: take.phase, loop: take.loop, rows: take.rows });
    }
    fillers.push({ text: filler.text, rate: filler.rate, audio: filler.audio, takes });
  }
  return { box: info.box, fps: FILLER_FPS, fade: info.fade, fillers };
}

// Ties the director, the videos and the compositor together: the player tells it what happens in the conversation and calls
// update() once for every picture it draws.
class ClipStage {
  constructor({ manifest, reels, director, compositor, frames = null, restFrames = null, now = 0, random = Math.random,
    compose = null }) {
    this.manifest = manifest;
    this.reels = reels;
    this.director = director;
    this.compositor = compositor;
    this.weight = 0;
    this.frames = frames;
    this.epoch = now;
    this.shown = -1;
    this.loopClip = frames ? Object.values(manifest.clips)[0] : null;
    this.restFrames = restFrames;
    this.restAt = now;
    this.random = random;
    this.blinkSlots = restFrames && manifest.continuous.rest.randomBlink;
    this.blinkStart = null;
    // A filler phrase being said over the rest: {at, end, step, fps, box, rows, images}; `compose(rest picture, crop,
    // box)` puts one of its pictures together.
    this.filler = null;
    this.compose = compose;
  }

  // The rest step (at or after `lead` seconds from now) on which a filler can begin, eyes open, from one of `takes`
  // ([{phase: rest frame within the period, loop: speaking-loop frame}]), lasting `seconds`: {at, step, take}. Just after
  // speech, while the body is still on its way back to the rest, it begins once the rest has. From `at` the body is in
  // the speaking loop (from the take's loop frame) until the loop next comes back to the rest after the filler, so that a
  // reply ready by then carries on the same motion.
  planFiller(now, lead, takes, seconds) {
    if (!this.restFrames || this.filler) return null;
    const fps = this.loopClip.fps, period = this.manifest.continuous.rest.anchorEvery;
    if (!period) return null;
    const byPhase = new Map(takes.map((take) => [take.phase % period, take]));
    let step = Math.max(0, Math.ceil((now + lead - this.restAt) * fps - 1e-6));
    const current = Math.floor((now - this.restAt) * fps + 1e-6);
    // A blink that has begun is let finish; one yet to come waits until the filler is over.
    const blinkEnd = this.blinkSlots && this.blinkStart !== null && current >= this.blinkStart ?
      this.blinkStart + this.blinkSlots.length : 0;
    for (let tries = 0; tries <= period * 2; tries++, step++) {
      const take = byPhase.get(step % period);
      if (take && step >= blinkEnd) {
        const at = this.restAt + step / fps, end = at + seconds;
        const count = this.frames.length;
        let restAt = at + ((this.manifest.continuous.rest.anchor - take.loop + count) % count) / fps;
        while (restAt < end - 1e-6) restAt += count / fps;
        this.filler = { at, end, step, take, rows: null, images: null, entered: false,
          before: { epoch: this.epoch, restAt: this.restAt }, after: { epoch: at - take.loop / fps, restAt } };
        return { at, step, take };
      }
    }
    return null;
  }

  // The filler planned with planFiller, now that its pictures are decoded (the stage closes them).
  startFiller({ rows, images, box, fps, fade }) {
    if (!this.filler) {
      for (const image of images) if (image) image.close();
      return false;
    }
    Object.assign(this.filler, { rows, images, box, fps, fade });
    return true;
  }

  // A filler stopped before it began leaves the body at rest; one under way leaves it in the speaking loop, which goes
  // back to the rest by itself.
  stopFiller() {
    if (!this.filler) return;
    if (this.filler.images) for (const image of this.filler.images) if (image) image.close();
    if (!this.filler.entered) Object.assign(this, this.filler.before);
    this.filler = null;
    this.blinkStart = null;
    this.shown = -1;
  }

  ask(now) {
    if (this.director) this.director.ask(now);
  }

  setState(state, now) {
    if (this.director) this.director.setState(state, now);
  }

  speak(text) {
    if (this.director) this.director.speak(text);
  }

  length(seconds) {
    if (this.director) this.director.length(seconds);
  }

  progress(seconds) {
    if (this.director) this.director.progress(seconds);
  }

  speechStart(at) {
    if (this.director) this.director.speechStart(at);
  }

  finish() {
    if (this.director) this.director.finish();
  }

  stop() {
    if (this.director) this.director.stop();
  }

  phase(now) {
    if (this.restFrames && now >= this.restAt) return this.manifest.continuous.rest.anchor;
    const count = this.frames.length;
    return ((Math.floor((now - this.epoch) * this.loopClip.fps + 1e-6) % count) + count) % count;
  }

  arm(now, delay) {
    // A reply waits for the filler before it to finish.
    let { epoch, restAt } = this;
    if (this.filler) {
      delay = Math.max(delay, this.filler.end - now);
      ({ epoch, restAt } = this.filler.after);
    }
    if (this.restFrames && now + delay < restAt) {
      // The body is still in the speaking loop then (after a filler or a reply): the reply carries on its motion, and the
      // body stays in the loop until the reply begins (a whole loop later if it is late).
      const fps = this.loopClip.fps, count = this.frames.length, period = count / fps;
      const step = Math.ceil((now + delay - epoch) * fps - 1e-6);
      const at = epoch + step / fps;
      while (restAt < at + period) restAt += period;
      if (this.filler) this.filler.after.restAt = restAt;
      if (!this.filler || this.filler.entered) this.restAt = restAt;
      return { frame: ((step % count) + count) % count, at, period };
    }
    if (this.restFrames) {
      const count = this.restFrames.length, fps = this.loopClip.fps;
      const period = (this.manifest.continuous.rest.anchorEvery || count) / fps;
      const blinkFrames = this.manifest.continuous.rest.blink;
      let at;
      if (blinkFrames) {
        const blink = new Set(blinkFrames);
        let step = Math.max(0, Math.ceil((now + delay - restAt) * fps - 1e-6));
        while (blink.has(step % count)) step++;
        at = restAt + step / fps;
      } else {
        at = restAt + Math.max(0, Math.ceil((now + delay - restAt) / period)) * period;
      }
      return { frame: this.manifest.continuous.rest.anchor, at, period };
    }
    const step = Math.ceil((now + delay - this.epoch) * this.loopClip.fps);
    return { frame: step % this.frames.length, at: this.epoch + step / this.loopClip.fps,
      period: this.frames.length / this.loopClip.fps };
  }

  resume(frame, now) {
    const current = frame % this.frames.length;
    this.epoch = now - current / this.loopClip.fps;
    if (this.restFrames) {
      const distance = (this.manifest.continuous.rest.anchor - current + this.frames.length) % this.frames.length;
      this.restAt = now + distance / this.loopClip.fps;
      this.blinkStart = null;
    }
    this.shown = -1;
  }

  restIndex(step) {
    if (!this.blinkSlots) return step % this.restFrames.length;
    const { every, offset, length } = this.blinkSlots, fps = this.loopClip.fps;
    const count = this.manifest.continuous.rest.anchorEvery;
    if (this.blinkStart === null || step >= this.blinkStart + length) {
      // Use the current clock after a suspended tab; never replay a backlog of blinks.
      const from = this.blinkStart === null ? step : Math.max(this.blinkStart, step - length);
      const first = Math.ceil((from + 6 * fps - offset) / every);
      const last = Math.floor((from + 12 * fps - offset) / every);
      this.blinkStart = (first + Math.floor(this.random() * (last - first + 1))) * every + offset;
    }
    return step % count + (step >= this.blinkStart ? count : 0);
  }

  // How much of the clip is drawn now (0-1); the body's own sway gives way to it by that much.
  update(now) {
    if (this.frames) {
      let filler = this.filler;
      if (filler && now >= filler.end) {
        this.stopFiller();
        filler = null;
      }
      if (filler && filler.rows && now >= filler.at && !filler.entered) {
        Object.assign(this, filler.after);
        filler.entered = true;
      }
      if (filler && !filler.entered) {
        // Before the filler starts (or while its pictures come), the body rests, eyes open: a blink under way ends, no
        // other begins.
        const { restAt } = filler.before, period = this.manifest.continuous.rest.anchorEvery;
        if (now >= restAt) {
          const step = Math.floor((now - restAt) * this.loopClip.fps + 1e-6);
          const blinking = this.blinkSlots && this.blinkStart !== null && step >= this.blinkStart &&
            step < this.blinkStart + this.blinkSlots.length;
          const index = blinking ? this.restIndex(step) : step % period;
          if (this.frames.length + index !== this.shown) {
            this.compositor.setClipFrame(this.restFrames[index]);
            this.shown = this.frames.length + index;
          }
        } else {
          const index = ((Math.floor((now - filler.before.epoch) * this.loopClip.fps + 1e-6) % this.frames.length) +
            this.frames.length) % this.frames.length;
          if (index !== this.shown) {
            this.compositor.setClipFrame(this.frames[index]);
            this.shown = index;
          }
        }
        this.compositor.setClip({ mix: 1, head: null });
        this.weight = 1;
        return 1;
      }
      if (filler) {
        // The filler is said in the speaking loop; over its first pictures the rest it began on dissolves into it.
        const i = Math.floor((now - filler.at) * (filler.fps || FILLER_FPS) + 1e-6);
        const row = i >= 0 && i < filler.rows.length ? filler.rows[i] : null;
        if (row && filler.images[i] && this.compose) {
          if (this.shown !== -2 - i) {
            const fade = Math.min(1, (i + 1) / (filler.fade || 1));
            const rest = fade < 1 ? this.restFrames[(filler.step + Math.floor((now - filler.at) * this.loopClip.fps +
              1e-6)) % this.manifest.continuous.rest.anchorEvery] : null;
            this.compositor.setClipFrame(this.compose(this.frames[row[0]], filler.images[i], filler.box, rest, fade));
            this.shown = -2 - i;
          }
          this.compositor.setClip({ mix: 1, head: null });
          this.weight = 1;
          return 1;
        }
      }
      const resting = this.restFrames && now >= this.restAt;
      const step = resting ? Math.floor((now - this.restAt) * this.loopClip.fps + 1e-6) : 0;
      const index = resting ? this.restIndex(step) : this.phase(now);
      const bank = resting ? this.restFrames : this.frames;
      const shown = resting ? this.frames.length + index : index;
      if (shown !== this.shown) {
        this.compositor.setClipFrame(bank[index]);
        this.shown = shown;
      }
      this.compositor.setClip({ mix: 1, head: null });
      this.weight = 1;
      return 1;
    }
    const out = this.director.tick(now, this.reels.media());
    // The clip to play is put on before the pool makes room for the ones that may follow it: a gesture chosen because the one
    // the sentence asks for is not in yet would otherwise be the one to give way to it.
    const wishes = (command) => command.type === "prepare";
    for (const command of out.commands.filter((c) => !wishes(c)).concat(out.commands.filter(wishes))) {
      if (command.type === "prepare") this.reels.prepare(command.names);
      else this.reels[command.type](command.name);
    }
    if (out.name) {
      const video = this.reels.frame();
      if (video) this.compositor.setClipFrame(video);
    }
    const clip = out.name && this.manifest.clips[out.name];
    const state = { mix: out.weight, head: clip ? clipHead(clip, out.time) : null };
    if (clip && clip.faceTrack) state.face = clipFace(clip, out.time);
    this.compositor.setClip(state);
    this.weight = out.weight;
    return out.weight;
  }

  dispose() {
    this.compositor.clearClips();
    if (this.reels) this.reels.dispose();
    if (this.frames) for (const frame of new Set(this.frames)) frame.close();
    if (this.restFrames) for (const frame of new Set(this.restFrames)) frame.close();
    this.stopFiller();
    this.frames = null;
    this.restFrames = null;
    this.weight = 0;
  }
}
