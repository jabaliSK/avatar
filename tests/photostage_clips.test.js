"use strict";

const nodeAssert = require("assert");
const fs = require("fs");
const path = require("path");
const vm = require("vm");

const plain = (value) => JSON.parse(JSON.stringify(value));
// What the browser context makes has that context's Array and Object, so values are compared by what they hold.
const assert = Object.assign((...args) => nodeAssert(...args), nodeAssert, {
  deepStrictEqual: (actual, expected, message) => nodeAssert.deepStrictEqual(plain(actual), plain(expected), message),
});

const browser = vm.createContext({ console: { warn() {}, error() {}, log() {} } });
vm.runInContext(fs.readFileSync(path.join(__dirname, "../photostage/web/clips.js"), "utf8"), browser);
const api = vm.runInContext(
  `({ parseClipManifest, quietClipManifest, clipHead, clipFace, clipNextExit, clipSentences, clipMood, clipFormat, clipTone,
      CLIP_FADE, CLIP_FAST_FADE, CLIP_MAX_WAIT, CLIP_HANDOFF_WAIT, CLIP_START_TIMEOUT, CLIP_STALL, CLIP_MIN_TAIL, CLIP_THINK_SWAP, CLIP_SLOTS,
      director: (clips, options) => new ClipDirector(clips, options),
      reels: (manifest, env) => new ClipReels(manifest, env),
      checkFillers, stage: (parts) => new ClipStage(parts) })`,
  browser,
);
const { parseClipManifest, clipHead, clipNextExit, clipSentences, clipMood, clipFormat, clipTone } = api;

const DT = 1 / 60;
// A fade is a smoothstep over CLIP_FADE seconds, whose steepest step at 60 pictures a second is this (a fast one is twice it).
const STEP = 1.5 / api.CLIP_FADE * DT + 1e-6;
const think = (extra) => Object.assign({ role: "think", mood: "think", duration: 3, fps: 24, loop: true, exits: [1.5], files: { mp4: "t.mp4" } }, extra);
const talk = (mood, duration, extra) => Object.assign({ role: "talk", mood, duration, fps: 24, loop: false, exits: [duration / 2], files: { mp4: "g.mp4" } }, extra);

function manifest(clips, extra) {
  return Object.assign({
    version: 1, size: [1000, 2000], box: [0, 500, 1000, 2000], pivot: [500, 800], follow: [200, 700],
    types: { mp4: 'video/mp4; codecs="avc1.640029"', webm: 'video/webm; codecs="vp9"' }, clips,
  }, extra);
}

const PACK = {
  "think-a": think(),
  "think-b": think({ duration: 4, exits: [2] }),
  calm: talk("calm", 3),
  explain: talk("explain", 4),
  emphasis: talk("emphasis", 3.5),
  greet: talk("greet", 3),
  ask: talk("ask", 3),
};

// A video world: plays what the director commands, after `delay`, and reports where it is as the pool would.
class Sim {
  constructor(clips, options = {}) {
    this.pack = parseClipManifest(manifest(clips), null);
    this.clips = this.pack ? this.pack.clips : {};
    this.parked = [];
    this.prepared = [];
    this.playing = null;
    this.now = 100;
    this.delay = options.delay === undefined ? 0.08 : options.delay;
    this.frozen = options.frozen || new Set();
    this.dead = options.dead || new Set();
    this.ready = options.ready || (() => true);
    let seed = options.seed || 7;
    this.random = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
    this.director = api.director(this.clips, {
      ready: this.ready, failed: (name) => this.dead.has(name), random: this.random, motion: options.motion, idle: options.idle,
    });
    this.log = [];
    this.commands = [];
  }

  media() {
    const p = this.playing;
    if (!p || this.dead.has(p.name)) return null;
    const clip = this.clips[p.name];
    const t = this.now - p.at - this.delay;
    if (t < 0) return null;
    if (this.frozen.has(p.name)) return { name: p.name, time: Math.min(t, 0.04), ended: false };
    if (clip.loop) return { name: p.name, time: t % clip.duration, ended: false };
    return t >= clip.last ? { name: p.name, time: clip.last, ended: true } : { name: p.name, time: t, ended: false };
  }

  step(dt = DT) {
    this.now += dt;
    const out = this.director.tick(this.now, this.media());
    for (const c of out.commands) {
      this.commands.push(Object.assign({ at: this.now }, plain(c)));
      if (c.type === "play") {
        assert.strictEqual(this.playing, null, `${c.name} was started while ${this.playing && this.playing.name} was still there`);
        this.playing = { name: c.name, at: this.now };
      } else if (c.type === "park") {
        this.parked.push(c.name);
        if (this.playing && this.playing.name === c.name) this.playing = null;
      } else if (c.type === "prepare") {
        this.prepared.push(plain(c.names));
      }
    }
    this.log.push({ at: this.now, name: out.name, weight: out.weight, time: out.time });
    return out;
  }

  run(seconds, hook) {
    const steps = Math.round(seconds / DT);
    for (let i = 0; i < steps; i++) {
      if (hook) hook(this, i);
      this.step();
    }
  }

  played() {
    return this.commands.filter((c) => c.type === "play").map((c) => c.name);
  }
}

function maxJump(log) {
  let worst = 0;
  for (let i = 1; i < log.length; i++) worst = Math.max(worst, Math.abs(log[i].weight - log[i - 1].weight));
  return worst;
}

// -- the manifest

{
  const spec = manifest({ loop: think({ role: "idle", fps: 16, duration: 2 }) }, {
    mode: "avatar", box: [0, 0, 1000, 2000], face: [400, 200, 600, 400],
    continuous: { index: "quiet.json", frames: "quiet.bin" },
  });
  const pack = parseClipManifest(spec, [1000, 2000]);
  assert.ok(pack.continuous);
  assert.strictEqual(parseClipManifest({ ...spec, mode: "body" }, null), null);
  assert.strictEqual(parseClipManifest({ ...spec, continuous: { index: "../quiet.json", frames: "quiet.bin" } }, null), null);
  const frames = Array.from({ length: 32 }, (_, id) => ({ id, close() { this.closed = true; } }));
  const shown = [];
  const stage = api.stage({ manifest: pack, frames, now: 100,
    compositor: { setClipFrame: (frame) => shown.push(frame.id), setClip: (state) => assert.strictEqual(state.mix, 1), clearClips() {} } });
  stage.update(100);
  stage.ask(100.1);
  stage.setState("THINKING", 100.1);
  const armed = stage.arm(100.7, 2.5);
  assert.strictEqual(stage.phase(armed.at), armed.frame);
  stage.speechStart(armed.at);
  stage.setState("SPEAKING", armed.at);
  stage.update(armed.at);
  assert.strictEqual(shown[shown.length - 1], armed.frame);
  stage.finish();
  stage.resume(31, 108);
  stage.update(108);
  stage.update(108 + 1 / 16);
  assert.deepStrictEqual(shown.slice(-2), [31, 0], "handback continues the last body phase, including a loop wrap");
  stage.stop();
  stage.dispose();
  assert.ok(frames.every((frame) => frame.closed));

  const rest = { index: "rest.json", frames: "rest.bin", anchor: 11, blink: [8, 9, 10, 11] };
  const rested = parseClipManifest({ ...spec, continuous: { ...spec.continuous, rest } }, null);
  assert.ok(rested.continuous.rest);
  for (const invalid of [{ ...rest, anchor: -1 }, { ...rest, anchor: 32 }, { ...rest, anchor: true },
    { ...rest, frames: "../rest.bin" }, { ...rest, blink: [8, 8] }, { ...rest, blink: [true] },
    { ...rest, anchorEvery: 8 }, { ...rest, blink: null, anchorEvery: 0 }]) {
    assert.strictEqual(parseClipManifest({ ...spec, continuous: { ...spec.continuous, rest: invalid } }, null), null);
  }
  const restFrames = Array.from({ length: 48 }, (_, id) => ({ id: `rest-${id}`, close() { this.closed = true; } }));
  const calm = api.stage({ manifest: rested, frames, restFrames, now: 100,
    compositor: { setClipFrame: (frame) => shown.push(frame.id), setClip() {}, clearClips() {} } });
  for (const state of ["IDLE", "LISTENING", "THINKING"]) {
    calm.setState(state, 100);
    calm.update(100.4);
    assert.strictEqual(shown[shown.length - 1], "rest-6", "quiet states use the separate idle bank");
  }
  calm.update(100.5);
  assert.strictEqual(shown[shown.length - 1], "rest-8", "the idle bank plays its brief blink frames");
  const blinkGuard = calm.arm(100, 0.5);
  assert.strictEqual(blinkGuard.at, 100.75, "speech waits only until the blink ends");
  const boundary = calm.arm(100.7, 2.5);
  assert.strictEqual(boundary.at, 103.25);
  assert.strictEqual(boundary.period, 3);
  assert.strictEqual(boundary.frame, 11);
  calm.update(boundary.at);
  assert.strictEqual(shown[shown.length - 1], "rest-4", "speech starts on an open shared-anchor frame");
  calm.resume(30, 110);
  calm.update(110);
  assert.strictEqual(shown[shown.length - 1], 30, "return first retains the exact last speech phase");
  calm.update(110 + 12 / 16);
  assert.strictEqual(shown[shown.length - 1], 10);
  calm.update(110 + 13 / 16);
  assert.strictEqual(shown[shown.length - 1], "rest-0", "body follows its existing frames to the matching anchor");
  calm.resume(11, 115);
  calm.update(115);
  assert.strictEqual(shown[shown.length - 1], "rest-0");
  calm.dispose();
  assert.ok(restFrames.every((frame) => frame.closed));
  const legacyRest = parseClipManifest({ ...spec, continuous: { ...spec.continuous, rest: { index: "rest.json", frames: "rest.bin", anchor: 11 } } }, null);
  const legacy = api.stage({ manifest: legacyRest, frames, restFrames, now: 100,
    compositor: { setClipFrame() {}, setClip() {}, clearClips() {} } });
  assert.strictEqual(legacy.arm(100.7, 2.5).at, 106, "older rest packs retain boundary-aligned speech starts");
  legacy.dispose();
  const movingPack = parseClipManifest({ ...spec, continuous: { ...spec.continuous,
    rest: { index: "rest.json", frames: "rest.bin", anchor: 11, anchorEvery: 24 } } }, null);
  const moving = api.stage({ manifest: movingPack, frames, restFrames, now: 100, compositor: { clearClips() {} } });
  const next = moving.arm(100, 1);
  assert.deepStrictEqual([next.at, next.period, next.frame], [101.5, 1.5, 11],
    "the next shared pose, not the end of the longer blink timeline, is a safe start and retry");

  const subtle = { ...rest, blink: null, anchorEvery: 64, randomBlink: { every: 16, offset: 8, length: 4 } };
  const packWith = (s) => parseClipManifest({ ...spec, continuous: { ...spec.continuous, rest: s } }, null);
  assert.strictEqual(packWith({ ...subtle, randomBlink: { every: 16, offset: 0, length: 4 } }), null);
  const draws = [0, .999, .25, .5, .75];
  const quiet = api.stage({ manifest: packWith(subtle), frames, restFrames: Array(128), random: () => draws.shift() });
  const starts = [], phases = [];
  for (let i = 0; i <= 301; i++) {
    const picture = quiet.restIndex(i);
    assert.strictEqual(picture % 64, i % 64, "selecting eyelids must never move the body phase");
    if (picture >= 64) {
      if (!phases.length || phases[phases.length - 1] !== i - 1) starts.push(i);
      phases.push(i);
    }
  }
  assert.deepStrictEqual(starts, [104, 296], "the random draw chooses when to blink, not each body wrap");
  assert.deepStrictEqual(phases, [104, 105, 106, 107, 296, 297, 298, 299]);
  assert.ok(quiet.restIndex(9000) < 64, "long-suspended tabs must not replay overdue blinks");
  quiet.resume(11, 200);
  assert.ok(quiet.restIndex(0) < 64 && quiet.blinkStart < 200, "a new idle epoch gets a fresh blink schedule");

  // -- filler phrases said in the speaking loop, from a rest phase, while a reply is made
  const fillerSpec = { index: "fillers.json", frames: "fillers.bin" };
  assert.ok(packWith(subtle) && parseClipManifest({ ...spec, continuous: { ...spec.continuous, rest: subtle, fillers: fillerSpec } }, null)
    .continuous.fillers, "fillers ride on a period-anchored rest bank");
  const noRest = parseClipManifest({ ...spec, continuous: { ...spec.continuous, fillers: fillerSpec } }, null);
  assert.ok(noRest && !noRest.continuous.fillers, "fillers without a rest bank are dropped, the pack kept");
  const rows = (loop) => Array.from({ length: 30 }, (_, i) => [(loop + Math.floor(i * 16 / 25)) % 32, 100 + i * 10, 10]);
  const takes = [{ phase: 0, loop: 11, rows: rows(11) }, { phase: 16, loop: 5, rows: rows(5) }];
  const info = { version: 2, fps: 25, restFps: 16, loopFps: 16, fade: 6, box: [10, 20, 100, 60],
    fillers: [{ text: "Let me check that.", rate: 24000, audio: [0, 100], takes }] };
  const checked = api.checkFillers(info, 500, [1000, 2000], 32, 16, 64);
  assert.deepStrictEqual(checked.fillers[0].takes.map((take) => [take.phase, take.loop]), [[0, 11], [16, 5]]);
  assert.strictEqual(checked.fade, 6);
  const withTake = (take) => ({ ...info, fillers: [{ ...info.fillers[0], takes: [take] }] });
  for (const bad of [{ ...info, version: 1 }, { ...info, restFps: 25 }, { ...info, loopFps: 25 }, { ...info, fade: 0 },
    { ...info, box: [950, 20, 100, 60] }, { ...info, fillers: [{ ...info.fillers[0], audio: [0, 101] }] },
    withTake({ phase: 0, loop: 12, rows: rows(11) }), withTake({ phase: 64, loop: 11, rows: rows(11) }),
    withTake({ phase: 0, loop: 32, rows: rows(32) }), { ...info, fillers: [{ ...info.fillers[0], takes: [takes[0], takes[0]] }] }]) {
    assert.strictEqual(api.checkFillers(bad, 500, [1000, 2000], 32, 16, 64), null);
  }
  assert.strictEqual(api.checkFillers(info, 300, [1000, 2000], 32, 16, 64), null, "takes beyond the data");

  const drawn = [];
  const restPics = Array.from({ length: 128 }, (_, id) => ({ id: `rest-${id}`, close() {} }));
  const speaker = api.stage({ manifest: packWith(subtle), frames, restFrames: restPics, now: 100, random: () => 0,
    compose: (frame, crop, box, rest, fade) => ({ id: `${frame.id}+${crop.id}` + (rest ? `/${rest.id}@${fade.toFixed(2)}` : "") }),
    compositor: { setClipFrame: (frame) => drawn.push(frame.id), setClip() {}, clearClips() {} } });
  speaker.blinkStart = 20;
  const plan = speaker.planFiller(100.5, 0.45, takes, 1.2);
  assert.deepStrictEqual([plan.step, plan.take.phase, plan.at], [16, 16, 101], "the first allowed phase at or after the lead");
  assert.strictEqual(speaker.planFiller(100.5, 0.45, takes, 1.2), null, "one filler at a time");
  speaker.update(100.5 + 4 / 16);
  assert.strictEqual(drawn[drawn.length - 1], "rest-12", "the body rests, eyes open, until it starts");
  const crops = Array.from({ length: 30 }, (_, id) => ({ id: `crop-${id}`, close() { this.closed = true; } }));
  assert.ok(speaker.startFiller({ rows: rows(5), images: crops, box: [10, 20, 100, 60], fps: 25, fade: 6 }));
  speaker.update(plan.at + 0.001);
  assert.strictEqual(drawn[drawn.length - 1], "5+crop-0/rest-16@0.17", "the rest it began on dissolves into the loop");
  speaker.update(plan.at + 10 / 25 + 0.001);
  assert.strictEqual(drawn[drawn.length - 1], "11+crop-10", "the mouth is said in the speaking loop");
  assert.strictEqual(speaker.epoch, plan.at - 5 / 16);
  assert.strictEqual(speaker.restAt, plan.at + 6 / 16 + 2, "back to the rest at the loop's first anchor after the filler");
  const reply = speaker.arm(plan.at + 0.5, 0.1);
  assert.deepStrictEqual([reply.frame, reply.at, reply.period], [25, plan.at + 1.25, 2],
    "the reply waits for the filler and carries on the loop's motion");
  assert.strictEqual(speaker.restAt, plan.at + 6 / 16 + 4, "the body stays in the loop until the reply begins");
  speaker.update(plan.at + 1.25);
  assert.ok(!speaker.filler && crops.every((crop) => crop.closed), "its pictures are let go when it ends");
  assert.strictEqual(drawn[drawn.length - 1], 25);
  const blinking = api.stage({ manifest: packWith(subtle), frames, restFrames: restPics, now: 100,
    compositor: { setClipFrame() {}, setClip() {}, clearClips() {} } });
  const early = blinking.planFiller(99, 0.5, [takes[1]], 1);
  assert.deepStrictEqual([early.step, early.at], [16, 101], "on the way back to the rest, it waits for the rest");
  const before = [blinking.epoch, blinking.restAt];
  blinking.stopFiller();
  assert.deepStrictEqual([blinking.epoch, blinking.restAt], before, "a filler stopped before it began leaves the rest");
  blinking.blinkStart = 15;
  assert.strictEqual(blinking.planFiller(100 + 16 / 16, 0, [takes[1], { ...takes[1], phase: 32 }], 1).step, 32,
    "a blink under way is let finish first");
}

{
  const spec = manifest({ a: think(), b: talk("calm", 3), bad: { role: "talk" }, odd: talk("calm", 2, { role: "dance" }) });
  const pack = parseClipManifest(spec, [1000, 2000]);
  assert.deepStrictEqual(Object.keys(pack.clips).sort(), ["a", "b"], "clips that cannot be played are left out, the others kept");
  assert.strictEqual(parseClipManifest(spec, [1000, 2001]), null, "a pack made for another picture size is not used");
  assert.strictEqual(parseClipManifest(Object.assign({}, spec, { version: 2 }), null), null);
  assert.strictEqual(parseClipManifest(manifest({ bad: {} }), null), null, "a pack with nothing playable is not used");
  assert.strictEqual(parseClipManifest(null, null), null);
  // Where the service dubs the speech into the clips itself, the player plays only the clips around it.
  const quiet = api.quietClipManifest(pack);
  assert.deepStrictEqual(Object.keys(quiet.clips), ["a"]);
  assert.deepStrictEqual(Object.keys(pack.clips).sort(), ["a", "b"], "the pack itself is left as it was");
  assert.strictEqual(quiet.box, pack.box);
  assert.strictEqual(api.quietClipManifest(parseClipManifest(manifest({ b: talk("calm", 3) }), null)), null,
    "a pack of gestures alone leaves nothing to play");
  assert.strictEqual(parseClipManifest(manifest({ a: think() }, { box: [0, 10, 0, 20] }), null), null, "an empty box");
  assert.strictEqual(parseClipManifest(manifest({ a: think() }, { follow: [1] }), null), null);
  // A clip that is not a loop ends one frame before its duration; its exits are sorted, within the clip and end with its end.
  const clip = parseClipManifest(manifest({ g: talk("calm", 3, { exits: [2, 0.5, 2, 9, -1, 0] }) }), null).clips.g;
  assert.deepStrictEqual(plain(clip.exits), [0.5, 2, 3 - 1 / 24]);
  assert.ok(Math.abs(clip.last - (3 - 1 / 24)) < 1e-12);
  const loop = parseClipManifest(manifest({ t: think({ exits: [] }) }), null).clips.t;
  assert.deepStrictEqual(plain(loop.exits), [3], "a loop can always end where it begins again");
  // A track has one row of four numbers per frame.
  const row = [1, 0, 0, 0];
  const rows = (n) => Array.from({ length: n }, () => row);
  assert.ok(parseClipManifest(manifest({ g: talk("calm", 2, { track: rows(48) }) }), null));
  assert.strictEqual(parseClipManifest(manifest({ g: talk("calm", 2, { track: rows(47) }) }), null), null);
  assert.strictEqual(parseClipManifest(manifest({ g: talk("calm", 2, { track: [[1, 0, 0]].concat(rows(47)) }) }), null), null);
  const full = manifest({ t: think() }, { mode: "avatar", box: [0, 0, 1000, 2000], face: [300, 200, 700, 700] });
  assert.strictEqual(parseClipManifest(full, null).mode, "avatar");
  assert.strictEqual(parseClipManifest(Object.assign({}, full, { box: [0, 500, 1000, 2000] }), null), null);
  assert.strictEqual(parseClipManifest(Object.assign({}, full, { face: null }), null), null);
  const wholeHead = Object.assign({}, full, {faceMode:"head",clips:{t:think({track:rows(72)})}});
  assert.strictEqual(parseClipManifest(wholeHead, null).faceMode, "head");
  assert.strictEqual(parseClipManifest(Object.assign({}, full, {faceMode:"head"}), null), null, "full heads require collar tracks");
  assert.strictEqual(parseClipManifest(Object.assign({}, wholeHead, {mode:"body"}), null), null);
  assert.strictEqual(parseClipManifest(Object.assign({}, wholeHead, {facePose:[1,0,0,0,1,0]}), null), null,
    "the full-head path must not accidentally retain the old face-only registration");
  const facePose = [1, 0, 0.02, 0, 1, -0.01];
  const faceTrack = Array.from({length:72}, () => facePose.slice());
  const registered = Object.assign({}, full, {facePose, clips:{t:think({faceTrack})}});
  const accepted = parseClipManifest(registered, null);
  assert.deepStrictEqual(accepted.facePose, facePose);
  assert.deepStrictEqual(api.clipFace(accepted.clips.t, 0.5), facePose);
  const faceAnchors = Array.from({length:13}, (_, i) => [0.3 + (i % 4) * 0.1, 0.35 + Math.floor(i / 4) * 0.1]);
  assert.deepStrictEqual(parseClipManifest(Object.assign({}, registered, {faceAnchors}), null).faceAnchors, faceAnchors);
  for (const bad of [0, [], faceAnchors.slice(1), [[-0.1, 0.5], ...faceAnchors.slice(1)]]) {
    assert.strictEqual(parseClipManifest(Object.assign({}, registered, {faceAnchors:bad}), null), null,
      "bad reference anchors must not enter the compositor");
  }
  const mouth = {box:[0.35,0.55,0.65,0.8],center:[0.5,0.65]};
  assert.strictEqual(parseClipManifest(Object.assign({}, registered, {faceMode:"mouth",mouth}), null).faceMode,"mouth");
  assert.strictEqual(parseClipManifest(Object.assign({}, registered, {faceMode:"mouth"}), null),null);
  assert.strictEqual(parseClipManifest(Object.assign({}, registered, {faceMode:"mouth",mouth:{box:mouth.box,center:[0,0]}}), null),null);
  assert.strictEqual(parseClipManifest(Object.assign({}, full, {facePose}), null), null, "registered packs need per-frame face poses");
  faceTrack[10] = [1, 2, 0, 0.5, 1, 0];
  assert.strictEqual(parseClipManifest(registered, null), null, "singular facial registrations are refused");
}

// -- the head's seat

{
  const clip = { fps: 10, track: [[1, 0, 0, 0], [1.1, 0.2, 10, -4], [1.2, 0.4, 20, -8]] };
  assert.deepStrictEqual(plain(clipHead({ fps: 10, track: null }, 1)), { scale: 1, turn: 0, x: 0, y: 0 });
  assert.deepStrictEqual(plain(clipHead(null, 1)), { scale: 1, turn: 0, x: 0, y: 0 });
  const half = clipHead(clip, 0.05);
  assert.ok(Math.abs(half.scale - 1.05) < 1e-9 && Math.abs(half.turn - 0.1) < 1e-9 && Math.abs(half.x - 5) < 1e-9 && Math.abs(half.y + 2) < 1e-9);
  assert.strictEqual(clipHead(clip, 99).x, 20, "past the end the last frame holds");
  assert.strictEqual(clipHead(clip, -1).x, 0);
  const faces = {fps:10,faceTrack:[[1,0,0,0,1,0],[0.9,0.1,0.02,0.03,1,-0.02]]};
  assert.deepStrictEqual(api.clipFace(faces,0.05),[0.95,0.05,0.01,0.015,1,-0.01]);
  assert.deepStrictEqual(api.clipFace(faces,99),faces.faceTrack[1]);
}

// -- exits

{
  const loop = parseClipManifest(manifest({ t: think({ exits: [1, 2] }) }), null).clips.t;
  assert.strictEqual(clipNextExit(loop, 0.2, 0.3), 1);
  assert.strictEqual(clipNextExit(loop, 0.8, 0.3), 2, "an exit closer than the margin is too late to fade out for");
  assert.strictEqual(clipNextExit(loop, 2.9, 0.3), 4, "after the last exit it is the first one of the next time round");
  assert.strictEqual(clipNextExit(loop, 3.5, 0.3), 4);
  const once = parseClipManifest(manifest({ g: talk("calm", 3, { exits: [1] }) }), null).clips.g;
  assert.strictEqual(clipNextExit(once, 0.5, 0.3), 1);
  assert.strictEqual(clipNextExit(once, 2.0, 0.3), once.last, "a gesture ends at its end");
  assert.strictEqual(clipNextExit(once, once.last - 0.1, 0.3), once.last, "even when that is closer than the margin");
}

// -- what is being said

{
  const sentences = clipSentences("Hello there! How can I help you today? Tell me more about what you need.");
  assert.strictEqual(sentences.length, 3);
  assert.deepStrictEqual(sentences.map((s) => s.words), [2, 6, 7]);
  assert.ok(Math.abs(sentences[0].to - 2 / 15) < 1e-9 && sentences[1].from === sentences[0].to && Math.abs(sentences[2].to - 1) < 1e-9);
  assert.deepStrictEqual(sentences.map((s, i) => clipMood(s, i)), ["greet", "ask", "explain"]);
  assert.strictEqual(clipMood({ text: "Hello there!", words: 2 }, 1), "emphasis", "a greeting is only one at the start");
  assert.strictEqual(clipMood({ text: "Yes, of course.", words: 3 }, 1), "calm");
  assert.strictEqual(clipMood({ text: "word ".repeat(20).trim() + ".", words: 20 }, 1), "explain");
  assert.strictEqual(clipMood({ text: "one two three four five six seven eight.", words: 8 }, 1), "emphasis");
  assert.strictEqual(clipMood({ text: "one two three four five six seven eight.", words: 8 }, 2), "explain");
  assert.strictEqual(clipMood({ text: "Is it? Really?)", words: 3 }, 1), "ask");
  assert.strictEqual(clipMood({ text: "Could you explain the new schedule", words: 6 }, 1), "ask");
  assert.strictEqual(clipMood({ text: "First, check the camera framing.", words: 5 }, 1), "explain");
  assert.strictEqual(clipMood({ text: "I understand that this is frustrating.", words: 7 }, 1), "calm");
  assert.strictEqual(clipMood({ text: "Congratulations on the launch.", words: 4 }, 1), "emphasis");
  assert.strictEqual(clipSentences("").length, 0);
  assert.strictEqual(clipSentences("...  !?").length, 0);
  assert.strictEqual(clipSentences("No punctuation at all").length, 1);
}

// -- formats

{
  const types = { mp4: "video/mp4", webm: "video/webm" };
  assert.strictEqual(clipFormat(types, () => "probably"), "mp4");
  assert.strictEqual(clipFormat(types, (t) => (t === "video/webm" ? "probably" : "maybe")), "webm", "a probably beats a maybe");
  assert.strictEqual(clipFormat(types, (t) => (t === "video/webm" ? "maybe" : "")), "webm");
  assert.strictEqual(clipFormat(types, () => ""), null);
  assert.strictEqual(clipFormat(null, () => "probably"), null);
}

// -- the director

{
  const sim = new Sim({
    resting: think({ role: "idle", mood: "idle" }),
    attentive: think({ role: "listen", mood: "listen" }),
    considering: think(),
    palms: talk("explain", 2),
  }, { idle: true });
  sim.run(1);
  assert.strictEqual(sim.director.cur.name, "resting");
  sim.director.setState("LISTENING", sim.now);
  sim.run(1);
  assert.strictEqual(sim.director.cur.name, "attentive", "typing or recording selects the listening animation");
  sim.director.setState("THINKING", sim.now);
  sim.director.ask(sim.now);
  sim.run(1);
  assert.strictEqual(sim.director.cur.name, "considering", "sending selects a different thinking animation");
  sim.director.speak("Let me explain.");
  sim.director.length(8);
  sim.director.speechStart(sim.now);
  sim.director.setState("SPEAKING", sim.now);
  sim.run(1);
  assert.strictEqual(sim.director.cur.name, "palms");
  sim.director.finish();
  sim.director.setState("LISTENING", sim.now);
  sim.run(2);
  assert.strictEqual(sim.director.cur.name, "attentive", "active input resumes listening, not idle");
  sim.director.setState("IDLE", sim.now);
  sim.run(1);
  assert.strictEqual(sim.director.cur.name, "resting");
  assert.ok(maxJump(sim.log) < STEP, "state changes retain video-clocked smooth fades");
}

{
  const available = new Set(["resting", "legacy"]);
  const sim = new Sim({
    resting: think({ role: "idle" }), attentive: think({ role: "listen" }), legacy: think(),
  }, { idle: true, ready: (name) => available.has(name) });
  sim.run(1);
  sim.director.setState("LISTENING", sim.now);
  sim.run(1);
  assert.strictEqual(sim.director.cur.name, "legacy", "a missing listening video uses the legacy quiet fallback");
  available.add("attentive");
  sim.run(1);
  assert.strictEqual(sim.director.cur.name, "attentive", "a late-loaded state clip replaces its fallback");
}

{
  const available = new Set(["resting", "thinking"]);
  const sim = new Sim({ resting: think({ role: "idle" }), thinking: think() },
    { idle: true, ready: (name) => available.has(name) });
  sim.director.ask(sim.now);
  sim.run(1);
  available.delete("thinking");
  sim.run(0.8);
  assert.strictEqual(sim.director.cur.name, "thinking",
    "a playing state clip must not be replaced just because its decoder briefly loses readiness at a loop seek");
  assert.strictEqual(sim.director.cur.released, false);
}

{
  const sim = new Sim({ palms: talk("explain", 2) });
  sim.director.ask(sim.now);
  sim.director.speak("For example, this.");
  sim.director.length(0.9);
  sim.director.speechStart(sim.now);
  sim.run(0.8);
  assert.deepStrictEqual(sim.played(), ["palms"], "a short reply can open with a short open-hand gesture");
}

{
  const sim = new Sim({ short: talk("greet", 2), long: talk("explain", 5) });
  sim.director.recent = ["short"];
  sim.director.ask(sim.now);
  sim.director.speak("Hello there.");
  sim.director.length(1.5);
  sim.step();
  assert.ok(sim.prepared[sim.prepared.length - 1].includes("short"),
    "knowing a short reply's length must preload a fitting gesture even when it was recently used");
}

{
  const sim = new Sim({ palms: talk("explain", 2) }, { ready: () => sim.now >= 110 });
  sim.director.ask(sim.now);
  sim.director.speak("For example, we can start with this.");
  sim.director.length(4);
  sim.director.speechStart(sim.now);
  sim.director.progress(0.8);
  sim.run(11);
  assert.deepStrictEqual(sim.played(), ["palms"], "rebuffering must not consume the remaining speech on the wall clock");
}

{
  const sim = new Sim(PACK, { idle: true });
  sim.run(2);
  assert.ok(sim.prepared.some((names) => names.length === 2 && names.every((name) => PACK[name].role === "think")));
  assert.strictEqual(sim.director.cur.clip.role, "think", "the base video keeps listening and blinking before any message");
  assert.strictEqual(sim.director.cur.weight, 1);
  sim.director.ask(sim.now);
  sim.director.speak("Let me explain how this works.");
  sim.director.length(10);
  sim.director.speechStart(sim.now);
  sim.run(3);
  assert.ok(sim.played().some((name) => PACK[name].role === "talk"), "speaking still selects conversational gestures");
  sim.director.finish();
  sim.run(3);
  assert.strictEqual(sim.director.cur.clip.role, "think", "listening resumes after the reply");
}

{
  const sim = new Sim({});
  assert.strictEqual(sim.director.usable, false);
  sim.director.ask(sim.now);
  sim.run(3);
  assert.ok(sim.log.every((entry) => entry.name === null && entry.weight === 0), "without clips there is nothing to draw");
  assert.strictEqual(sim.commands.length, 0);
}

{
  // The thinking clip begins on the first tick after the message is sent.
  const sim = new Sim(PACK);
  assert.deepStrictEqual(sim.director.tick(sim.now, null).commands.map((c) => c.type), ["prepare"], "the thinking clip is wanted from the start");
  sim.director.ask(sim.now);
  const first = sim.step();
  assert.deepStrictEqual(first.commands.map((c) => c.type), ["prepare", "play"]);
  assert.ok(/^think-/.test(first.commands[1].name));
  assert.strictEqual(first.weight, 0, "nothing is drawn before the video shows a frame");
  sim.run(1);
  assert.ok(sim.log[sim.log.length - 1].weight > 0.99);
  assert.ok(maxJump(sim.log) < STEP);
}

{
  // Nothing is asked: nothing plays.
  const sim = new Sim(PACK);
  sim.run(5);
  assert.deepStrictEqual(sim.played(), []);
}

{
  // The thinking clip loops without a jump while the reply is prepared, then gives way to a gesture when the speech begins.
  const sim = new Sim(PACK);
  sim.director.ask(sim.now);
  sim.run(6);
  assert.ok(sim.log.slice(60).every((entry) => entry.weight > 0.999), "a loop holds full weight across its wraps");
  assert.ok(sim.log.some((entry, i) => i && entry.name && entry.time < sim.log[i - 1].time - 1), "it did wrap");
  sim.run(4);
  assert.ok(sim.played().length >= 2, "after a while another thinking clip takes over");
  const before = sim.log.length;
  sim.director.speak("Hello and welcome to the demonstration of the system today. This is a short answer about the thing you asked.");
  sim.director.length(8);
  sim.director.speechStart(sim.now);
  sim.run(9);
  const names = sim.played();
  const gesture = names.find((name) => !/^think/.test(name));
  assert.ok(gesture, `a gesture follows the thinking: ${names}`);
  assert.strictEqual(gesture, "greet", "the first sentence is a greeting");
  const started = sim.log.findIndex((entry, i) => i >= before && entry.name === gesture);
  assert.ok(sim.log.slice(before, started).every((entry) => !entry.name || /^think/.test(entry.name)));
  assert.ok(maxJump(sim.log) < STEP, `the weight never jumps: ${maxJump(sim.log)}`);
  assert.ok(started - before < 5 * 60, "the thinking is let go within about its own length, so the gesture is not late");
}

{
  // The thinking clip is let go at an exit and not in the middle of a move: its weight is 1 until the fade at the exit.
  const sim = new Sim({ t: think({ duration: 4, exits: [1, 3] }), g: talk("explain", 3) });
  sim.director.ask(sim.now);
  sim.run(2.55);
  sim.director.speak("A reply of some length that goes on for a while.");
  sim.director.length(10);
  sim.director.speechStart(sim.now);
  const at = sim.log.length;
  sim.run(2);
  const fading = sim.log.slice(at).findIndex((entry) => entry.weight < 0.99);
  const t = sim.log[at + fading].time;
  assert.ok(t > 2.6 && t <= 3, `fades towards the exit at 3 s (media time ${t})`);
}

{
  // Speech handoff is bounded even when the next authored exit is far away.
  const sim = new Sim({ t: think({ duration: 4, exits: [3] }), g: talk("explain", 3) });
  sim.director.ask(sim.now);
  sim.run(1.5);
  sim.director.speak("Here is the answer.");
  sim.director.length(5);
  const speechAt = sim.now;
  sim.director.speechStart(speechAt);
  sim.run(1.2);
  const gestureStart = sim.commands.find((c) => c.type === "play" && c.name === "g");
  assert.ok(gestureStart, "the first gesture follows the thinking clip");
  assert.ok(gestureStart.at - speechAt < 1.05, `handoff takes less than a second (${gestureStart.at - speechAt}s)`);
  assert.ok(maxJump(sim.log) < STEP * 2, "the fast handoff remains a smooth fade");
}

{
  // Only clips that are ready are played, and a clip is not repeated at once.
  const ready = new Set(["think-a", "calm", "explain"]);
  const sim = new Sim(PACK, { ready: (name) => ready.has(name) });
  sim.director.ask(sim.now);
  sim.run(1);
  sim.director.speak("Is this working? It should be. Let me show you how it works in a bit more detail than before, please.");
  sim.director.length(30);
  sim.director.speechStart(sim.now);
  sim.run(30);
  const names = sim.played();
  assert.ok(names.every((name) => ready.has(name)), `played ${names}`);
  const gestures = names.filter((name) => !/^think/.test(name));
  assert.ok(gestures.length >= 4, `a long speech has several gestures: ${gestures}`);
  for (let i = 1; i < gestures.length; i++) assert.notStrictEqual(gestures[i], gestures[i - 1], "no gesture twice in a row when there is a choice");
}

{
  // A pack with only one gesture still uses it again and again.
  const sim = new Sim({ t: think(), g: talk("explain", 2) });
  sim.director.ask(sim.now);
  sim.run(0.5);
  sim.director.speak("A long reply that goes on and on and on.");
  sim.director.length(20);
  sim.director.speechStart(sim.now);
  sim.run(20);
  assert.ok(sim.played().filter((name) => name === "g").length >= 3);
}

{
  // The gesture follows the mood of what is said: a question gets the questioning one, from the pack's clips.
  for (const [text, mood] of [
    ["Hello and welcome to the demonstration of the system today.", "greet"],
    ["Would you like to see how it works?", "ask"],
    ["That is wonderful news!", "emphasis"],
    ["Okay.", "calm"],
    ["The system listens to what you say and answers in a natural voice while the avatar moves its body in time.", "explain"],
  ]) {
    const sim = new Sim(PACK);
    sim.director.ask(sim.now);
    sim.run(0.5);
    sim.director.speak(text);
    sim.director.length(6);
    sim.director.speechStart(sim.now);
    sim.run(6);
    const first = sim.played().find((name) => !/^think/.test(name));
    assert.strictEqual(first, mood, `"${text}" -> ${first}`);
  }
  // A mood the pack has no clip for takes the nearest.
  const sim = new Sim({ "think-x": think(), calm: talk("calm", 2), explain: talk("explain", 2) });
  sim.director.ask(sim.now);
  sim.run(0.5);
  sim.director.speak("Would you like to see how it works?");
  sim.director.length(6);
  sim.director.speechStart(sim.now);
  sim.run(6);
  assert.strictEqual(sim.played().find((name) => !/^think/.test(name)), "explain");
}

{
  // The gesture moves on with the speech: a long reply gets the mood of each of its sentences in turn.
  const sim = new Sim(PACK, { seed: 3 });
  sim.director.ask(sim.now);
  sim.run(0.5);
  sim.director.speak("Hello and welcome to the demonstration of the system today! Would you like to see how it works? Please tell me what you think about all of this.");
  sim.director.length(12);
  sim.director.speechStart(sim.now);
  sim.run(12);
  const gestures = sim.played().filter((name) => !/^think/.test(name));
  assert.strictEqual(gestures[0], "greet");
  assert.ok(gestures.length >= 2 && gestures.indexOf("ask") > 0, `later sentences are looked at too: ${gestures}`);
}

{
  // No gesture begins with less speech than the tail guard, and none that would last far beyond the speech.
  const sim = new Sim(PACK);
  sim.director.ask(sim.now);
  sim.run(0.5);
  sim.director.speak("Yes.");
  sim.director.length(api.CLIP_MIN_TAIL - 0.1);
  sim.director.speechStart(sim.now);
  sim.run(6);
  assert.deepStrictEqual(sim.played().filter((name) => !/^think/.test(name)), [], "too short for a gesture");

  const second = new Sim(PACK);
  second.director.ask(second.now);
  second.run(0.5);
  second.director.speak("Hello there. This is a short one.");
  second.director.length(2);
  second.director.speechStart(second.now);
  second.run(8);
  const longest = Math.max(...second.played().filter((name) => !/^think/.test(name)).map((name) => second.clips[name].duration));
  assert.ok(longest <= 2 + 1.5 + 1e-9, `a gesture of ${longest} s for 2 s of speech`);

  // A gesture does not begin in the last stretch of a speech that has gone on.
  const third = new Sim({ t: think(), g: talk("calm", 1) });
  third.director.ask(third.now);
  third.run(0.5);
  third.director.speak("A reply of a certain length that has a few gestures in it for sure.");
  third.director.length(10);
  third.director.speechStart(third.now);
  third.run(12);
  const starts = third.commands.filter((c) => c.type === "play" && c.name === "g").map((c) => c.at - third.log[0].at);
  assert.ok(starts.length >= 3);
  assert.ok(starts.every((s) => s <= 0.5 + 10 - api.CLIP_MIN_TAIL + 0.05), `no start in the last ${api.CLIP_MIN_TAIL} s: ${starts}`);
}

{
  // The reply is complete: the gesture that is playing ends at an exit, and no other begins.
  const sim = new Sim({ t: think(), g: talk("explain", 4, { exits: [1.5, 3] }) });
  sim.director.ask(sim.now);
  sim.run(0.5);
  sim.director.speak("A reply of a certain length that has a few gestures in it for sure.");
  sim.director.length(30);
  sim.director.speechStart(sim.now);
  sim.run(2.2);
  const playing = sim.log[sim.log.length - 1].name;
  assert.strictEqual(playing, "g");
  const plays = sim.played().length;
  sim.director.finish();
  const mark = sim.log.length;
  sim.run(4);
  assert.strictEqual(sim.played().length, plays, "nothing new begins");
  const gone = sim.log.slice(mark).findIndex((entry) => entry.name === null);
  assert.ok(gone > 0 && gone < 2 * 60, "the gesture ends at its next exit");
  assert.ok(maxJump(sim.log) < STEP);
}

{
  // Stop: everything fades quickly and nothing begins again.
  const sim = new Sim(PACK);
  sim.director.ask(sim.now);
  sim.run(2);
  assert.ok(sim.log[sim.log.length - 1].weight > 0.99);
  sim.director.stop();
  const mark = sim.log.length;
  sim.run(1);
  const out = sim.log.slice(mark);
  const gone = out.findIndex((entry) => entry.weight === 0);
  assert.ok(gone > 0 && gone <= Math.ceil((api.CLIP_FAST_FADE + 0.1) * 60), `stopped within ${gone} ticks`);
  assert.ok(out.every((entry, i) => !i || entry.weight <= out[i - 1].weight + 1e-9), "it only fades");
  assert.ok(maxJump(out) < 0.2);
  assert.strictEqual(sim.played().length, 1);
}

{
  // A message sent while the last gesture is still playing: it is let go, and the thinking begins after it.
  const sim = new Sim({ t: think(), g: talk("explain", 4, { exits: [1.5, 3] }) });
  sim.director.ask(sim.now);
  sim.run(0.5);
  sim.director.speak("A reply of a certain length that has a few gestures in it for sure.");
  sim.director.length(30);
  sim.director.speechStart(sim.now);
  sim.run(2.2);
  sim.director.finish();
  sim.director.ask(sim.now);
  sim.run(5);
  const names = sim.played();
  assert.strictEqual(names[names.length - 1].slice(0, 1), "t", "and it thinks again");
  assert.ok(maxJump(sim.log) < STEP);
}

{
  // A clip that never shows a frame is given up after the timeout, and another one takes its place.
  const sim = new Sim({ a: think(), b: think() });
  sim.director.failed = () => false; // the pool does not know: the video just never shows anything
  sim.director.ask(sim.now);
  sim.step();
  const first = sim.playing.name;
  sim.dead.add(first);
  sim.run(api.CLIP_START_TIMEOUT + 3);
  const names = sim.played();
  assert.strictEqual(names.filter((n) => n === first).length, 1, "a clip that does not play is not tried again");
  assert.ok(names.length === 2 && names[1] !== first && sim.director.broken.has(first));
  assert.ok(sim.log[sim.log.length - 1].weight > 0.99, "and the other one is on screen");
}

{
  // A clip that stops in the middle is given up too.
  const sim = new Sim({ a: think(), b: think() }, { seed: 5 });
  sim.director.ask(sim.now);
  sim.run(1);
  const first = sim.playing.name;
  sim.frozen.add(first);
  sim.run(api.CLIP_STALL + 0.5);
  assert.ok(sim.director.broken.has(first), "frozen video given up");
  sim.run(1.5);
  const names = sim.played();
  assert.ok(names.length === 2 && names[1] !== first, `the other clip takes over: ${names}`);
}

{
  // A page that was not drawn for a while (a hidden tab) is no fault of the video.
  const sim = new Sim(PACK);
  sim.director.ask(sim.now);
  sim.run(1);
  sim.now += 20; // the tab slept; the video, as videos do, went on or paused
  sim.playing.at += 20;
  sim.run(1);
  assert.strictEqual(sim.director.broken.size, 0);
  assert.ok(sim.log[sim.log.length - 1].weight > 0.99);
}

{
  // More motion, shorter rests: the same reply has more gestures at motion 2 than at motion 0.25.
  const count = (motion) => {
    const sim = new Sim({ t: think(), g: talk("calm", 1.5), h: talk("explain", 1.5) }, { motion, seed: 11 });
    sim.director.ask(sim.now);
    sim.run(0.5);
    sim.director.speak("A very long reply that has a great many words in it so that the avatar keeps on talking.");
    sim.director.length(40);
    sim.director.speechStart(sim.now);
    sim.run(40);
    return sim.played().filter((name) => name !== "t").length;
  };
  const lively = count(2), calm = count(0.25);
  assert.ok(lively > calm + 3, `${lively} against ${calm}`);
}

{
  // Two thinking clips: after a long wait the avatar changes what it does while it thinks.
  const sim = new Sim({ a: think(), b: think({ duration: 5, exits: [2.5] }), g: talk("calm", 2) });
  sim.director.ask(sim.now);
  sim.run(api.CLIP_THINK_SWAP + 6);
  const names = sim.played();
  assert.ok(names.length >= 2 && names[0] !== names[1], `${names}`);
  assert.ok(maxJump(sim.log) < STEP);
}

{
  // Gestures are not played while the thinking continues, and the speech start only counts once.
  const sim = new Sim(PACK);
  sim.director.ask(sim.now);
  sim.director.speak("Hello there, this is the answer.");
  sim.director.length(5);
  sim.run(2);
  assert.ok(sim.played().every((name) => /^think/.test(name)), "the speech has not begun");
  const at = sim.now + 0.5;
  sim.director.speechStart(at);
  sim.director.speechStart(at + 100);
  assert.strictEqual(sim.director.turn.at, at);
}

{
  // A think-only pack just thinks, and a gesture-only pack has no filler: both stay well-behaved.
  const only = new Sim({ t: think() });
  only.director.ask(only.now);
  only.run(1);
  only.director.speak("A reply of a certain length.");
  only.director.length(6);
  only.director.speechStart(only.now);
  only.run(8);
  assert.strictEqual(only.log[only.log.length - 1].name, null);
  const bare = new Sim({ g: talk("calm", 2) });
  bare.director.ask(bare.now);
  bare.run(1);
  assert.deepStrictEqual(bare.played(), []);
  bare.director.speak("A reply of a certain length.");
  bare.director.length(6);
  bare.director.speechStart(bare.now);
  bare.run(6);
  assert.ok(bare.played().length >= 1);
}

{
  // The weight of a clip that plays once is a smooth bump: 0 at both ends, 1 in the middle.
  const sim = new Sim({ t: think(), g: talk("explain", 3) });
  sim.director.ask(sim.now);
  sim.run(0.5);
  sim.director.speak("A reply of a certain length that has a few gestures in it for sure.");
  sim.director.length(30);
  sim.director.speechStart(sim.now);
  sim.run(8);
  const mine = sim.log.filter((entry) => entry.name === "g").slice(0, 150);
  assert.ok(mine.length > 60);
  assert.ok(mine[0].weight < 0.05, "it fades in");
  assert.ok(Math.max(...mine.map((entry) => entry.weight)) > 0.999);
  assert.ok(maxJump(sim.log) < STEP);
}

// -- the pool of videos

class FakeVideo {
  constructor() {
    this.listeners = {};
    this.currentTime = 0;
    this.readyState = 0;
    this.paused = true;
    this.ended = false;
    this.loop = false;
    this.src = "";
    this.removed = false;
    this.loads = 0;
    this.frameWatch = null;
  }

  addEventListener(name, handler) {
    this.listeners[name] = handler;
  }

  fire(name) {
    if (this.listeners[name]) this.listeners[name]();
  }

  load() {
    this.loads++;
    this.readyState = 0;
  }

  loaded() {
    this.readyState = 4;
    this.fire("loadeddata");
  }

  play() {
    this.paused = false;
    return this.refuse ? Promise.reject(new Error("blocked")) : Promise.resolve();
  }

  pause() {
    this.paused = true;
  }

  removeAttribute() {
    this.src = "";
  }

  remove() {
    this.removed = true;
  }

  requestVideoFrameCallback(callback) {
    this.frameWatch = callback;
    return 1;
  }

  frame(mediaTime) {
    const watch = this.frameWatch;
    this.frameWatch = null;
    if (watch) watch(0, { mediaTime });
  }
}

function pool({ frames = true, files = {}, fail = new Set(), captureFrame, clips } = {}) {
  const videos = [];
  const revoked = [];
  const fetched = [];
  const spec = parseClipManifest(manifest(clips || {
    "think-a": think(), "think-b": think(), calm: talk("calm", 3), explain: talk("explain", 4), ask: talk("ask", 3),
  }), null);
  for (const name of Object.keys(files)) spec.clips[name].files = files[name];
  const reels = api.reels(spec, {
    base: "/assets/x/",
    download: async (url) => {
      fetched.push(url);
      if (fail.has(url)) throw new Error("gone");
      return { url };
    },
    makeVideo: () => {
      const video = new FakeVideo();
      if (!frames) video.requestVideoFrameCallback = undefined;
      videos.push(video);
      return video;
    },
    captureFrame,
    url: { createObjectURL: (blob) => `blob:${blob.url}`, revokeObjectURL: (url) => revoked.push(url) },
    canPlay: () => "probably",
    makeBlob: (blob, type) => ({ url: `${blob.url}|${type}` }),
  });
  return { reels, videos, revoked, fetched, spec };
}

(async () => {
  {
    const { reels, spec, fetched } = pool({ clips: {
      gesture: talk("explain", 2),
      resting: think({ role: "idle", files: { mp4: "idle.mp4" } }),
      attentive: think({ role: "listen", files: { mp4: "listen.mp4" } }),
      considering: think({ files: { mp4: "think.mp4" } }),
    } });
    await reels.load();
    await reels.background;
    assert.deepStrictEqual(fetched.slice(0, 3), ["/assets/x/idle.mp4", "/assets/x/listen.mp4", "/assets/x/think.mp4"]);
    const director = api.director(spec.clips, { idle: true, ready: (n) => reels.ready(n) });
    const stage = api.stage({ manifest: spec, reels, director, compositor: { setClip() {}, setClipFrame() {}, clearClips() {} } });
    let now = 0;
    const run = (seconds) => {
      for (let i = 0; i < seconds / DT; i++) {
        now += DT;
        for (const slot of reels.slots) {
          if (slot.state === "loading") slot.video.loaded();
          if (!slot.video.paused) {
            slot.video.currentTime = (slot.video.currentTime + DT) % spec.clips[slot.name].duration;
            slot.video.frame(slot.video.currentTime);
          }
        }
        stage.update(now);
      }
    };
    run(1);
    assert.strictEqual(director.cur.name, "resting");
    stage.setState("LISTENING", now);
    run(1);
    assert.strictEqual(director.cur.name, "attentive");
    stage.setState("THINKING", now);
    stage.ask(now);
    run(1);
    assert.strictEqual(director.cur.name, "considering");
    stage.speak("First, let me explain.");
    stage.length(10);
    stage.speechStart(now);
    run(1);
    assert.strictEqual(director.cur.name, "gesture", "quiet preloads leave room for a speaking video");
    assert.strictEqual(reels.slots.length, 3);
    stage.dispose();
  }

  {
    const { reels, fetched } = pool({ files: { calm: { webm: "calm.webm" } } });
    await reels.load();
    assert.strictEqual(reels.format, "mp4");
    assert.deepStrictEqual(fetched.slice(0, 2).sort(), ["/assets/x/t.mp4", "/assets/x/t.mp4"], "the thinking clips come first");
    await reels.background;
    assert.ok(reels.failed("calm"), "a clip without a file in the chosen format is lost, the others go on");
    assert.ok(!reels.failed("explain"));
    assert.strictEqual(reels.urls.get("explain"), "blob:/assets/x/g.mp4|video/mp4");
    assert.throws(() => api.reels(parseClipManifest(manifest({ a: think() }), null), { canPlay: () => "" }), /plays none/);
  }

  {
    const { reels, videos, fetched } = pool({ fail: new Set(["/assets/x/g.mp4"]) });
    await reels.load();
    await reels.background;
    assert.ok(reels.failed("calm") && reels.failed("explain") && reels.failed("ask"), "failed downloads are lost");
    assert.ok(!reels.failed("think-a"));
    reels.prepare(["calm", "think-a"]);
    assert.strictEqual(videos.length, 1, "a lost clip is not put in the pool");
    assert.ok(fetched.length >= 5);
  }

  {
    // Slots: a clip is loaded when wanted, ready when its video has data, and the pool never grows past its size.
    const { reels, videos } = pool();
    await reels.load();
    await reels.background;
    assert.ok(!reels.ready("think-a"));
    reels.prepare(["think-a", "calm"]);
    assert.strictEqual(videos.length, 2);
    assert.ok(!reels.ready("think-a"), "not ready before its data is in");
    videos[0].loaded();
    assert.ok(reels.ready("think-a") && !reels.ready("calm"));
    assert.strictEqual(videos[0].loop, true);
    assert.strictEqual(videos[1].loop, false);
    reels.prepare(["think-a", "calm"]);
    assert.strictEqual(videos[0].loads, 1, "asking again for what is there costs nothing");
    reels.prepare(["explain"]);
    assert.strictEqual(videos.length, 3);
    videos[1].loaded();
    videos[2].loaded();
    // Play think-a; something else is wanted: the oldest free slot goes, never the playing one.
    reels.play("think-a");
    assert.strictEqual(videos[0].paused, false);
    reels.prepare(["ask"]);
    assert.strictEqual(videos.length, api.CLIP_SLOTS, "the pool does not grow");
    assert.ok(reels.slots.find((s) => s.name === "think-a"), "the playing clip stays");
    assert.ok(reels.slots.find((s) => s.name === "ask"));
    assert.ok(!reels.slots.find((s) => s.name === "calm"), "the one used longest ago went");
    // A thinking clip is the last to go.
    reels.park("think-a");
    reels.prepare(["ask", "explain"]);
    assert.ok(reels.slots.find((s) => s.name === "explain") && reels.slots.find((s) => s.name === "ask"));
    reels.prepare(["calm"]);
    assert.ok(reels.slots.find((s) => s.name === "calm"));
    assert.ok(reels.slots.find((s) => s.name === "think-a"), "the thinking clip was kept, a gesture went instead");
  }

  {
    // Playing: the first frame is not reported until the video shows it; frames come with their media time, once each.
    const { reels, videos } = pool();
    await reels.load();
    await reels.background;
    reels.prepare(["calm"]);
    videos[0].loaded();
    videos[0].currentTime = 1.5;
    reels.play("calm");
    assert.strictEqual(videos[0].currentTime, 0, "a clip always begins at its start");
    assert.strictEqual(reels.media(), null);
    assert.strictEqual(reels.frame(), null);
    videos[0].frame(0);
    assert.deepStrictEqual(plain(reels.media()), { name: "calm", time: 0, ended: false });
    assert.strictEqual(reels.frame(), videos[0]);
    assert.strictEqual(reels.frame(), null, "a frame is taken once");
    videos[0].frame(0.5);
    assert.strictEqual(reels.media().time, 0.5);
    assert.strictEqual(reels.frame(), videos[0]);
    videos[0].ended = true;
    assert.strictEqual(reels.media().ended, true);
    reels.park("calm");
    assert.strictEqual(reels.media(), null);
    assert.strictEqual(videos[0].paused, true);
    assert.ok(reels.ready("calm"), "parked, it is ready to play again");
    reels.play("calm");
    assert.strictEqual(reels.media(), null, "the old frames do not count for the new play");
    videos[0].frame(0);
    assert.strictEqual(reels.media().time, 0);
  }

  {
    // The video can advance after its callback, before the compositor gets its turn.
    const captured = [];
    const { reels, videos } = pool({ captureFrame: (video, time) => {
      const snapshot = { pixels: video.currentTime, time, closed: false, close() { this.closed = true; } };
      captured.push(snapshot);
      return snapshot;
    } });
    await reels.load();
    reels.prepare(["think-a"]);
    videos[0].loaded();
    reels.play("think-a");
    videos[0].currentTime = 0.25;
    videos[0].frame(0.25);
    videos[0].currentTime = 0.75;
    assert.strictEqual(reels.media().time, 0.25);
    assert.strictEqual(reels.frame().pixels, 0.25, "pixels must belong to the tracked frame, not the advancing video");
    videos[0].frame(0.75);
    assert.ok(captured[0].closed, "replaced snapshots release their decoder resources");
    assert.strictEqual(reels.frame().pixels, 0.75);
    reels.park("think-a");
    assert.ok(captured[1].closed, "parking releases the retained frame");
    reels.play("think-a");
    videos[0].frame(0);
    reels.dispose();
    assert.ok(captured[2].closed, "disposing releases the last retained frame");
    videos[0].frame(1);
    assert.strictEqual(captured.length, 3, "a late callback cannot allocate after disposal");
  }

  {
    // Without requestVideoFrameCallback the video's own time has to do.
    const { reels, videos } = pool({ frames: false });
    await reels.load();
    reels.prepare(["think-a"]);
    videos[0].loaded();
    reels.play("think-a");
    assert.strictEqual(reels.media(), null, "no time has passed: no frame");
    videos[0].currentTime = 0.2;
    assert.strictEqual(reels.media().time, 0.2);
    assert.strictEqual(reels.frame(), videos[0]);
    assert.strictEqual(reels.frame(), null);
    videos[0].currentTime = 0.25;
    assert.strictEqual(reels.frame(), videos[0]);
  }

  {
    const { reels, videos } = pool();
    await reels.load();
    reels.prepare(["think-a"]);
    videos[0].loaded();
    const reject = [];
    videos[0].play = () => new Promise((resolve, fail) => reject.push(fail));
    reels.play("think-a");
    reels.park("think-a");
    reels.play("think-a");
    reject[0](new Error("Interrupted by an intentional pause"));
    await new Promise((resolve) => setImmediate(resolve));
    assert.ok(!reels.failed("think-a"), "an old aborted play request must not disable a reusable animation");
    assert.strictEqual(reels.playing.name, "think-a", "a late rejection must not release the new play request");
    reject[1](new Error("Current playback failed"));
    await new Promise((resolve) => setImmediate(resolve));
    assert.ok(reels.failed("think-a"), "a failure of the current play request is still reported");
    reels.dispose();
  }

  {
    // A video that errors, or will not play, is lost for good and the slot is freed.
    const { reels, videos } = pool();
    await reels.load();
    await reels.background;
    reels.prepare(["calm"]);
    videos[0].fire("error");
    assert.ok(reels.failed("calm") && !reels.ready("calm"));
    reels.prepare(["explain"]);
    videos[0].loaded();
    videos[0].refuse = true;
    reels.play("explain");
    await new Promise((resolve) => setImmediate(resolve));
    assert.ok(reels.failed("explain"));
    assert.strictEqual(reels.media(), null);
  }

  {
    // Disposing gives everything back.
    const { reels, videos, revoked } = pool();
    await reels.load();
    await reels.background;
    reels.prepare(["think-a"]);
    videos[0].loaded();
    reels.play("think-a");
    reels.dispose();
    assert.ok(videos[0].removed && videos[0].paused);
    assert.strictEqual(revoked.length, 5);
    assert.strictEqual(reels.media(), null);
  }

  // -- the stage

  {
    const sets = [], frames = [];
    const compositor = { setClip: (state) => sets.push(plain(state)), setClipFrame: (video) => frames.push(video),
      clearClips: () => sets.push({ mix: 0, head: null }) };
    const clips = parseClipManifest(manifest({
      t: think({ track: Array.from({ length: 72 }, (_, i) => [1 + i / 1000, 0, i, 0]) }),
    }), null);
    const calls = [];
    let fresh = null;
    const reels = {
      prepare: (names) => calls.push(["prepare", plain(names)]), play: (name) => calls.push(["play", name]), park: (name) => calls.push(["park", name]),
      media: () => ({ name: "t", time: 1, ended: false }), frame: () => fresh, failed: () => false, ready: () => true, dispose: () => calls.push(["dispose"]),
    };
    const director = api.director(clips.clips, { ready: () => true, random: () => 0.5 });
    const stage = api.stage({ manifest: clips, reels, director, compositor });
    stage.ask(10);
    assert.strictEqual(stage.update(10.01), 0);
    assert.deepStrictEqual(calls.filter((c) => c[0] !== "prepare"), [["play", "t"]]);
    assert.deepStrictEqual(sets[0], { mix: 0, head: { scale: 1, turn: 0, x: 0, y: 0 } }, "nothing is drawn before a frame is shown");
    fresh = { frame: 1 };
    const weight = stage.update(10.5);
    assert.ok(weight > 0.9 && stage.weight === weight);
    assert.deepStrictEqual(frames, [fresh], "a fresh frame goes to the compositor");
    const last = sets[sets.length - 1];
    assert.ok(Math.abs(last.head.x - 24) < 1e-9 && last.mix === weight, "the head follows the frame shown: media time 1 s is frame 24");
    fresh = null;
    stage.update(10.52);
    assert.strictEqual(frames.length, 1, "no new frame, no upload");
    stage.dispose();
    assert.deepStrictEqual(sets[sets.length - 1], { mix: 0, head: null });
    assert.deepStrictEqual(calls[calls.length - 1], ["dispose"]);
  }

  {
    // The pool, the director and the stage as the player has them together: the gesture the director picks plays even when the
    // pool is making room, at the same moment, for the one the sentence asks for (here a question, whose clip is not in yet).
    const { reels, videos, spec } = pool();
    await reels.load();
    await reels.background;
    const director = api.director(spec.clips, { ready: (n) => reels.ready(n), failed: (n) => reels.failed(n), random: () => 0.5 });
    const stage = api.stage({ manifest: spec, reels, director, compositor: { setClip() {}, setClipFrame() {}, clearClips() {} } });
    let now = 10;
    const began = new Map();
    // Every 1/24 s the picture of each video that plays is shown, with its media time.
    const run = (seconds) => {
      for (const end = now + seconds; now < end;) {
        now += 1 / 24;
        stage.update(now);
        for (const video of videos) {
          const slot = reels.slots.find((s) => s.video === video);
          if (video.paused || !slot || slot.state !== "playing") {
            began.delete(video);
            continue;
          }
          if (!began.has(video)) began.set(video, now);
          video.frame((now - began.get(video)) % spec.clips[slot.name].duration);
        }
      }
    };
    stage.ask(now);
    stage.speak("Fine. Would you like to know more about how this works for you today?");
    run(0.1);
    videos.forEach((video) => video.loaded());
    run(3.4);
    assert.strictEqual(director.cur && director.cur.name, "think-a", "it thinks while the reply is awaited");
    stage.length(9);
    stage.speechStart(now);
    run(2.5);
    assert.strictEqual(director.cur && director.cur.name, "explain", "the thinking is over, the gesture goes on");
    assert.strictEqual(reels.playing && reels.playing.name, "explain", "its video is the one that plays");
    assert.ok(director.cur.seen && director.cur.weight > 0.99, "and it is shown");
    assert.strictEqual(director.broken.size, 0);
    assert.ok(reels.slots.find((s) => s.name === "ask"), "the clip the sentence asks for is brought in, too");
  }

  // -- the tone

  {
    // A known change of colour is found again, from pictures with plenty of structure.
    let seed = 3;
    const random = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
    const photo = Array.from({ length: 900 }, () => [0.1 + 0.8 * random(), 0.1 + 0.8 * random(), 0.1 + 0.8 * random()]);
    const matrix = [0.96, 0.03, 0, 0.01, 1.04, -0.02, 0, 0.02, 0.97], offset = [0.02, -0.015, 0.01];
    const clip = photo.map((p) => {
      // clip = inverse of (photo = matrix * clip + offset), computed by the test with a plain 3x3 inverse
      const [a, b, c, d, e, f, g, h, i] = matrix;
      const r = [p[0] - offset[0], p[1] - offset[1], p[2] - offset[2]];
      const det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
      return [
        ((e * i - f * h) * r[0] - (b * i - c * h) * r[1] + (b * f - c * e) * r[2]) / det,
        (-(d * i - f * g) * r[0] + (a * i - c * g) * r[1] - (a * f - c * d) * r[2]) / det,
        ((d * h - e * g) * r[0] - (a * h - b * g) * r[1] + (a * e - b * d) * r[2]) / det,
      ];
    });
    const tone = plain(clipTone(clip, photo));
    assert.ok(tone.fitted);
    let worst = 0;
    for (let i = 0; i < 900; i++) {
      for (let k = 0; k < 3; k++) {
        const got = tone.matrix[k * 3] * clip[i][0] + tone.matrix[k * 3 + 1] * clip[i][1] + tone.matrix[k * 3 + 2] * clip[i][2] + tone.offset[k];
        worst = Math.max(worst, Math.abs(got - photo[i][k]));
      }
    }
    assert.ok(worst < 0.01, `the fitted colours are within ${worst} of the photo's`);
    // A tenth of the picture that does not match at all (a head, a hand) does not drag the fit away.
    const spoiled = clip.map((c, i) => (i % 10 === 0 ? [random(), random(), random()] : c));
    const robust = plain(clipTone(spoiled, photo));
    assert.ok(robust.fitted);
    let robustWorst = 0;
    for (let i = 0; i < 900; i += 10 === 0 ? 1 : 1) {
      if (i % 10 === 0) continue;
      for (let k = 0; k < 3; k++) {
        const got = robust.matrix[k * 3] * clip[i][0] + robust.matrix[k * 3 + 1] * clip[i][1] + robust.matrix[k * 3 + 2] * clip[i][2] + robust.offset[k];
        robustWorst = Math.max(robustWorst, Math.abs(got - photo[i][k]));
      }
    }
    assert.ok(robustWorst < 0.03, `with outliers: ${robustWorst}`);
    // A clip that is the photo needs no change; a clip that is something else is left alone.
    const same = plain(clipTone(photo, photo));
    assert.ok(same.fitted && same.offset.every((x) => Math.abs(x) < 0.01) && Math.abs(same.matrix[0] - 1) < 0.01);
    const other = plain(clipTone(photo.map((p) => [1 - p[0], p[2] * 0.2, 0.9]), photo));
    assert.deepStrictEqual(other.matrix, [1, 0, 0, 0, 1, 0, 0, 0, 1]);
    assert.strictEqual(other.fitted, false);
    assert.strictEqual(plain(clipTone(photo.slice(0, 10), photo.slice(0, 10))).fitted, false, "too little to fit");
    const flat = plain(clipTone(photo.map(() => [0.5, 0.5, 0.5]), photo.map(() => [0.52, 0.5, 0.5])));
    assert.ok(Math.abs(flat.offset[0] - 0.02) < 0.01 || !flat.fitted, "a flat picture is fitted with an offset or left alone, never blown up");
  }

  console.log("photostage clips tests passed");
})().catch((error) => {
  console.error(error);
  process.exit(1);
});
