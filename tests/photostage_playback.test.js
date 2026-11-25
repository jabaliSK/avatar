"use strict";

const assert = require("assert");
const fs = require("fs");
const path = require("path");
const vm = require("vm");

let clock = 0;
let hold = null;
let failType = null;
const images = [];

const browser = vm.createContext({
  atob: (text) => Buffer.from(text, "base64").toString("binary"),
  Blob: class { constructor(parts, options) { this.parts = parts; this.type = options.type; } },
  createImageBitmap(blob, options) {
    assert.strictEqual(options.colorSpaceConversion, "none");
    assert.strictEqual(options.premultiplyAlpha, "none");
    if (blob.type === failType) return Promise.reject(new Error("undecodable"));
    const image = { type: blob.type, closed: false, close() { this.closed = true; } };
    images.push(image);
    return hold ? hold.then(() => image) : Promise.resolve(image);
  },
  performance: { now: () => clock },
  setTimeout,
  window: {},
});
vm.runInContext(fs.readFileSync(path.join(__dirname, "../photostage/web/playback.js"), "utf8"), browser);
const playerSource = fs.readFileSync(path.join(__dirname, "../photostage/web/player.js"), "utf8");
vm.runInContext(playerSource.slice(playerSource.indexOf("async function loadQuietBank("),
  playerSource.indexOf("// Ordinary clip failures")), browser);
browser.DATA_IMAGE = { colorSpaceConversion: "none", premultiplyAlpha: "none" };
browser.clipNumbers = (row, count) => Array.isArray(row) && row.length === count && row.every(Number.isFinite);

function context() {
  return {
    currentTime: 1,
    destination: {},
    createBuffer: (channels, length, rate) => ({ duration: length / rate, getChannelData: () => new Float32Array(length) }),
    createBufferSource: () => ({ connect() {}, disconnect() {}, start() {}, stop() {} }),
  };
}

function player({ real = false } = {}) {
  const playback = vm.runInContext("new LivePlayback()", browser);
  const starts = [];
  playback.context = context();
  playback.stats = { firstPlaybackSeconds: null, underruns: 0, maxQueuedFrames: 0, segments: 0 };
  if (!real) playback.schedulePacket = (packet, start) => starts.push(start);
  return { playback, starts };
}

function segment({ frames = 24, guided = true, seconds = 0.96, total = 0 } = {}) {
  const encode = (prefix) => Array.from({ length: frames }, (_, i) => Buffer.from(`${prefix}${i}`).toString("base64"));
  return {
    index: 0, fps: 25, rate: 16000,
    audio: Buffer.from(new Int16Array(Math.round(seconds * 16000)).buffer).toString("base64"),
    frames: encode("jpeg"),
    ...(guided ? { guides: encode("png") } : {}),
    ...(total ? { total } : {}),
  };
}

async function arrive(playback, at, options) {
  clock = at * 1000;
  assert.strictEqual(await playback.enqueue(segment(options)), true);
}

async function main() {
  {
    const bank = vm.runInContext("loadQuietBank", browser);
    let info = { fps: 16, frames: [[0, 4], [0, 4], [4, 4]] };
    browser.getJson = async () => info;
    browser.download = async () => ({ arrayBuffer: async () => new ArrayBuffer(12) });
    const beforeUnique = images.length;
    const repeated = await bank("/", { index: "rest.json", frames: "rest.bin" }, 16);
    assert.strictEqual(repeated.length, 3);
    assert.strictEqual(repeated[0], repeated[1], "repeated idle frames share one decoded image");
    assert.strictEqual(images.length - beforeUnique, 2, "duplicate frame offsets are decoded only once");
    const moving = { index: "rest.json", frames: "rest.bin", anchorEvery: 2 };
    await assert.rejects(bank("/", moving, 16), /same anchor/);
    info.frames = [[0, 4], [4, 4], [8, 4], [4, 4]];
    await assert.rejects(bank("/", moving, 16), /same anchor/);
    info.frames[2] = [0, 4];
    const exitFrames = await bank("/", moving, 16);
    assert.strictEqual(exitFrames[2], exitFrames[0]);
    const subtle = { ...moving, randomBlink: { every: 2, offset: 1, length: 1 } };
    await assert.rejects(bank("/", subtle, 16), /variants/);
    info.frames[3] = [8, 4];
    assert.strictEqual((await bank("/", subtle, 16)).length, 4);
    info.frames.pop();
    await assert.rejects(bank("/", { index: "quiet.json", frames: "quiet.bin" }, 16, 4), /does not match/);
    info = { fps: 16, frames: [[0, 4], [8, 9]] };
    await assert.rejects(bank("/", { index: "rest.json", frames: "rest.bin" }, 16), /does not match/);
    info = { fps: 16, frames: [[0, 4], [4, 4]] };
    const before = images.length;
    const decode = browser.createImageBitmap;
    let first = true;
    browser.createImageBitmap = (blob, options) => {
      if (first) { first = false; return decode(blob, options); }
      return Promise.reject(new Error("broken idle JPEG"));
    };
    await assert.rejects(bank("/", { index: "rest.json", frames: "rest.bin" }, 16), /broken idle JPEG/);
    assert.ok(images.slice(before).every((image) => image.closed), "failed rest-bank decoding releases successful images");
    browser.createImageBitmap = decode;
  }

  {
    const { playback, starts } = player();
    let calls = 0;
    playback.alignStart = (earliest) => { calls++; return 3 + Math.max(0, Math.ceil((earliest - 3) / 2)) * 2; };
    await arrive(playback, 0, { frames: 1 });
    playback.context.currentTime = 3.2;
    await arrive(playback, 0.1, { frames: 1 });
    assert.strictEqual(starts[0], 5, "a missed preparation deadline waits for the same phase on the next loop");
    assert.strictEqual(starts[1], 5.96);
    playback.context.currentTime = playback.next - 0.1;
    await arrive(playback, 0.2, { frames: 1 });
    assert.strictEqual(calls, 1, "later chunks must not realign the audio or insert gaps");
    playback.reset();
    assert.strictEqual(playback.alignStart, null);
    playback.previousRate = 0.65;
    assert.ok(playback.preparationDelay(11) > playback.preparationDelay(2), "long replies must forecast the adaptive buffer");
    const slow = playback.preparationDelay(11);
    playback.stats.renderRate = 1.2;
    playback.reset();
    assert.ok(playback.preparationDelay(11) < slow, "the next turn should use the observed delivery rate");
  }

  {
    const { playback } = player({ real: true });
    clock = 10000;
    playback.context.state = "running";
    playback.context.getOutputTimestamp = () => ({ contextTime: 1.03, performanceTime: 10000 });
    playback.done = true;
    await playback.enqueue(segment({ frames: 3, seconds: 0.12, guided: false }));
    const [first, , last] = playback.frames.slice();
    playback.context.currentTime = 1.2;
    assert.strictEqual(playback.poll().image, null, "a frame must wait for its sound to reach the output device");
    assert.strictEqual(playback.position, 0);
    clock = 10050;
    assert.strictEqual(playback.poll().image, first.image);
    assert.ok(Math.abs(playback.position - 0.03) < 1e-9);
    clock = 10101;
    const shown = playback.poll();
    assert.strictEqual(shown.image, last.image);
    assert.strictEqual(shown.finished, false, "the render clock must not finish playback before the device does");
    clock = 10160;
    assert.strictEqual(playback.poll().finished, true);
    playback.context.state = "suspended";
    const stopped = playback.audibleTime;
    clock += 1000;
    assert.strictEqual(playback.audibleTime, stopped);
    playback.context.state = "running";
    playback.context.getOutputTimestamp = () => ({ contextTime: 0, performanceTime: 0 });
    assert.strictEqual(playback.audibleTime, playback.context.currentTime, "an uninitialized device timestamp uses the audio clock");
  }

  {
    const { playback } = player({ real: true });
    await arrive(playback, 0, { frames: 1, seconds: 1.4 });
    assert.strictEqual(playback.position, 0, "scheduled audio has not yet been heard");
    playback.context.currentTime = 1.55;
    assert.ok(Math.abs(playback.position - 0.5) < 1e-9);
    playback.context.currentTime = 10;
    assert.strictEqual(playback.position, 1.4, "a gap advances no speech");
    await arrive(playback, 1, { frames: 1, seconds: 2 });
    assert.strictEqual(playback.position, 1.4, "restart scheduling headroom is not speech");
    playback.context.currentTime = 10.55;
    assert.ok(Math.abs(playback.position - 1.9) < 1e-9);
    assert.ok(Math.abs(playback.position - 1.9) < 1e-9, "a suspended audio clock holds the same position");
    playback.reset();
    assert.strictEqual(playback.position, 0, "a new turn forgets the previous speech timeline");
  }

  {
    const { playback, starts } = player();
    await arrive(playback, 0, { frames: 1 });
    assert.deepStrictEqual(starts, [], "less than 1.3 s of speech must keep buffering");
    await arrive(playback, 0.1, { frames: 1 });
    assert.strictEqual(starts.length, 2);
    const end = playback.next;
    playback.context.currentTime = end - 0.05;
    await arrive(playback, 0.2, { frames: 1 });
    assert.strictEqual(starts[2], end, "an on-time chunk must not insert silence");
  }

  {
    const { playback, starts } = player();
    await arrive(playback, 0, { frames: 1, seconds: 1.4 });
    assert.strictEqual(starts.length, 1);
    playback.context.currentTime = playback.next + 0.3;
    await arrive(playback, 1, { frames: 1 });
    assert.strictEqual(playback.stats.underruns, 1);
    assert.strictEqual(starts.length, 1, "an underrun must rebuffer before resuming");
    await arrive(playback, 2, { frames: 1 });
    assert.strictEqual(starts.length, 3);
    assert.strictEqual(starts[2], starts[1] + 0.96);
  }

  {
    const { playback, starts } = player();
    await arrive(playback, 0, { frames: 1, seconds: 0.4 });
    assert.strictEqual(starts.length, 0);
    playback.done = true;
    playback.schedule();
    assert.strictEqual(starts.length, 1, "completion must release a short reply");
  }

  {
    // The renderer delivers half a second of speech per second: starting after the
    // minimum buffer would stall mid-sentence, so speech is held back first.
    const { playback, starts } = player();
    await arrive(playback, 2, { frames: 1, total: 4 });
    await arrive(playback, 3.92, { frames: 1, total: 4 });
    assert.strictEqual(starts.length, 0, "a slow renderer needs a deeper buffer than 1.3 s");
    await arrive(playback, 5.84, { frames: 1, total: 4 });
    assert.strictEqual(starts.length, 3);
    assert.ok(Math.abs(playback.stats.renderRate - 0.5) < 0.01, "the delivery rate is reported");
  }

  {
    const { playback, starts } = player();
    await arrive(playback, 2, { frames: 1, total: 20 });
    for (const at of [4, 6, 8]) await arrive(playback, at, { frames: 1, total: 20 });
    assert.strictEqual(starts.length, 0);
    await arrive(playback, 10, { frames: 1, total: 20 });
    assert.strictEqual(starts.length, 5, "the hold-back is capped, so a long reply still starts");
  }

  for (const [label, times, total] of [
    ["a fast renderer", [2, 2.5], 4],
    ["an unknown reply length", [2, 4], 0],
  ]) {
    const { playback, starts } = player();
    await arrive(playback, times[0], { frames: 1, total });
    await arrive(playback, times[1], { frames: 1, total });
    assert.strictEqual(starts.length, 2, `${label} must start after the minimum buffer`);
  }

  {
    const { playback } = player({ real: true });
    playback.done = true;
    await arrive(playback, 0, { frames: 3 });
    assert.strictEqual(playback.frames.length, 3);
    playback.frames.forEach((frame, index) => {
      assert.strictEqual(frame.image.type, "image/jpeg");
      assert.strictEqual(frame.guide.type, "image/png");
      assert.ok(Math.abs(frame.at - (playback.frames[0].at + index / 25)) < 1e-9);
    });
    const [first, second, third] = playback.frames.slice();
    playback.context.currentTime = third.at + 0.001;
    const shown = playback.poll();
    assert.strictEqual(shown.image, third.image);
    assert.strictEqual(shown.guide, third.guide);
    assert.deepStrictEqual([first.image, first.guide, second.image, second.guide].map((x) => x.closed), [true, true, true, true]);
    assert.deepStrictEqual([third.image.closed, third.guide.closed], [false, false], "the caller owns the returned frame");
  }

  {
    const { playback } = player({ real: true });
    playback.done = true;
    await arrive(playback, 0, { frames: 2, guided: false });
    assert.deepStrictEqual(Array.from(playback.frames, (frame) => frame.guide), [null, null]);
  }

  {
    const before = images.length;
    const { playback } = player({ real: true });
    failType = "image/png";
    await assert.rejects(playback.enqueue(segment({ frames: 2 })), /undecodable/);
    failType = null;
    assert.ok(images.slice(before).every((image) => image.closed), "a failed decode must not leak decoded frames");
    assert.strictEqual(playback.pending.length, 0);
  }

  {
    const before = images.length;
    const { playback } = player({ real: true });
    let release;
    hold = new Promise((resolve) => { release = resolve; });
    const result = playback.enqueue(segment({ frames: 2 }));
    playback.reset();
    release();
    assert.strictEqual(await result, false, "a segment that finishes decoding after a reset is dropped");
    hold = null;
    assert.ok(images.slice(before).every((image) => image.closed));
    assert.strictEqual(playback.pending.length, 0);
  }
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
