"use strict";

const assert = require("assert");
const fs = require("fs");
const path = require("path");
const vm = require("vm");

const browser = vm.createContext({});
vm.runInContext(fs.readFileSync(path.join(__dirname, "../photostage/web/body.js"), "utf8"), browser);
const { BodyMotion, SpeechBeats, Wander, bump, soften, follow, answer, liftOf, outOf, KNEE, STROKE_APEX, APEX_LEAD, LATE_START, BURST_SECONDS, OPENING_SECONDS } =
  vm.runInContext(
    "({ BodyMotion, SpeechBeats, Wander, bump, soften, follow, answer, liftOf, outOf, KNEE, STROKE_APEX, APEX_LEAD, LATE_START, BURST_SECONDS, OPENING_SECONDS })",
    browser,
  );

const RATE = 16000;
const FPS = 30;
const TAU = 2 * Math.PI;

// The body.json of an avatar with two hands, one on each side of the picture (limits as the rig builder works them out).
const SPEC = {
  version: 1,
  size: [1536, 2752],
  sway: 17.744,
  lift: 8.872,
  hands: [
    { box: [478.73, 1509.3, 870.03, 1844.7], axis: [0.98066, -0.19574], turn: 0.33896, slide: 19.44 },
    { box: [779.73, 1990.9, 1119.43, 2287.6], axis: [-0.63762, 0.77035], turn: 0.39592, slide: 16.05 },
  ],
};

const random = (seed) => {
  let state = seed >>> 0;
  return () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 4294967296;
  };
};

// Something like speech: a voiced tone that swells with every syllable, changes level every 0.22 s and falls nearly silent
// between sentences.
function speech(seconds, seed) {
  const next = random(seed);
  const out = new Float32Array(Math.round(seconds * RATE));
  let level = 0.3;
  for (let i = 0; i < out.length; i++) {
    const t = i / RATE;
    if (i % Math.round(RATE * 0.22) === 0) level = 0.25 + 0.6 * next();
    const syllable = 0.5 + 0.5 * Math.sin(TAU * 4.5 * t);
    const sentence = t % 5.5 < 4.6 ? 1 : 0.02;
    out[i] = Math.sin(TAU * 140 * t) * syllable * level * sentence * 0.4;
  }
  return out;
}

// Silence with short bursts of voice at the given [time, loudness]s.
function bursts(seconds, list, rate = RATE) {
  const out = new Float32Array(Math.round(seconds * rate));
  for (const [centre, loudness] of list) {
    const first = Math.max(0, Math.round((centre - 0.4) * rate));
    const last = Math.min(out.length, Math.round((centre + 0.4) * rate));
    for (let i = first; i < last; i++) {
      const t = i / rate;
      out[i] += loudness * Math.exp(-0.5 * ((t - centre) / 0.06) ** 2) * Math.sin(TAU * 150 * t);
    }
  }
  return out;
}

const find = (audio, start = 0, rate = RATE) => new SpeechBeats().feed(audio, rate, start);

// A hand's pose as multiples of its limits: [turn, slide x, slide y].
const limits = (pose, index) => {
  const { angle, x, y } = pose.hands[index];
  const hand = SPEC.hands[index];
  return [angle / hand.turn, x / hand.slide, y / hand.slide];
};
const reach = (pose, index) => Math.max(...limits(pose, index).map(Math.abs));
const mean = (values) => values.reduce((sum, value) => sum + value, 0) / values.length;
const sum = (values) => values.reduce((total, value) => total + value, 0);

// Plays `total` seconds at 30 frames a second: the audio of `speak` = [from, to] is scheduled at the start, the avatar speaks
// in that window and listens the rest of the time. Returns the frames, the beats (stresses that made a beat) and the strokes
// (flicks of a hand) that were made, and the driver.
function play(intensity, seed, { speak = [4, 24], total = 40, spec = SPEC, audio = null } = {}) {
  const motion = new BodyMotion(spec, { intensity, seed });
  const beats = [];
  const strokes = [];
  const beat = motion.beat.bind(motion);
  motion.beat = (strength, at, clock) => {
    const before = motion.lastBeat;
    beat(strength, at, clock);
    if (motion.lastBeat !== before) beats.push(motion.time);
  };
  const stroke = motion.stroke.bind(motion);
  motion.stroke = (index, size, at, clock, late) => {
    const made = stroke(index, size, at, clock, late);
    if (made) strokes.push({ time: motion.time, index, size, late });
    return made;
  };
  if (speak) motion.hear(audio || speech(speak[1] - speak[0], seed + 7), RATE, speak[0]);
  const frames = [];
  for (let n = 1; n <= total * FPS; n++) {
    const t = n / FPS;
    const pose = motion.tick(1 / FPS, t, Boolean(speak) && t >= speak[0] && t < speak[1]);
    frames.push({ t, pose, bursting: Boolean(motion.burst) });
  }
  return { frames, beats, strokes, motion };
}

const SEEDS = [1, 2, 3, 4, 5];
const during = (frames, from, to) => frames.filter((frame) => frame.t >= from && frame.t < to);
// How far the hands are from where the photo has them, the larger of the two, as a multiple of the limits.
const both = ({ pose }) => Math.max(reach(pose, 0), reach(pose, 1));

{
  // Nothing moves without a rig, with a rig this code does not know, or at intensity 0, and nothing is taken in either.
  const inactive = [
    new BodyMotion(null, { seed: 1 }),
    new BodyMotion(undefined),
    new BodyMotion(Object.assign({}, SPEC, { version: 2 }), { seed: 1 }),
    new BodyMotion(SPEC, { intensity: 0, seed: 1 }),
    new BodyMotion(SPEC, { intensity: -3, seed: 1 }),
  ];
  for (const motion of inactive) {
    motion.hear(speech(1, 1), RATE, 0);
    assert.strictEqual(motion.tick(1 / FPS, 0.5, true), null);
    assert.strictEqual(motion.events.length, 0);
  }
  assert.strictEqual(new BodyMotion(SPEC, { intensity: 7 }).intensity, 2, "intensity is held to 2");
  assert.strictEqual(new BodyMotion(SPEC, { intensity: NaN }).intensity, 1, "an unreadable intensity is the normal one");
  assert.ok(new BodyMotion(SPEC).active, "intensity 1 is the default");
}

{
  // An avatar whose hands could not be moved still breathes and sways, and nothing breaks without a list of hands.
  for (const spec of [Object.assign({}, SPEC, { hands: [] }), { version: 1, size: SPEC.size, sway: 10, lift: 5 }]) {
    const motion = new BodyMotion(spec, { seed: 2 });
    motion.hear(speech(6, 2), RATE, 0);
    let lowest = 0;
    for (let n = 1; n <= 8 * FPS; n++) {
      const pose = motion.tick(1 / FPS, n / FPS, n / FPS < 6);
      assert.strictEqual(pose.hands.length, 0);
      assert.ok(pose.sway.every(Number.isFinite));
      lowest = Math.min(lowest, pose.sway[1]);
    }
    assert.ok(lowest < -0.3 * spec.lift, "the breath lifts the body");
  }
  // A hand that has no turn or no slide (the rig builder left it out) simply does not move that way.
  const bare = { version: 1, size: SPEC.size, sway: 10, lift: 5, hands: [{ box: SPEC.hands[0].box, axis: SPEC.hands[0].axis }] };
  const motion = new BodyMotion(bare, { seed: 3 });
  motion.hear(speech(6, 3), RATE, 0);
  for (let n = 1; n <= 8 * FPS; n++) {
    const pose = motion.tick(1 / FPS, n / FPS, n / FPS < 6);
    assert.deepStrictEqual(JSON.parse(JSON.stringify(pose.hands)), [{ angle: 0, x: 0, y: 0 }]);
  }
}

{
  // Breathing never stops, scales with the intensity and stays within what the rig allows, and the body shifts its weight
  // slowly and less while it is listening than when it speaks.
  const ranges = {};
  for (const intensity of [0.5, 1, 2]) {
    for (const seed of SEEDS) {
      const motion = new BodyMotion(SPEC, { intensity, seed });
      const lows = [];
      let lowest = 0;
      let previous = [];
      for (let n = 1; n <= 40 * FPS; n++) {
        const [x, y] = motion.tick(1 / FPS, n / FPS, false).sway;
        assert.ok(y <= 0 && y >= -SPEC.lift * intensity, "the body is lifted by the breath, and no more than the rig allows");
        assert.ok(Math.abs(x) <= 0.25 * SPEC.sway * intensity, "it sways little while listening: " + x);
        lowest = Math.min(lowest, y);
        previous.push(y);
        if (previous.length === 5) {
          const [a, b, c, d, e] = previous;
          if (c < b && c <= d && c < a && c < e && c < -0.25 * SPEC.lift * intensity) lows.push(n);
          previous = previous.slice(1);
        }
      }
      assert.ok(lows.length >= 8 && lows.length <= 11, "about one breath every four seconds: " + lows.length + " in 40 s");
      if (seed === 3) ranges[intensity] = lowest;
    }
  }
  assert.ok(Math.abs(ranges[2] / ranges[0.5] - 4) < 0.05, "the breath moves four times as much at intensity 2 as at 0.5");

  for (const seed of SEEDS) {
    let farthest = 0;
    let fastest = 0;
    let last = null;
    const motion = new BodyMotion(SPEC, { intensity: 1, seed });
    for (let n = 1; n <= 60 * FPS; n++) {
      const [x] = motion.tick(1 / FPS, n / FPS, true).sway;
      assert.ok(Math.abs(x) <= 0.6 * SPEC.sway, "a speaking avatar sways at most this far: " + x);
      if (n > 5 * FPS) farthest = Math.max(farthest, Math.abs(x));
      if (last !== null) fastest = Math.max(fastest, Math.abs(x - last) * FPS);
      last = x;
    }
    assert.ok(farthest > 0.2 * SPEC.sway, "a speaking avatar sways further than a listening one can: " + farthest);
    assert.ok(fastest < 0.5 * SPEC.sway, "and slowly: " + fastest);
  }
}

{
  // A pose is bent to stay inside the limits: its own up to the knee, always short of the limit, and without a kink.
  assert.ok(KNEE > 0 && KNEE < 1);
  for (const value of [0, 0.1, -0.3, KNEE, -KNEE]) assert.strictEqual(soften(value), value);
  let before = soften(KNEE);
  for (let value = KNEE + 0.01; value < 4; value += 0.01) {
    const now = soften(value);
    assert.ok(now > before && now < 1, "rises with the request and never reaches the limit: " + value);
    assert.strictEqual(soften(-value), -now, "and the same either way");
    assert.ok(now - before < 0.0101, "with no step at the knee or after it");
    before = now;
  }
  assert.ok(soften(1.5) > 0.9, "a hand that is asked for more than the limit is brought close to it");

  // A push has no speed where it starts or ends and peaks at its apex, early in its length.
  assert.strictEqual(bump(0), 0);
  assert.strictEqual(bump(1), 0);
  assert.strictEqual(bump(-0.5), 0);
  assert.strictEqual(bump(2), 0);
  assert.strictEqual(bump(NaN), 0);
  assert.ok(Math.abs(bump(STROKE_APEX) - 1) < 1e-12);
  assert.ok(STROKE_APEX < 0.5, "out quickly and back more slowly");
  assert.ok(bump(0.01) < 0.01 && bump(0.99) < 0.01, "a push is not kicked into motion or stopped short");
  for (let u = 0.01; u < STROKE_APEX; u += 0.01) assert.ok(bump(u + 0.01) > bump(u), "out");
  for (let u = STROKE_APEX + 0.01; u < 0.99; u += 0.01) assert.ok(bump(u + 0.01) < bump(u), "back");
}

{
  // A spring goes to its goal the same way wherever the time is cut, and never changes its speed at once when the goal changes.
  const make = () => ({ x: 0, v: 0 });
  const once = make();
  follow(once, 1, 4, 1, 0.6);
  const often = make();
  for (let i = 0; i < 60; i++) follow(often, 1, 4, 1, 0.01);
  assert.ok(Math.abs(once.x - often.x) < 0.01 && Math.abs(once.v - often.v) < 0.02, "in one step or in many");
  const still = make();
  follow(still, 5, 4, 1, 0);
  assert.deepStrictEqual(still, make(), "no time passed, no change");

  const critical = make();
  let last = 0;
  for (let i = 0; i < 600; i++) {
    follow(critical, 1, 4, 1, 1 / FPS);
    assert.ok(critical.x >= last - 1e-9 && critical.x <= 1, "a critically damped spring does not overshoot");
    last = critical.x;
  }
  assert.ok(Math.abs(critical.x - 1) < 1e-4 && Math.abs(critical.v) < 1e-3);

  const lively = make();
  let top = 0;
  for (let i = 0; i < 300; i++) {
    follow(lively, 1, 11, 0.6, 1 / FPS);
    top = Math.max(top, lively.x);
  }
  assert.ok(top > 1.05 && top < 1.15, "one that is damped less overshoots (by about a tenth): " + top);
  assert.ok(Math.abs(lively.x - 1) < 1e-3, "and settles");

  const turning = make();
  follow(turning, 1, 4, 1, 0.5);
  const speed = turning.v;
  follow(turning, -1, 4, 1, 1e-6);
  assert.ok(Math.abs(turning.v - speed) < 1e-3, "a new goal does not change the speed");

  // `answer` tells when a spring that is pushed with a bump reaches its apex and how high: what a cut of a different length
  // finds, to within a step.
  for (const [rate, damping, length] of [[11, 0.6, 0.26], [9.5, 0.54, 0.42], [8.25, 0.66, 0.34], [11, 1, 0.3]]) {
    const { at, top: height } = answer(rate, damping, length);
    const spring = make();
    let peak = 0;
    let when = 0;
    for (let n = 1; n <= 1500; n++) {
      follow(spring, bump((n * 0.001 - 0.0005) / length), rate, damping, 0.001);
      if (spring.x > peak) {
        peak = spring.x;
        when = n * 0.001;
      }
    }
    assert.ok(Math.abs(when - at) < 0.008, "the apex of a flick: " + when + " against " + at);
    assert.ok(Math.abs(peak - height) < 0.01 * height, "and its height: " + peak + " against " + height);
    assert.ok(at > STROKE_APEX * length && height > 0.2 && height < 1.2);
  }
}

{
  // Wandering is smooth, stays within [-1, 1], never stops and is different with each seed.
  const track = (seed) => {
    const wander = new Wander(random(seed), [5.3, 2.7, 1.4]);
    const out = [];
    for (let n = 0; n < 60 * 120; n++) out.push(wander.at(n / 60));
    return out;
  };
  const one = track(1);
  assert.ok(one.every((value) => Math.abs(value) <= 1));
  let fastest = 0;
  for (let i = 1; i < one.length; i++) fastest = Math.max(fastest, Math.abs(one[i] - one[i - 1]) * 60);
  assert.ok(fastest < 1.5, "it changes slowly: " + fastest + " a second");
  assert.ok(Math.max(...one) > 0.3 && Math.min(...one) < -0.3, "but it does go somewhere");
  assert.deepStrictEqual(track(1), one);
  assert.notDeepStrictEqual(track(2), one);

  // Fingers go up with a turn of -1 for a hand that points to the right, and +1 for one that points to the left; out is
  // away from the middle of the picture.
  assert.strictEqual(liftOf(SPEC.hands[0]), -1);
  assert.strictEqual(liftOf(SPEC.hands[1]), 1);
  assert.strictEqual(liftOf({}), 1);
  assert.strictEqual(outOf(SPEC.hands[0], SPEC.size), -1);
  assert.strictEqual(outOf(SPEC.hands[1], SPEC.size), 1);
  assert.strictEqual(outOf(SPEC.hands[0], null), 1);
}

{
  // Stresses are found in speech: where it swells, not in silence or in hiss, once for a swell, at its peak, and the louder
  // the stronger.
  assert.strictEqual(find(new Float32Array(3 * RATE)).length, 0);
  const next = random(5);
  assert.strictEqual(find(new Float32Array(3 * RATE).map(() => (next() - 0.5) * 0.01)).length, 0, "hiss is not speech");

  const three = find(bursts(3.5, [[0.5, 0.6], [1.5, 0.6], [2.6, 0.6]]));
  assert.strictEqual(three.length, 3);
  three.forEach((beat, i) => {
    assert.ok(Math.abs(beat.at - [0.5, 1.5, 2.6][i]) < 0.05, "at the peak: " + beat.at);
    assert.ok(beat.strength > 0.9 && beat.strength <= 1);
  });

  const close = find(bursts(2.5, [[1, 0.6], [1.3, 0.6]]));
  assert.strictEqual(close.length, 1, "two stresses 0.3 s apart make one beat");
  assert.ok(Math.abs(close[0].at - 1) < 0.05);
  assert.strictEqual(find(bursts(2.5, [[1, 0.6], [1.7, 0.6]])).length, 2, "and two that are 0.7 s apart make two");

  const levels = find(bursts(3.5, [[0.5, 0.6], [1.5, 0.3], [2.5, 0.1]]));
  assert.strictEqual(levels.length, 2, "a swell well below the speaker's level is no stress");
  assert.ok(levels[1].strength < 0.8 * levels[0].strength, "and a softer stress is a weaker one");

  const rates = [8000, 16000, 22050, 24000, 48000];
  for (const rate of rates) {
    const beats = find(bursts(3.5, [[0.5, 0.6], [1.5, 0.6], [2.6, 0.6]], rate), 0, rate);
    assert.strictEqual(beats.length, 3, "at " + rate + " Hz");
    beats.forEach((beat, i) => assert.ok(Math.abs(beat.at - [0.5, 1.5, 2.6][i]) < 0.05));
  }

  // Times are the clock times of the audio, and the way it is cut up (in whole segments) does not matter.
  const next2 = random(9);
  const list = [];
  for (let at = 0.4; at < 9.5; at += 0.35 + 0.5 * next2()) list.push([at, 0.15 + 0.5 * next2()]);
  const audio = bursts(10, list);
  const whole = find(audio);
  assert.ok(whole.length >= 6);
  const pieces = new SpeechBeats();
  const parts = [];
  for (let at = 0; at < audio.length; at += 15360) parts.push(...pieces.feed(audio.subarray(at, at + 15360), RATE, at / RATE));
  assert.strictEqual(parts.length, whole.length);
  parts.forEach((beat, i) => {
    assert.ok(Math.abs(beat.at - whole[i].at) < 1e-9);
    assert.strictEqual(beat.strength, whole[i].strength);
  });
  const later = find(audio, 100);
  assert.strictEqual(later.length, whole.length);
  later.forEach((beat, i) => {
    assert.ok(Math.abs(beat.at - 100 - whole[i].at) < 1e-9);
    assert.strictEqual(beat.strength, whole[i].strength);
  });

  // A new turn is judged on its own: after a reset a quiet voice counts again, though the one before was loud.
  const loud = bursts(1, [[0.5, 0.6]]);
  const quiet = bursts(1, [[0.5, 0.1]]);
  const carried = new SpeechBeats();
  carried.feed(loud, RATE, 0);
  assert.strictEqual(carried.feed(quiet, RATE, 2).length, 0);
  const fresh = new SpeechBeats();
  fresh.feed(loud, RATE, 0);
  fresh.reset();
  const after = fresh.feed(quiet, RATE, 2);
  assert.strictEqual(after.length, 1);
  assert.ok(Math.abs(after[0].at - 2.5) < 0.05);
}

{
  // Pauses are found too: 0.3 s without voice after some, reported once, at the clock time of the audio, whatever the way it
  // is cut up, and not between two syllables.
  const found = (audio, start = 0, rate = RATE) => {
    const pauses = [];
    new SpeechBeats().feed(audio, rate, start, pauses);
    return pauses.map(({ at }) => at);
  };
  assert.deepStrictEqual(found(new Float32Array(3 * RATE)), [], "silence that follows no voice is no pause");
  const two = found(bursts(4, [[0.5, 0.6], [2, 0.6]]));
  assert.strictEqual(two.length, 2);
  assert.ok(two[0] > 0.8 && two[0] < 1.3, "just after the voice ends: " + two[0]);
  assert.ok(two[1] > 2.3 && two[1] < 2.8, "also at the end: " + two[1]);
  assert.strictEqual(found(bursts(3, [[0.5, 0.6], [1, 0.6]])).length, 1, "a gap of a syllable is no pause, the end is");
  assert.strictEqual(found(bursts(3, [[0.5, 0.6], [1.5, 0.6]]), 100).length, 2);
  assert.ok(Math.abs(found(bursts(4, [[0.5, 0.6], [2, 0.6]]), 50)[0] - 50 - two[0]) < 1e-9);

  const audio = bursts(6, [[0.5, 0.6], [1.6, 0.4], [2.2, 0.5], [4, 0.6]]);
  const whole = found(audio);
  const pieces = new SpeechBeats();
  const parts = [];
  for (let at = 0; at < audio.length; at += 15360) pieces.feed(audio.subarray(at, at + 15360), RATE, at / RATE, parts);
  assert.strictEqual(parts.length, whole.length);
  parts.forEach(({ at }, i) => assert.ok(Math.abs(at - whole[i]) < 1e-9));
}

{
  // A hand never reaches its limit, at any intensity, and the body never goes further than the rig's sway and lift.
  for (const intensity of [0.5, 1, 2]) {
    for (const seed of SEEDS) {
      for (const { pose } of play(intensity, seed).frames) {
        const where = "intensity " + intensity + ", seed " + seed;
        const [x, y] = pose.sway;
        assert.ok(Math.abs(x) <= SPEC.sway * intensity + 1e-9, "sway, " + where);
        assert.ok(y <= 0 && y >= -SPEC.lift * intensity - 1e-9, "lift, " + where + ": " + y);
        for (let index = 0; index < 2; index++) {
          assert.ok(limits(pose, index).every((value) => Math.abs(value) < 1), "hand " + index + ", " + where);
        }
      }
    }
  }
}

{
  // Gestures come in bursts with the hands at rest in between: a burst is not longer than it should be, the rest that follows
  // it is not shorter, and the hands are let down when the speech is over.
  for (const intensity of [0.5, 1, 2]) {
    for (const seed of SEEDS) {
      const { frames, motion } = play(intensity, seed, { total: 26.5 });
      const where = "intensity " + intensity + ", seed " + seed;
      const runs = [];
      let from = null;
      for (const { t, bursting } of frames) {
        if (bursting && from === null) from = t;
        if (!bursting && from !== null) {
          runs.push([from, t]);
          from = null;
        }
      }
      assert.ok(runs.length >= 2 && runs.length <= 6, "bursts in 20 s of speech, " + where + ": " + runs.length);
      assert.ok(runs[0][0] >= 4.3 && runs[0][0] < 9, "the first comes a little after the voice does, " + where + ": " + runs[0][0]);
      for (const [start, end] of runs) assert.ok(end - start <= BURST_SECONDS[1] + 0.1, "a burst is a phrase or two, " + where + ": " + (end - start));
      runs.slice(1).forEach(([start], i) => assert.ok(start - runs[i][1] >= 1.2, "with a rest between, " + where + ": " + (start - runs[i][1])));
      const inBurst = during(frames, 5, 24).filter((frame) => frame.bursting).length / during(frames, 5, 24).length;
      assert.ok(inBurst > 0.2 && inBurst < 0.85, "and not gesturing all the time or hardly ever, " + where + ": " + inBurst);
      assert.ok(frames.filter(({ t }) => t > 25.5).every(({ bursting }) => !bursting), "let down after the speech, " + where);
      for (const hand of motion.hands) {
        assert.ok(hand.place.every((value) => Math.abs(value) <= 0.1 * motion.reach + 1e-9), "to a place of its own, " + where + ": " + hand.place);
      }
    }
  }
}

{
  // A burst that is more than a moment old may end at a pause, and one that has just begun does not.
  const ended = (age, seed) => {
    const motion = new BodyMotion(SPEC, { intensity: 1, seed });
    motion.time = 5;
    motion.spokeAt = 5;
    motion.begin();
    motion.burst = { lead: 0, from: 5 - age, until: 100 };
    motion.gaps = [{ at: 5 }];
    motion.advance(5.1, true);
    return !motion.burst;
  };
  const seeds = Array.from({ length: 80 }, (_, i) => i + 1);
  const old = seeds.filter((seed) => ended(4, seed)).length;
  assert.ok(old > 20 && old < 60, "some end and some go on: " + old + " of 80");
  assert.strictEqual(seeds.filter((seed) => ended(0.5, seed)).length, 0, "a burst that has just begun goes on");
}

{
  // After a long silence the hands do not begin with the voice, any more than after a short one: a turn gets its first burst a
  // moment after it starts, and the first flick with it.
  for (const intensity of [1, 2]) {
    for (const seed of SEEDS) {
      const { frames, strokes } = play(intensity, seed, { speak: [40, 60], total: 66 });
      const where = "intensity " + intensity + ", seed " + seed;
      const first = frames.find(({ t, bursting }) => t > 40 && bursting);
      assert.ok(first && first.t >= 40 + OPENING_SECONDS[0] && first.t < 45, "the first burst of a turn comes a little after the voice does, " + where + ": " + (first && first.t));
      assert.ok(strokes.every(({ time }) => time < 40 || time >= 40 + OPENING_SECONDS[0]), "and so does the first flick, " + where);
    }
  }
}

{
  // The beats: how often there are, and how they are spaced. A hand flicks at most every 0.6 s, a louder stress is not a
  // lower one, and the hands make more of them, and reach further, the higher the intensity.
  const rate = {};
  const spread = {};
  for (const intensity of [0.5, 1, 2]) {
    let beats = 0;
    const averages = [];
    for (const seed of SEEDS) {
      const run = play(intensity, seed);
      beats += run.beats.length;
      run.beats.slice(1).forEach((time, i) => assert.ok(time - run.beats[i] >= 0.6 - 1e-9, "no two beats within 0.6 s: " + (time - run.beats[i])));
      assert.ok(run.beats.every((time) => time > 3 && time < 26), "beats are made when the avatar speaks");
      averages.push(mean(during(run.frames, 4, 24).map(both)));
    }
    rate[intensity] = beats / (SEEDS.length * 20 / 60);
    spread[intensity] = mean(averages);
  }
  assert.ok(rate[1] > 12 && rate[1] < 26, "about one beat in three seconds at the normal strength: " + rate[1] + " a minute");
  assert.ok(rate[0.5] > 4 && rate[0.5] < 14 && rate[0.5] < rate[1] && rate[1] < rate[2] && rate[2] < 40, "fewer at 0.5 and more at 2: " + JSON.stringify(rate));
  assert.ok(spread[0.5] + 0.04 < spread[1] && spread[1] + 0.04 < spread[2], "and the hands move further the higher it is: " + JSON.stringify(spread));
}

{
  // What the hands do over a turn is not that of a pendulum: they rest much of the time and are not at the limit for long.
  for (const intensity of [1, 2]) {
    let atRest = 0;
    let frames = 0;
    for (const seed of SEEDS) {
      const run = during(play(intensity, seed).frames, 4.6, 24);
      const reaches = run.map(both);
      let stay = 0;
      for (const value of reaches) {
        stay = value > 0.6 ? stay + 1 / FPS : 0;
        assert.ok(stay < 1.2, "a hand does not stay out at the limit: " + stay + " s at intensity " + intensity + ", seed " + seed);
      }
      assert.ok(reaches.filter((value) => value >= 0.95).length < 0.05 * reaches.length, "and is hardly ever at it");
      assert.ok(Math.max(...reaches) > 0.4, "it does reach for something: " + Math.max(...reaches));
      atRest += reaches.filter((value) => value < 0.15).length;
      frames += reaches.length;
    }
    assert.ok(atRest / frames > 0.1, "the hands rest at times when speaking: " + atRest / frames + " at intensity " + intensity);
  }
}

{
  // A strong beat lifts the shoulders a little and a weak one does not, whatever the body does, and the lift stays within what
  // the rig allows when the stresses come as fast as they can be taken, no two beats being closer than 0.6 s.
  const motion = new BodyMotion(SPEC, { intensity: 1, seed: 2 });
  motion.begin();
  motion.beat(0.4, 5, 4.5);
  assert.strictEqual(motion.pulses.length, 0);
  motion.beat(0.9, 5, 4.5);
  assert.strictEqual(motion.pulses.length, 1);

  const list = [];
  for (let at = 0.5; at < 19.6; at += 0.46) list.push([at, 0.6]);
  const rapid = bursts(20, list);
  for (const seed of SEEDS) {
    const { frames, beats } = play(2, seed, { speak: [0, 20], total: 22, audio: rapid });
    const lowest = Math.min(...frames.map(({ pose }) => pose.sway[1]));
    assert.ok(lowest >= -0.8 * 2 * SPEC.lift, "seed " + seed + ": " + lowest);
    assert.ok(beats.length >= 6 && beats.length <= 20 / 0.6, "beats, seed " + seed + ": " + beats.length);
    beats.slice(1).forEach((time, i) => assert.ok(time - beats[i] >= 0.6 - 1e-9));
  }
}

{
  // A listening avatar keeps its hands low: they drift and make a small adjustment now and then. After speaking they are put
  // down, to a place of their own.
  for (const intensity of [0.5, 1, 2]) {
    const quiet = [];
    for (const seed of SEEDS) {
      const silent = play(intensity, seed, { speak: null }).frames.map(both);
      assert.ok(Math.max(...silent) < 0.45, "an adjustment is small: " + Math.max(...silent));
      assert.ok(mean(silent) < 0.2);
      quiet.push(mean(silent));
    }
    const talking = mean(SEEDS.map((seed) => mean(during(play(intensity, seed).frames, 4.6, 24).map(both))));
    assert.ok(talking > 1.4 * mean(quiet), "speaking moves the hands more than listening does: " + talking + " against " + mean(quiet));
  }
}

{
  // The same seed is the same movement, and another seed is another.
  const render = (seed) => JSON.stringify(play(1, seed, { total: 20 }).frames.map(({ pose }) => pose));
  assert.strictEqual(render(4), render(4));
  assert.notStrictEqual(render(4), render(5));
}

{
  // Nothing is sudden: the speed and the change of speed of every part of every pose stay small, with the hands at their
  // liveliest, and there is no frame in which anything jumps.
  for (const seed of SEEDS) {
    const { frames } = play(2, seed);
    for (let index = 0; index < 2; index++) {
      const track = frames.map(({ pose }) => limits(pose, index));
      for (let i = 2; i < track.length; i++) {
        for (let part = 0; part < 3; part++) {
          const speed = Math.abs(track[i][part] - track[i - 1][part]) * FPS;
          const change = Math.abs(track[i][part] - 2 * track[i - 1][part] + track[i - 2][part]) * FPS * FPS;
          assert.ok(speed < 10, "speed " + speed + " at " + frames[i].t);
          assert.ok(change < 120, "acceleration " + change + " at " + frames[i].t);
        }
      }
    }
  }
}

{
  // A frame that comes late (a tab that was in the background) moves the body no further than one of a tenth of a second,
  // and a frame that does not advance the time does not move it.
  const warm = () => {
    const motion = new BodyMotion(SPEC, { intensity: 1, seed: 2 });
    motion.hear(speech(10, 9), RATE, 0);
    let last = null;
    for (let n = 1; n <= 3 * FPS; n++) last = motion.tick(1 / FPS, n / FPS, true);
    return { motion, last };
  };
  const numbers = (pose) => [].concat(pose.sway, ...pose.hands.map(({ angle, x, y }) => [angle, x, y]));
  assert.deepStrictEqual(numbers(warm().motion.tick(10, 3.1, true)), numbers(warm().motion.tick(0.1, 3.1, true)));
  const { motion, last } = warm();
  for (const step of [0, -1]) {
    const still = numbers(motion.tick(step, 3, true));
    numbers(last).forEach((value, i) => assert.ok(Math.abs(still[i] - value) < 1e-9, "step " + step));
  }
}

{
  // A flick lands on the stress it is made for: its apex, and the hand's farthest point, are where the audio gets to it, a
  // little before as the ear hears the sound a little after. Its height is what it is aimed at, whatever the hand's spring.
  const flicker = (seed, intensity = 1) => {
    const motion = new BodyMotion(SPEC, { intensity, seed });
    motion.begin();
    motion.motif = [1, 0.3, 0]; // fingers up and a little out, so that there is a turn to measure
    return motion;
  };
  for (const seed of [1, 2, 3, 4, 5, 6]) {
    for (const index of [0, 1]) {
      for (const late of [0, 0.09]) {
        const motion = flicker(seed);
        assert.ok(motion.stroke(index, 0.5, 2, 0, late));
        const hand = motion.hands[index];
        const { push, length } = hand.pulses[0];
        const gain = answer(hand.rate, hand.damping, length).top;
        let top = 0;
        let at = 0;
        const step = 1 / 240;
        for (let n = 1; n <= 4 * 240; n++) {
          motion.tick(step, 100 + n * step, false);
          const size = Math.abs(hand.flick[0].x);
          if (size > top) {
            top = size;
            at = n * step;
          }
        }
        const where = "seed " + seed + ", hand " + index + ", late " + late;
        assert.ok(Math.abs(at - (2 + late - APEX_LEAD)) < 0.012, "the apex is at the stress: " + at + ", " + where);
        assert.ok(Math.abs(top - Math.abs(push[0]) * gain) < 0.03 * top, "and as high as it is aimed: " + top + ", " + where);
        assert.ok(top > 0.3 && top < 0.7, "of a size that is a flick: " + top + ", " + where);
      }
    }
  }

  // A flick that cannot start in time starts late, up to a point, and one that is later than that is let go.
  const motion = flicker(3);
  assert.ok(!motion.stroke(0, 0.5, 0, 0, 0), "a stress that the audio is already at is too late");
  assert.strictEqual(motion.hands[0].pulses.length, 0);
  let at = 0;
  while (!motion.stroke(0, 0.5, at, 0, 0) && at < 1) at += 0.01;
  const { start, length } = motion.hands[0].pulses[0];
  const apex = answer(motion.hands[0].rate, motion.hands[0].damping, length).at;
  assert.ok(at > 0 && at < 0.6, "a stress that is a little late is not let go: " + at);
  assert.ok(at - APEX_LEAD - apex >= -LATE_START - 1e-9, "and not by more than it may be");
  assert.strictEqual(start, motion.time, "it starts at once");
  assert.ok(LATE_START > 0 && LATE_START < 0.25);
}

{
  // What was scheduled is dropped when a turn ends, and a stress that the clock has long passed is let go rather than
  // answered with a late flick.
  const considered = (setup, clockAt) => {
    const motion = new BodyMotion(SPEC, { intensity: 1, seed: 3 });
    motion.hear(bursts(2, [[1, 0.6]]), RATE, 1);
    assert.strictEqual(motion.events.length, 1);
    let calls = 0;
    motion.consider = () => calls++;
    setup(motion);
    for (let n = 1; n <= 3.5 * FPS; n++) motion.tick(1 / FPS, clockAt(n / FPS), false);
    return calls;
  };
  assert.strictEqual(considered(() => {}, (t) => t), 1, "a stress that is scheduled is thought of");
  assert.strictEqual(considered((motion) => motion.hush(), (t) => t), 0, "but not once the turn is over");
  assert.strictEqual(considered(() => {}, (t) => t + 4), 0, "or when the clock is already far past it");

  const motion = new BodyMotion(SPEC, { intensity: 1, seed: 3 });
  motion.hear(bursts(2, [[1, 0.6]]), RATE, 1);
  motion.hush();
  assert.strictEqual(motion.events.length, 0);
  assert.strictEqual(motion.gaps.length, 0);
  motion.hear(bursts(2, [[1, 0.1]]), RATE, 1);
  assert.strictEqual(motion.events.length, 1, "the next turn is judged on its own, however quiet it is");
}

{
  // Speech that has no stresses (a monotone, or audio that was never taken in) still gets some gesture, now and then and
  // gently, and a stress that is thought of starts a burst when there is none.
  for (const seed of SEEDS) {
    const { frames, strokes } = play(1, seed, { speak: [2, 20], total: 22, audio: new Float32Array(1) });
    assert.ok(strokes.length >= 2, "seed " + seed + ": " + strokes.length);
    assert.ok(Math.max(...during(frames, 2, 20).map(both)) < 0.7, "and gentle");
  }
  const motion = new BodyMotion(SPEC, { intensity: 1, seed: 7 });
  assert.strictEqual(motion.burst, null);
  let began = 0;
  for (let n = 1; n <= 20 && !motion.burst; n++) {
    motion.time = 10 + n;
    motion.restUntil = 0;
    motion.consider({ at: 11, strength: 0.9 }, 10.9, true);
    began++;
  }
  assert.ok(motion.burst && began < 10, "a strong stress starts a burst: " + began + " tries");
  const weak = new BodyMotion(SPEC, { intensity: 1, seed: 7 });
  for (let n = 1; n <= 40; n++) weak.consider({ at: 11, strength: 0.1 }, 10.9, true);
  assert.strictEqual(weak.burst, null, "a weak one does not");
}
