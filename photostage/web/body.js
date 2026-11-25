"use strict";

// How the body moves while the head talks. The driver decides how much and the compositor draws it: a pose is the body's
// displacement and each hand's turn and slide (photo pixels and radians), and it never leaves the limits in body.json,
// which the server checked by drawing every hand at them. Whatever moves is the photo's own pixels.
//
// It copies how people gesture as they talk, not a loop of gestures:
//   - A hand is held where the photo has it and drifts about that place, slowly. A listening avatar does no more, plus a
//     small adjustment of a hand every ten seconds or so.
//   - Gestures come in bursts of a phrase or two with the hands at rest in between. A burst has a leading hand, which is
//     raised a little to a place in front of the avatar and makes the same kind of beat again and again, each a little
//     different. The other hand joins in now and then, smaller and a little late.
//   - A beat is a quick flick that peaks on a stress of the speech. SpeechBeats finds the stresses in the audio the player
//     is about to play, a strong one is likelier to make a beat than a weak one, and the flick is what a spring does when it
//     is pushed briefly: it starts and ends smoothly, overshoots a little and leaves the hand a little off its place for a
//     moment, as the weight of a hand does. Where a hand is held changes with a slower spring, so nothing is ever sudden.
//   - The body breathes all the time, shifts its weight now and then, and lifts a little with a strong beat.
const TAU = 2 * Math.PI;
const BREATH_SECONDS = 4.2;
const ENVELOPE_RATE = 100; // the speech envelope is measured every 10 ms
const BEAT_GAP = 0.45; // the least time between two stresses that make beats
const BEAT_LATE = 0.4; // a stress this far behind the audio clock is let go
const PAUSE_FRAMES = 30; // 0.3 s of silence ends a phrase
const STROKE_APEX = 0.3; // a push goes up quickly and is let go more slowly: it peaks this far into its length
const APEX_LEAD = 0.03; // a flick peaks this long before its stress, as sound reaches the ear a little after the audio clock
const MAX_STEP = 0.004; // springs are taken in steps this short, whatever the frame rate
const LEAD_IN = 0.55; // a beat is started this long before its stress at the latest: more than a flick takes to peak
const LATE_START = 0.15; // a beat that can no longer start in time but is not later than this starts late; a later one is let go
// The quick spring that makes a flick, and the slow one that holds a hand:
const FLICK_RATE = 11; // natural frequency, radians per second
const FLICK_DAMPING = 0.6; // below 1, so that a flick overshoots a little
const SLIDE_RATE = 0.75; // a slide moves the whole hand, so it follows more slowly than the wrist turns
const HOLD_RATE = 3.2; // critically damped: a hand changes its place in about a second and a half
const KICK_SECONDS = 1.6; // a beat can leave the hand a little further on, and it comes back this slowly
const KICK_SIZE = [0.15, 0.4]; // of the beat, for the half of the beats that leave one
const KICK_LIMIT = 0.25; // of the limits
const WANDER_PERIODS = [5.3, 2.7, 1.4]; // seconds: a held hand drifts slowly, with a little quicker drift on top
const WANDER_QUIET = 0.05; // of the limits, for a listening avatar
const WANDER_TALK = 0.16; // and for a speaking one
// Beats and bursts of them:
const STROKE_SIZE = [0.2, 0.6]; // of the limits, for the weakest and the strongest stress
const STROKE_SECONDS = [0.26, 0.42]; // how long the push of a flick lasts: a small flick is quicker
const HAND_GAP = 0.6; // the least time between two beats
const BEAT_CHANCE = [0.15, 0.35]; // that a stress makes a beat in a burst: for the weakest stress, and the more for the strongest
const BOTH_HANDS = 0.15; // the chance that the hand that does not lead makes the beat too: smaller and a little late
const OTHER_SIZE = 0.6;
const OTHER_LATE = [0.04, 0.12];
const FALLBACK_SECONDS = 2.4; // speech this long in a burst without a stress gets a gentle beat
const BURST_SECONDS = [2.5, 6];
const REST_SECONDS = [1.5, 4.5];
const OPENING_SECONDS = [0.3, 1.2]; // a turn gets its first burst after this long, and not sooner: the hands follow the voice
const STARVED_SECONDS = 2.5; // speech without stresses gets a burst this long after it could have one
const BURST_STRESS = 0.3; // the weakest stress that starts a burst
const BURST_ENDS = 0.5; // the chance that a pause ends a burst that is not new
const BURST_AGE = 1.8; // seconds
const PLACE_SIZE = [0.25, 0.42]; // of the limits: how far a hand is raised for a burst
const OTHER_PLACE = 0.35; // the chance that the hand that does not lead is raised too
const REST_SIZE = 0.4; // how much a resting hand's place differs from the photo's own, at most a quarter of this
const ADJUST_EVERY = [7, 15]; // a listening avatar adjusts a hand this often
const ADJUST_SIZE = 0.2;
const ADJUST_SECONDS = [2.5, 5];
const LINGER_SECONDS = 1.2; // the hands are let down this long after the speech ends
const KNEE = 0.7; // a pose is its own up to this share of a limit and is bent to stay inside it beyond
const JITTER = 0.25; // how much a place or a beat differs from its model, of its size
const HEADROOM = 0.92; // a flick and the place the hand is held at together stay within this much of a limit
const LEAST_ROOM = 0.4; // a flick that has less room than this share of its size goes the other way, and one that has more is cut short
const REACH_GROWTH = 0.3; // above the normal intensity, hands reach this much further for each step
const LIVELY_GROWTH = 0.4; // ... and make this many more beats
// What a hand does is told in what it does to the fingers and the hand, so that it is the same for both hands:
// [roll, out, rise] as multiples of its size: the fingers up, the hand away from the middle of the picture, the hand up.
// The kinds of beat (one is used for a whole burst) ...
const MOTIFS = [[-0.5, 0, -1], [0.5, 0, 1], [1, 0.3, 0], [-1, -0.2, 0], [0.3, 1, 0.3], [-0.2, -0.9, 0], [0.8, 0.7, 0.6]];
// ... and the places a hand is raised to (a speaker's gesture space).
const PLACES = [[0.5, 0.6, 0.5], [0.6, 0.1, 0.8], [0.8, 0.4, 0.2], [-0.3, 0.5, -0.1], [-0.5, -0.4, 0.3]];
// The body:
const SHIFT_EVERY = [8, 16]; // it shifts its weight this often
const SHIFT_SIZE = 0.6; // of the sway, at most
const SHIFT_LEAST = 0.35; // and by this much at the least
const SHIFT_RATE = 1; // critically damped: a shift takes about four seconds
const SHIFT_SHARE = 0.7; // of the sway: the shifts of weight ...
const MICRO_SHARE = 0.12; // ... and the small unevenness of standing
const PULSE_SHARE = 0.08; // of the lift, at most: the shoulders rise a little with a strong beat

const mulberry32 = (seed) => {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = Math.imul(state ^ (state >>> 15), state | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
};

const between = (random, [low, high]) => low + (high - low) * random();

// Smooth random wandering in about [-1, 1] that never repeats: random values at regular intervals joined by a smooth curve
// (one per period), the slow periods counting for more. It is asked for later times only.
class Wander {
  constructor(random, periods) {
    this.random = random;
    this.total = periods.reduce((sum, period) => sum + period, 0);
    this.octaves = periods.map((period) => ({ period, index: 0, knots: [0, 1, 2, 3].map(() => 2 * random() - 1) }));
  }

  at(t) {
    let sum = 0;
    for (const octave of this.octaves) {
      const position = Math.max(0, t) / octave.period;
      const index = Math.floor(position);
      while (octave.index < index) {
        octave.knots.shift();
        octave.knots.push(2 * this.random() - 1);
        octave.index++;
      }
      const u = position - index;
      const [a, b, c, d] = octave.knots;
      const curve = 0.5 * (2 * b + (c - a) * u + (2 * a - 5 * b + 4 * c - d) * u * u + (3 * (b - c) + d - a) * u * u * u);
      sum += octave.period * curve;
    }
    return sum / this.total;
  }
}

// 0 before and after the push, 1 at its apex, and no speed at either end (or at the apex).
const bump = (u) => {
  if (!(u > 0 && u < 1)) return 0;
  return u < STROKE_APEX
    ? 0.5 - 0.5 * Math.cos(Math.PI * u / STROKE_APEX)
    : 0.5 + 0.5 * Math.cos(Math.PI * (u - STROKE_APEX) / (1 - STROKE_APEX));
};

// A pose component (a multiple of a limit) that stays inside the limit: its own up to KNEE, and beyond that bent towards
// the limit without a kink, so that a hand that is asked for more eases up to the limit instead of stopping at it.
const soften = (value) => {
  const size = Math.abs(value);
  if (size <= KNEE) return value;
  return Math.sign(value) * (KNEE + (1 - KNEE) * Math.tanh((size - KNEE) / (1 - KNEE)));
};

// Moves a spring (its place x and its speed v) `step` seconds towards `goal`. `rate` is its natural frequency (radians per
// second) and `damping` its damping ratio: 1 is critical, less overshoots a little. Taken in small steps, so any step will do.
const follow = (spring, goal, rate, damping, step) => {
  const count = Math.max(1, Math.ceil(step / MAX_STEP));
  const h = step / count;
  for (let n = 0; n < count; n++) {
    spring.v += (rate * rate * (goal - spring.x) - 2 * damping * rate * spring.v) * h;
    spring.x += spring.v * h;
  }
};

// How a spring answers a push of height 1 and `length` seconds (the shape of `bump`): when the answer peaks, counted from the
// start of the push, and how high it gets.
const answer = (rate, damping, length) => {
  const spring = { x: 0, v: 0 };
  let top = 0;
  let at = 0;
  for (let t = 0; t < length + 0.6; t += MAX_STEP) {
    follow(spring, bump((t + MAX_STEP / 2) / length), rate, damping, MAX_STEP);
    if (spring.x > top) {
      top = spring.x;
      at = t + MAX_STEP;
    }
  }
  return { at, top };
};

// Which way a turn of the hand lifts its fingers (+1 or -1): the fingers lie along `axis` from the wrist, and a positive
// turn swings them clockwise on the picture.
const liftOf = (hand) => (hand.axis && hand.axis[0] > 0 ? -1 : 1);

// A flick of `value` (a multiple of a limit) for a hand that is held at `held`, so that the two together stay within
// HEADROOM and the limit never flattens the flick: cut short if the hand has some room that way, and the other way if it has
// hardly any (a hand that is held high comes down in a beat rather than going higher).
const inRoom = (value, held) => {
  const sign = Math.sign(value);
  const room = HEADROOM - sign * held;
  if (Math.abs(value) <= room) return value;
  if (room >= LEAST_ROOM * Math.abs(value)) return sign * room;
  return -sign * Math.min(Math.abs(value), HEADROOM + sign * held);
};

// Which way is out for a hand: away from the middle of the picture.
const outOf = (hand, size) => (hand.box && size && hand.box[0] + hand.box[2] < size[0] ? -1 : 1);

// Finds the stresses of speech: peaks of its loudness (measured every 10 ms and smoothed to the length of a syllable) that
// stand out from how loud the speaker has been lately. Audio is fed in as it comes, in any cut, and the stresses are
// reported with the clock time the player will play them at, and so are the pauses (to the `pauses` list, if there is one).
class SpeechBeats {
  constructor() {
    this.reset();
  }

  reset() {
    this.older = 0; // the smoothed loudness two frames ago
    this.recent = 0; // and one frame ago
    this.mean = 0; // the speaker's loudness lately, over what is voiced
    this.voiced = 0;
    this.ceiling = 0; // the loudest of the last few seconds
    this.last = -Infinity;
    this.quiet = 0; // how many frames it has been since the last voiced one
    this.talking = false; // whether there has been voice since the last pause
  }

  // `samples` are mono floats, `start` is when the first is played. Returns [{at, strength}], strength in (0, 1].
  feed(samples, rate, start, pauses) {
    const hop = Math.max(1, Math.round(rate / ENVELOPE_RATE));
    const beats = [];
    for (let from = 0; from + hop <= samples.length; from += hop) {
      let sum = 0;
      for (let i = from; i < from + hop; i++) sum += samples[i] * samples[i];
      const rms = Math.sqrt(sum / hop);
      const { older, recent } = this;
      const current = recent + (rms - recent) * (rms > recent ? 0.45 : 0.12);
      this.older = recent;
      this.recent = current;
      this.ceiling = Math.max(current, this.ceiling * 0.9985);
      if (current > 0.15 * this.ceiling) {
        this.voiced++;
        this.mean += (current - this.mean) * Math.max(1 / this.voiced, 1 / 150);
        this.quiet = 0;
        this.talking = true;
      } else if (++this.quiet === PAUSE_FRAMES && this.talking) {
        this.talking = false;
        if (pauses) pauses.push({ at: start + (from + hop) / rate });
      }
      // The previous frame is a stress if it rose above the one before it, is not being overtaken, and stands out.
      const at = start + (from - hop) / rate;
      const stands = recent > 1.3 * this.mean && recent > 0.25 * this.ceiling && recent > 0.01;
      if (recent > older && recent >= current && stands && at - this.last >= BEAT_GAP) {
        this.last = at;
        beats.push({ at, strength: Math.min(1, recent / this.ceiling) });
      }
    }
    return beats;
  }
}

// The body's pose over time. `spec` is the avatar's body.json (null for an avatar without one), `intensity` scales all
// motion: 0 is none, 1 the normal strength, 2 the most (the breath and the shifts of weight are twice as large, the hands
// make two fifths more beats and reach 30% further). A hand never goes past its limits, whatever the intensity.
class BodyMotion {
  constructor(spec, options) {
    const { intensity = 1, seed = 1 } = options || {};
    this.spec = spec && spec.version === 1 ? spec : null;
    this.intensity = Number.isFinite(intensity) ? Math.min(2, Math.max(0, intensity)) : 1;
    const above = Math.max(0, this.intensity - 1);
    this.reach = Math.min(1, this.intensity) + REACH_GROWTH * above;
    this.lively = Math.min(1, this.intensity) + LIVELY_GROWTH * above;
    this.active = Boolean(this.spec) && this.intensity > 0;
    this.random = mulberry32(seed);
    this.beats = new SpeechBeats();
    this.events = []; // stresses of the audio that is scheduled, in the order they are played
    this.gaps = []; // and its pauses
    this.time = 0;
    this.phase = 0;
    this.energy = 0; // how much the avatar is speaking, easing from 0 to 1
    this.pace = new Wander(this.random, [29, 11]);
    this.depth = new Wander(this.random, [23, 9]);
    this.micro = new Wander(this.random, [6.1, 3.3]);
    this.lean = { x: 0, v: 0 }; // the weight of the body on one foot or the other, and where it is shifting to
    this.leanTo = 0;
    this.shiftAt = between(this.random, [3, 7]);
    this.pulses = []; // how the shoulders rise on strong beats
    const trio = () => [0, 1, 2].map(() => ({ x: 0, v: 0 }));
    this.hands = ((this.spec && this.spec.hands) || []).map((hand) => ({
      lift: liftOf(hand),
      out: outOf(hand, this.spec.size),
      rate: FLICK_RATE * (0.85 + 0.3 * this.random()), // each hand has its own wrist
      damping: FLICK_DAMPING * (0.9 + 0.2 * this.random()),
      hold: trio(), // where the hand is, as multiples of its limits (turn, slide x, slide y) ...
      flick: trio(), // ... and what a flick adds to that
      place: [0, 0, 0], // where it is going to be held
      kick: [0, 0, 0], // how far the last beats left it on, fading
      pulses: [], // the pushes that make the flicks
      wander: [0, 1, 2].map(() => new Wander(this.random, WANDER_PERIODS)),
      revertAt: Infinity, // when an adjustment is taken back
    }));
    this.burst = null; // {lead, from, until} while the hands are gesturing
    this.motif = null;
    this.leader = -1; // the hand that led the last burst
    this.restUntil = 0; // no burst before this
    this.lastBeat = -Infinity;
    this.adjustAt = between(this.random, [4, 9]);
    this.spokeAt = -Infinity; // the last time the avatar was speaking
    this.speaking = false;
  }

  // Takes in audio that the player has scheduled, so that the beats can be started in time for its stresses.
  hear(samples, rate, start) {
    if (!this.active) return;
    for (const beat of this.beats.feed(samples, rate, start, this.gaps)) this.events.push(beat);
  }

  // The turn is over: nothing scheduled is going to be played.
  hush() {
    this.events = [];
    this.gaps = [];
    this.beats.reset();
  }

  // [turn, slide x, slide y] for a model [roll, out, rise] and a hand, as multiples of the limits, varied a little.
  aim(hand, [roll, out, rise], size) {
    const vary = () => (this.random() - 0.5) * 2 * JITTER;
    return [hand.lift * (roll + vary()) * size, hand.out * (out + vary()) * size, -(rise + vary()) * size];
  }

  // A hand's own place when it is let down: a little different from the photo's each time.
  resting(hand) {
    return this.aim(hand, [0, 0, 0], REST_SIZE * this.reach);
  }

  // Starts a burst of gestures: one hand leads, is raised to a place in front of the avatar and has a kind of beat; the
  // other one may be raised too.
  begin() {
    const count = this.hands.length;
    let lead = Math.floor(this.random() * count);
    if (count > 1 && lead === this.leader && this.random() < 0.7) lead = (lead + 1) % count;
    this.leader = lead;
    let motif = this.motif;
    while (motif === this.motif) motif = MOTIFS[Math.floor(this.random() * MOTIFS.length)];
    this.motif = motif;
    this.burst = { lead, from: this.time, until: this.time + between(this.random, BURST_SECONDS) };
    const chance = Math.min(0.7, OTHER_PLACE * this.lively);
    this.hands.forEach((hand, index) => {
      if (index !== lead && this.random() >= chance) {
        hand.place = this.resting(hand);
        return;
      }
      const size = between(this.random, PLACE_SIZE) * this.reach * (index === lead ? 1 : OTHER_SIZE);
      hand.place = this.aim(hand, PLACES[Math.floor(this.random() * PLACES.length)], size);
    });
  }

  // Ends the burst: the hands are let down, and the next burst comes after a rest.
  rest() {
    this.burst = null;
    this.restUntil = this.time + between(this.random, REST_SECONDS) / Math.sqrt(Math.max(0.25, this.lively));
    for (const hand of this.hands) hand.place = this.resting(hand);
  }

  // A listening avatar moves a hand a little, now and then, and takes it back a few seconds later.
  adjust() {
    const hand = this.hands[Math.floor(this.random() * this.hands.length)];
    const vector = [0, 1, 2].map(() => 2 * this.random() - 1);
    hand.place = this.aim(hand, vector, ADJUST_SIZE * this.reach);
    hand.revertAt = this.time + between(this.random, ADJUST_SECONDS);
  }

  // The body takes its weight to the other side.
  shiftWeight() {
    let to = this.leanTo;
    for (let tries = 0; tries < 6 && Math.abs(to - this.leanTo) < SHIFT_LEAST; tries++) to = between(this.random, [-SHIFT_SIZE, SHIFT_SIZE]);
    this.leanTo = to;
    this.shiftAt = this.time + between(this.random, SHIFT_EVERY);
  }

  // Makes a hand flick, with its apex `late` seconds after the stress at `at` (the audio clock). Returns false if it is too
  // late for that.
  stroke(index, size, at, clock, late) {
    const hand = this.hands[index];
    const length = (STROKE_SECONDS[0] + (STROKE_SECONDS[1] - STROKE_SECONDS[0]) * Math.min(1, size / 0.8)) * (0.9 + 0.2 * this.random());
    const aim = this.aim(hand, this.motif, size).map((value, i) => inRoom(value, hand.hold[i].x));
    // What is pushed is more than what the flick is to reach, as much as the spring holds back: so that it reaches `aim`, and
    // peaks when it should.
    const answers = [hand.rate, hand.rate * SLIDE_RATE, hand.rate * SLIDE_RATE].map((rate) => answer(rate, hand.damping, length));
    const start = this.time + (at - clock) + late - APEX_LEAD - answers[0].at;
    if (start < this.time - LATE_START) return false;
    hand.pulses.push({ start: Math.max(start, this.time), length, push: aim.map((value, i) => value / Math.max(0.4, answers[i].top)) });
    const carry = this.random() < 0.5 ? 0 : between(this.random, KICK_SIZE);
    hand.kick = hand.kick.map((value, i) => Math.max(-KICK_LIMIT, Math.min(KICK_LIMIT, value + carry * aim[i])));
    return true;
  }

  // A beat on a stress of `strength` that falls at `at` (the audio clock): the leading hand flicks and now and then the other
  // one does too, and the shoulders rise with a strong one.
  beat(strength, at, clock) {
    const { lead } = this.burst;
    const loud = Math.min(1, strength * (0.85 + 0.3 * this.random()));
    const size = this.reach * (STROKE_SIZE[0] + (STROKE_SIZE[1] - STROKE_SIZE[0]) * loud) * (0.8 + 0.4 * this.random());
    if (!this.stroke(lead, size, at, clock, 0)) return;
    this.lastBeat = this.time;
    if (strength > 0.5) this.pulses.push({ start: this.time + (at - clock) - STROKE_APEX * 0.7, length: 0.7, size: strength });
    const chance = Math.min(0.6, BOTH_HANDS * this.lively * (0.6 + strength));
    this.hands.forEach((_, index) => {
      if (index !== lead && this.random() < chance) this.stroke(index, size * OTHER_SIZE, at, clock, between(this.random, OTHER_LATE));
    });
  }

  // What to do about a stress that is about to be played: start a burst on it, and make a beat of it if it is likely.
  consider({ at, strength }, clock, speaking) {
    if (!this.hands.length) return;
    if (!this.burst && speaking && this.time >= this.restUntil && strength >= BURST_STRESS && this.random() < 0.8) this.begin();
    if (!this.burst || this.time - this.lastBeat < HAND_GAP) return;
    if (this.random() < Math.min(0.75, this.lively * (BEAT_CHANCE[0] + BEAT_CHANCE[1] * strength))) this.beat(strength, at, clock);
  }

  // Starts what is due at `clock` (the audio clock): the beats whose stress it is about to reach, the bursts, the end of a
  // burst at a pause, a shift of weight, and the adjustments of a listening avatar; lets the hands down when the speech has
  // been over for a while.
  advance(clock, speaking) {
    const time = this.time;
    if (speaking && !this.speaking) this.restUntil = time + between(this.random, OPENING_SECONDS);
    this.speaking = speaking;
    if (speaking) this.spokeAt = time;
    let paused = false;
    while (this.gaps.length && this.gaps[0].at <= clock) {
      this.gaps.shift();
      paused = true;
    }
    if (paused && time > this.shiftAt - 5 && this.random() < 0.5) this.shiftWeight();
    const over = this.burst && (time >= this.burst.until || (paused && time - this.burst.from > BURST_AGE && this.random() < BURST_ENDS));
    if (over || (this.burst && time - this.spokeAt > LINGER_SECONDS)) this.rest();
    while (this.events.length && this.events[0].at - LEAD_IN <= clock) {
      const event = this.events.shift();
      if (clock - event.at <= BEAT_LATE) this.consider(event, clock, speaking);
    }
    if (speaking && this.hands.length) {
      if (!this.burst && time >= this.restUntil + STARVED_SECONDS) this.begin();
      if (this.burst && !this.events.length && time - this.lastBeat > FALLBACK_SECONDS) this.beat(0.25, clock + 0.4, clock);
    }
    if (!speaking && !this.burst && this.hands.length && time >= this.adjustAt) {
      this.adjust();
      this.adjustAt = time + between(this.random, ADJUST_EVERY);
    }
    if (speaking) this.adjustAt = Math.max(this.adjustAt, time + 3);
    for (const hand of this.hands) {
      if (time >= hand.revertAt) {
        hand.revertAt = Infinity;
        hand.place = this.resting(hand);
      }
    }
    if (time >= this.shiftAt) this.shiftWeight();
  }

  // The pose `dt` seconds on, with the audio clock at `clock`: {sway: [x, y], hands: [{angle, x, y}, ...]} (the hands in
  // the order of body.json), or null where nothing moves.
  tick(dt, clock, speaking) {
    if (!this.active) return null;
    const step = Math.min(0.1, Math.max(0, dt));
    this.time += step;
    const t = this.time;
    this.energy += ((speaking ? 1 : 0) - this.energy) * (1 - Math.exp(-step / 0.5));
    this.advance(clock, Boolean(speaking));
    const { sway, lift } = this.spec;
    const scale = this.intensity;
    follow(this.lean, this.leanTo, SHIFT_RATE, 1, step);
    this.phase = (this.phase + step * TAU / (BREATH_SECONDS * (1 + 0.12 * this.pace.at(t)))) % TAU;
    // The breath rises faster than it falls, and its depth wanders.
    const breath = (0.85 + 0.15 * this.depth.at(t)) * 0.5 * (1 - Math.cos(this.phase - 0.35 * Math.sin(this.phase)));
    this.pulses = this.pulses.filter((pulse) => t < pulse.start + pulse.length);
    const pulse = Math.min(1, this.pulses.reduce((sum, p) => sum + p.size * bump((t - p.start) / p.length), 0));
    const body = 0.4 + 0.6 * this.energy; // a speaking body moves more than a listening one
    return {
      sway: [sway * scale * body * (SHIFT_SHARE * this.lean.x + MICRO_SHARE * this.micro.at(t)), -lift * scale * (0.7 * breath + PULSE_SHARE * pulse)],
      hands: this.hands.map((state, index) => this.pose(state, this.spec.hands[index], t, step)),
    };
  }

  // A hand's pose: where it is held (a spring after its place, plus its drift and what the beats left on), plus the flicks (a
  // quick spring after their pushes), each a multiple of a limit and never as much as the limit.
  pose(state, limits, t, step) {
    state.pulses = state.pulses.filter((push) => t < push.start + push.length);
    const wander = (WANDER_QUIET + (WANDER_TALK - WANDER_QUIET) * this.energy) * this.reach;
    const fade = Math.exp(-step / KICK_SECONDS);
    const value = [0, 1, 2].map((i) => {
      const pushed = state.pulses.reduce((sum, push) => sum + push.push[i] * bump((t - step / 2 - push.start) / push.length), 0);
      follow(state.flick[i], pushed, i ? state.rate * SLIDE_RATE : state.rate, state.damping, step);
      state.kick[i] *= fade;
      follow(state.hold[i], state.place[i] + state.kick[i] + wander * state.wander[i].at(t), HOLD_RATE, 1, step);
      return soften(state.hold[i].x + state.flick[i].x);
    });
    return { angle: (limits.turn || 0) * value[0], x: (limits.slide || 0) * value[1], y: (limits.slide || 0) * value[2] };
  }
}
