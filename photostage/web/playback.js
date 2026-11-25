"use strict";

function decodeBase64(encoded) {
  const raw = atob(encoded);
  const bytes = new Uint8Array(raw.length);
  for (let i = 0; i < bytes.length; i++) bytes[i] = raw.charCodeAt(i);
  return bytes;
}

// Frames and guides are data, not pictures: no colour management or alpha handling may touch them.
function decodeImage(encoded, type) {
  return createImageBitmap(new Blob([decodeBase64(encoded)], { type }), {
    colorSpaceConversion: "none", premultiplyAlpha: "none",
  });
}

function closeAll(images) {
  for (const image of new Set(images)) if (image) image.close();
}

// Schedules streamed speech: audio on the Web Audio clock and head frames (with the
// guides that carry the photo's detail onto them) against that same clock, so lips
// stay in sync. The owner draws whatever poll() returns.
class LivePlayback {
  constructor() {
    this.context = null;
    this.sources = new Set();
    this.frames = [];
    this.epoch = 0;
    this.next = 0;
    this.done = false;
    this.active = false;
    this.phase = "idle";
    this.stats = {};
    this.initialBufferSeconds = 1.3;
    this.maxLeadSeconds = 4;
    this.pending = [];
    this.pendingSeconds = 0;
    this.bufferTarget = this.initialBufferSeconds;
    this.arrivals = [];
    this.received = 0;
    this.total = 0;
    this.audioTimeline = [];
    this.playedSeconds = 0;
    this.alignStart = null;
    this.previousRate = 0.65;
  }

  reset() {
    if (Number.isFinite(this.stats.renderRate) && this.stats.renderRate > 0) this.previousRate = this.stats.renderRate;
    this.epoch++;
    for (const source of this.sources) source.stop();
    this.sources.clear();
    for (const frame of this.frames) closeAll([frame.image, frame.guide]);
    this.frames = [];
    for (const packet of this.pending) closeAll([...packet.images, ...packet.guides]);
    this.pending = [];
    this.pendingSeconds = 0;
    this.bufferTarget = this.initialBufferSeconds;
    this.arrivals = [];
    this.received = 0;
    this.total = 0;
    this.audioTimeline = [];
    this.playedSeconds = 0;
    this.alignStart = null;
    this.next = 0;
    this.done = false;
    this.active = false;
    this.phase = "idle";
  }

  // Creates and resumes the audio clock while a click is being handled, for a turn that starts after it: a recording's,
  // which some browsers would otherwise not let play.
  unlock() {
    const Audio = window.AudioContext || window.webkitAudioContext;
    if (!Audio) return;
    if (!this.context) this.context = new Audio();
    this.context.resume().catch(() => {});
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
    this.active = true;
    this.phase = "buffering";
  }

  async enqueue(segment) {
    const epoch = this.epoch;
    while (this.next - this.context.currentTime > 2.4) {
      await new Promise((resolve) => setTimeout(resolve, 25));
      if (epoch !== this.epoch) return false;
    }
    const guided = Array.isArray(segment.guides);
    const settled = await Promise.allSettled([
      ...segment.frames.map((encoded) => decodeImage(encoded, "image/jpeg")),
      ...(guided ? segment.guides : []).map((encoded) => decodeImage(encoded, "image/png")),
    ]);
    const decoded = settled.filter((result) => result.status === "fulfilled").map((result) => result.value);
    const failure = settled.find((result) => result.status === "rejected");
    if (failure || epoch !== this.epoch) {
      closeAll(decoded);
      if (failure) throw failure.reason;
      return false;
    }
    const images = decoded.slice(0, segment.frames.length), guides = decoded.slice(segment.frames.length);
    const bytes = decodeBase64(segment.audio);
    const data = new DataView(bytes.buffer);
    const buffer = this.context.createBuffer(1, bytes.length / 2, segment.rate);
    const samples = buffer.getChannelData(0);
    for (let i = 0; i < samples.length; i++) samples[i] = data.getInt16(i * 2, true) / 32768;
    this.pending.push({ images, guides, buffer, fps: segment.fps });
    this.pendingSeconds += buffer.duration;
    this.received += buffer.duration;
    if (segment.total > 0) this.total = segment.total;
    this.arrivals.push({ at: performance.now() / 1000, seconds: buffer.duration });
    if (this.arrivals.length > 4) this.arrivals.shift();
    this.stats.segments++;
    this.schedule();
    return true;
  }

  // Speech to hold back before (re)starting so that a renderer slower than real time still
  // finishes before the audio does: at a render rate r < 1 the buffer must cover
  // remaining * (1 / r - 1). The wait is capped; beyond it the rebuffering handles the rest.
  lead() {
    const recent = this.arrivals;
    if (recent.length < 2 || !(this.total > 0)) return 0;
    const span = recent[recent.length - 1].at - recent[0].at;
    const rendered = recent.slice(1).reduce((sum, arrival) => sum + arrival.seconds, 0);
    const rate = span > 0.05 ? rendered / span : Infinity;
    this.stats.renderRate = Math.min(rate, 99);
    if (rate >= 1) return 0;
    const remaining = Math.max(0, this.total - this.received);
    return Math.min(this.maxLeadSeconds, remaining * (1 / rate - 1) + 0.25);
  }

  // Forecast the same adaptive buffer before choosing a body phase. Whole one-second packets and transport/decode
  // headroom matter here: a missed phase otherwise costs a complete loop. Late data still takes the safe next loop.
  preparationDelay(total) {
    const rate = Math.min(1, this.previousRate);
    const needed = Math.max(this.initialBufferSeconds, Math.min(this.maxLeadSeconds, total * (1 - rate) + 0.25 * rate));
    return Math.min(total, Math.ceil(needed)) / rate + 1.5;
  }

  schedule() {
    if (!this.pending.length) return;
    const now = this.context.currentTime;
    if (this.next && this.next < now + 0.025) {
      this.stats.underruns++;
      this.next = 0;
      this.bufferTarget = Math.min(2.4, this.bufferTarget + 0.5);
    }
    if (!this.next && !this.done && this.pendingSeconds < Math.max(this.bufferTarget, this.lead())) {
      this.phase = "buffering";
      return;
    }
    // Keep on-time chunks contiguous; add scheduling headroom only on start or restart.
    let start = this.next || now + 0.05;
    if (!this.next && this.alignStart) {
      start = this.alignStart(start);
      if (!Number.isFinite(start) || start < now + 0.025) throw new Error("Invalid motion-aligned audio start.");
      this.alignStart = null;
    }
    for (const packet of this.pending) {
      this.schedulePacket(packet, start);
      start += packet.buffer.duration;
    }
    this.next = start;
    this.pending = [];
    this.pendingSeconds = 0;
    this.stats.maxQueuedFrames = Math.max(this.stats.maxQueuedFrames, this.frames.length);
  }

  schedulePacket({ images, guides, buffer, fps }, start) {
    const source = this.context.createBufferSource();
    source.buffer = buffer;
    source.connect(this.context.destination);
    source.onended = () => { source.disconnect(); this.sources.delete(source); };
    this.sources.add(source);
    source.start(start);
    this.audioTimeline.push({ start, duration: buffer.duration });
    if (this.onAudio) this.onAudio(buffer.getChannelData(0), buffer.sampleRate, start);
    images.forEach((image, index) => this.frames.push({ at: start + index / fps, image, guide: guides[index] || null }));
  }

  // currentTime is the audio rendering clock, ahead of what the output device is actually playing.
  get audibleTime() {
    const context = this.context;
    if (!context) return 0;
    if (context.state === "running" && typeof context.getOutputTimestamp === "function") {
      const stamp = context.getOutputTimestamp();
      if (Number.isFinite(stamp.contextTime) && stamp.contextTime >= 0 &&
          Number.isFinite(stamp.performanceTime) && stamp.performanceTime > 0) {
        return Math.max(0, Math.min(context.currentTime,
          stamp.contextTime + (performance.now() - stamp.performanceTime) / 1000));
      }
    }
    return context.currentTime;
  }

  // Only audible samples advance gesture timing; rebuffering and suspended audio clocks do not.
  get position() {
    if (!this.context) return 0;
    const now = this.audibleTime;
    while (this.audioTimeline.length && now >= this.audioTimeline[0].start + this.audioTimeline[0].duration) {
      this.playedSeconds += this.audioTimeline.shift().duration;
    }
    const next = this.audioTimeline[0];
    return this.playedSeconds + (next ? Math.max(0, now - next.start) : 0);
  }

  // Returns the newest frame that is due with its guide (the caller owns and must
  // close both) and whether every scheduled sample has now been played.
  poll() {
    const now = this.audibleTime;
    let newest = null;
    while (this.frames.length && this.frames[0].at <= now) {
      if (newest) closeAll([newest.image, newest.guide]);
      newest = this.frames.shift();
    }
    if (newest) {
      if (this.stats.firstPlaybackSeconds === null) {
        this.stats.firstPlaybackSeconds = (performance.now() - this.started) / 1000;
      }
      if (!this.pending.length) this.phase = "streaming";
    }
    if (!this.done && this.next && now >= this.next + 0.2 && this.phase === "streaming") {
      this.phase = "buffering";
    }
    return {
      image: newest && newest.image, guide: newest && newest.guide,
      finished: this.done && now >= this.next,
    };
  }
}
