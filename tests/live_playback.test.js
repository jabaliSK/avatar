"use strict";

const assert = require("assert");
const fs = require("fs");
const path = require("path");
const vm = require("vm");

const node = { addEventListener() {} };
const browser = vm.createContext({
  document: {
    addEventListener() {},
    body: { classList: { toggle() {} } },
    getElementById: () => node,
  },
  window: node,
  location: { search: "" },
  URLSearchParams,
  fetch: () => new Promise(() => {}),
  cancelAnimationFrame() {},
});
vm.runInContext(fs.readFileSync(path.join(__dirname, "../web/app.js"), "utf8"), browser);

function player() {
  const playback = vm.runInContext("new LivePlayback()", browser);
  const starts = [];
  playback.context = { currentTime: 1 };
  playback.schedulePacket = (packet, start) => starts.push(start);
  return { playback, starts };
}

function deliver(playback, duration) {
  playback.pending.push({ images: [], buffer: { duration }, fps: 25 });
  playback.pendingSeconds += duration;
  playback.schedule();
}

{
  const { playback, starts } = player();
  deliver(playback, 0.96);
  assert.deepStrictEqual(starts, []);
  deliver(playback, 0.96);
  assert.strictEqual(starts.length, 2);
  const end = playback.next;
  playback.context.currentTime = end - 0.05;
  deliver(playback, 0.96);
  assert.strictEqual(starts[2], end, "an on-time chunk must not insert silence");
}

{
  const { playback, starts } = player();
  deliver(playback, 1.4);
  playback.context.currentTime = playback.next + 0.3;
  playback.stats.underruns = 0;
  deliver(playback, 0.96);
  assert.strictEqual(playback.stats.underruns, 1);
  assert.strictEqual(starts.length, 1);
  deliver(playback, 0.96);
  assert.strictEqual(starts.length, 3);
  assert.strictEqual(starts[2], starts[1] + 0.96);
}

{
  const { playback, starts } = player();
  deliver(playback, 0.4);
  assert.strictEqual(starts.length, 0);
  playback.done = true;
  playback.schedule();
  assert.strictEqual(starts.length, 1, "completion must release a short reply");
}
