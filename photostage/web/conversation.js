"use strict";

// The player's conversation as four states, and what moves it between them. What the avatar shows in each:
//   IDLE       nobody is talking to it: the resting video loops, or the idle face and the body's sway.
//   LISTENING  the viewer is typing or recording: an attentive listening video loops.
//   THINKING   from the moment a message is sent, or a recording ends, until the reply's first sound: a thinking clip
//              loops over the transcription, the language model, the voice and the face renderer.
//   SPEAKING   from the first sound, which the first live face frame is drawn with, to the last: lip-sync and gestures.
// A turn that ends, is stopped or fails returns to LISTENING while the viewer is still typing or recording, else to IDLE.
// Each change is reported to onChange(state, from, cause).
const CONVERSATION_STATES = Object.freeze({ IDLE: "IDLE", LISTENING: "LISTENING", THINKING: "THINKING", SPEAKING: "SPEAKING" });
const CONVERSATION_LOG = 20; // the latest changes kept for review

class Conversation {
  constructor(onChange = () => {}) {
    this.state = CONVERSATION_STATES.IDLE;
    this.since = 0;
    this.inputs = new Set();
    this.asked = null;
    this.wait = null; // seconds from the latest message to its first sound
    this.log = [];
    this.onChange = onChange;
  }

  move(to, now, cause) {
    const from = this.state;
    if (from === to) return;
    this.state = to;
    this.since = now;
    this.log = this.log.concat({ from, to, at: now, cause }).slice(-CONVERSATION_LOG);
    this.onChange(to, from, cause);
  }

  // The viewer starts (active) or stops giving input from `source` ("typing" or "voice").
  input(source, active, now) {
    if (active) this.inputs.add(source);
    else this.inputs.delete(source);
    const { IDLE, LISTENING } = CONVERSATION_STATES;
    if (this.state === IDLE && this.inputs.size) this.move(LISTENING, now, source);
    else if (this.state === LISTENING && !this.inputs.size) this.move(IDLE, now, source);
  }

  // A message is sent, or a recording has ended and its transcript will be. True if a turn begins.
  ask(now) {
    const { IDLE, LISTENING, THINKING } = CONVERSATION_STATES;
    if (this.state !== IDLE && this.state !== LISTENING) return false;
    this.asked = now;
    this.wait = null;
    this.move(THINKING, now, "ask");
    return true;
  }

  // The reply's first sound is heard now.
  speaking(now) {
    if (this.state !== CONVERSATION_STATES.THINKING) return false;
    this.wait = now - this.asked;
    this.move(CONVERSATION_STATES.SPEAKING, now, "speech");
    return true;
  }

  // The turn is over: `outcome` is "done", "stopped" or "error".
  end(now, outcome) {
    const { THINKING, SPEAKING, IDLE, LISTENING } = CONVERSATION_STATES;
    if (this.state !== THINKING && this.state !== SPEAKING) return false;
    this.move(this.inputs.size ? LISTENING : IDLE, now, outcome);
    return true;
  }
}
