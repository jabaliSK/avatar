"use strict";

const assert = require("assert");
const fs = require("fs");
const path = require("path");
const vm = require("vm");

const browser = vm.createContext({});
vm.runInContext(fs.readFileSync(path.join(__dirname, "../photostage/web/conversation.js"), "utf8"), browser);
const { S, make } = vm.runInContext(
  "({ S: CONVERSATION_STATES, make: (onChange) => new Conversation(onChange) })", browser);

function machine() {
  const seen = [];
  const c = make((state, from, cause) => seen.push(`${from}>${state}:${cause}`));
  return { c, seen };
}

// Typing alone listens; clearing the box goes back to idle; a sent message thinks, then speaks at the first sound.
{
  const { c, seen } = machine();
  assert.strictEqual(c.state, S.IDLE);
  c.input("typing", true, 1);
  assert.strictEqual(c.state, S.LISTENING);
  c.input("typing", false, 2);
  assert.strictEqual(c.state, S.IDLE);
  c.input("typing", true, 3);
  assert.strictEqual(c.ask(4), true);
  c.input("typing", false, 4); // the box is cleared as the message goes
  assert.strictEqual(c.state, S.THINKING);
  assert.strictEqual(c.speaking(5.25), true);
  assert.strictEqual(c.state, S.SPEAKING);
  assert.strictEqual(c.wait, 1.25);
  assert.strictEqual(c.end(9, "done"), true);
  assert.strictEqual(c.state, S.IDLE);
  assert.deepStrictEqual(seen, [
    "IDLE>LISTENING:typing", "LISTENING>IDLE:typing", "IDLE>LISTENING:typing",
    "LISTENING>THINKING:ask", "THINKING>SPEAKING:speech", "SPEAKING>IDLE:done",
  ]);
}

// A recording that ends goes straight from LISTENING to THINKING, with no idle moment in between.
{
  const { c, seen } = machine();
  c.input("voice", true, 1);
  c.ask(3);
  c.input("voice", false, 3);
  assert.strictEqual(c.state, S.THINKING);
  assert.deepStrictEqual(seen, ["IDLE>LISTENING:voice", "LISTENING>THINKING:ask"]);
}

// Input during a turn does not interrupt it, and the turn then ends in LISTENING; one input still on keeps it there.
{
  const { c } = machine();
  c.ask(1);
  c.input("typing", true, 2);
  assert.strictEqual(c.state, S.THINKING);
  c.speaking(3);
  c.input("voice", true, 4);
  assert.strictEqual(c.state, S.SPEAKING);
  c.end(5, "stopped");
  assert.strictEqual(c.state, S.LISTENING);
  c.input("voice", false, 6);
  assert.strictEqual(c.state, S.LISTENING);
  c.input("typing", false, 7);
  assert.strictEqual(c.state, S.IDLE);
}

// Out-of-order events change nothing: no speech before a message, no second message while one is answered, no end without
// a turn. A turn that fails before any sound goes back without speaking.
{
  const { c, seen } = machine();
  assert.strictEqual(c.speaking(1), false);
  assert.strictEqual(c.end(1, "done"), false);
  c.ask(2);
  assert.strictEqual(c.ask(2.5), false);
  assert.strictEqual(c.end(3, "error"), true);
  assert.strictEqual(c.state, S.IDLE);
  assert.strictEqual(c.wait, null);
  assert.strictEqual(c.speaking(4), false);
  assert.deepStrictEqual(seen, ["IDLE>THINKING:ask", "THINKING>IDLE:error"]);
  assert.deepStrictEqual(c.log.map((entry) => entry.at).join(), "2,3");
}

console.log("photostage conversation tests passed");
