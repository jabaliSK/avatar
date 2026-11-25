"use strict";
// What the player page shows is a set of options. The server defines them (name, kind, default) and saves a value per
// avatar; the address of the page can override them for one browser, as in ?chat=0&mic=0, and the address wins.

function parseOption(option, text) {
  const word = String(text).trim().toLowerCase();
  if (option.kind === "switch") {
    if (["1", "true", "on", "yes"].indexOf(word) >= 0) return true;
    if (["0", "false", "off", "no"].indexOf(word) >= 0) return false;
  } else if (option.kind === "choice") {
    const found = option.choices.find((choice) => choice.value === word);
    if (found) return found.value;
  } else if (option.kind === "number" && word !== "" && Number.isFinite(Number(word))) {
    return Math.min(option.max, Math.max(option.min, Number(word)));
  }
  return undefined;
}

// `saved` holds the avatar's own values (the server sends every option, defaults included) and `params` is the page's
// URLSearchParams. An address value that makes no sense is ignored rather than guessed at.
function resolveOptions(schema, saved, params) {
  const values = {};
  for (const option of schema) {
    values[option.name] = saved && saved[option.name] !== undefined ? saved[option.name] : option.default;
    if (params.has(option.name)) {
      const given = parseOption(option, params.get(option.name));
      if (given !== undefined) values[option.name] = given;
    }
  }
  return values;
}

// A switch whose option needs another one (the microphone needs the chat box) is off while that one is off.
function activeOptions(schema, values) {
  const active = Object.assign({}, values);
  for (const option of schema) {
    if (option.needs && option.kind === "switch" && !values[option.needs]) active[option.name] = false;
  }
  return active;
}
