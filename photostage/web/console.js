"use strict";
// The console: every avatar as a card, and what is needed to manage them: open or copy the link, set what its page shows,
// rename, delete, and make a new one from a photo. The page is built from text nodes, never from HTML strings, so a name
// can hold any characters.
const $ = (id) => document.getElementById(id);
const TOKEN_KEY = "photostage-admin-token";
const MAX_UPLOAD = 15 * 1024 * 1024;
const IMAGE_TYPES = ["image/png", "image/jpeg", "image/webp"];
const FOCUSABLE = 'a[href], button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])';
// Quick ways to set the switches that decide what a viewer can do; the individual options stay available below them.
const PRESETS = [
  { label: "All controls", options: { chat: true, mic: true, modes: true, stop: true, reply: true, status: true, fullscreen: true } },
  { label: "No microphone", options: { chat: true, mic: false, modes: true, stop: true, reply: true, status: true, fullscreen: true } },
  { label: "Avatar only", options: { chat: false, status: false, fullscreen: false, home: false } },
];
const RENDERER = {
  ready: "ready", busy: "busy", warming: "warming up…", error: "problem", checking: "checking…",
};

let schema = [], avatars = [], fresh = "", listError = "", loaded = false, lastRefresh = 0, rendererState = "checking";

function append(node, child) {
  if (Array.isArray(child)) child.forEach((item) => append(node, item));
  else if (child !== null && child !== undefined && child !== false) node.appendChild(child.nodeType ? child : document.createTextNode(String(child)));
}

// el("a", {class: "btn", href: "/x", onclick: handler}, "text", child, ...): the attributes that are live properties of a
// control (its value, whether it is checked) are set as properties; the others as attributes.
function el(tag, attributes) {
  const node = document.createElement(tag), given = attributes || {};
  Object.keys(given).forEach((name) => {
    const value = given[name];
    if (value === null || value === undefined || value === false) return;
    if (name === "class") node.className = value;
    else if (name.slice(0, 2) === "on") node.addEventListener(name.slice(2), value);
    else if (["value", "checked", "selected", "disabled", "hidden"].indexOf(name) >= 0) node[name] = value;
    else node.setAttribute(name, value === true ? "" : String(value));
  });
  for (let i = 2; i < arguments.length; i++) append(node, arguments[i]);
  return node;
}

const escapeAttribute = (text) => String(text).replace(/&/g, "&amp;").replace(/"/g, "&quot;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
const sizeText = (size) => (Array.isArray(size) && size.length === 2 ? size[0] + " × " + size[1] : "");
const decimal = (value) => Number(value).toFixed(1);

function dateText(iso) {
  const date = new Date(iso);
  return isNaN(date.getTime()) ? "" : date.toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric" });
}

function motionText(motion) {
  const hands = (motion && motion.hands) || 0;
  if (hands > 0) return { text: "Hands + sway", title: "Breathing, sway and moving hands.", moves: true };
  if (motion && motion.sway) return { text: "Sway", title: "Breathing and sway. The hands in this photo are not clear enough to move cleanly.", moves: false };
  return { text: "Head only", title: "Only the head moves.", moves: false };
}

function choiceLabel(option, value) {
  const found = option.choices.filter((choice) => choice.value === value)[0];
  return found ? found.label : String(value);
}

// What an avatar's options change from the defaults, as short phrases for its card.
function customised(avatar) {
  const notes = [];
  schema.forEach((option) => {
    const value = avatar.options[option.name];
    if (value === undefined || value === option.default) return;
    if (option.kind === "switch") notes.push(option.label + (value ? " on" : " off"));
    else if (option.kind === "choice") notes.push(option.label + ": " + choiceLabel(option, value));
    else notes.push(option.label + ": " + decimal(value));
  });
  return notes;
}

// ---- Talking to the server --------------------------------------------------------------------------------------------

class ApiError extends Error {
  constructor(message, status) {
    super(message);
    this.status = status;
  }
}

function savedToken() {
  try {
    return sessionStorage.getItem(TOKEN_KEY) || "";
  } catch (error) {
    return "";
  }
}

function rememberToken(value) {
  try {
    sessionStorage.setItem(TOKEN_KEY, value);
  } catch (error) {
    // Without storage the token is simply asked for again next time.
  }
}

async function explain(response) {
  try {
    const detail = (await response.json()).detail;
    if (typeof detail === "string") return detail;
    if (Array.isArray(detail)) return detail.map((item) => item.msg).join(" ");
  } catch (error) {
    // Not JSON: the status says enough.
  }
  return "The service answered " + response.status + ".";
}

// A request to the management API. When the service asks for an admin token, it is asked of the person once and kept
// for the tab, and the request is repeated.
async function api(path, init) {
  for (let attempt = 0; ; attempt++) {
    const request = Object.assign({}, init);
    request.headers = Object.assign({}, init && init.headers);
    if (savedToken()) request.headers["X-Admin-Token"] = savedToken();
    let response;
    try {
      response = await fetch(path, request);
    } catch (error) {
      throw new ApiError("Cannot reach the avatar service.", 0);
    }
    if (response.status === 401 && attempt < 3) {
      await askToken(attempt > 0 || Boolean(savedToken()));
      continue;
    }
    if (!response.ok) throw new ApiError(await explain(response), response.status);
    return response;
  }
}

let tokenPrompt = null;

// One prompt at a time, however many requests are waiting for the token.
function askToken(rejected) {
  if (!tokenPrompt) {
    tokenPrompt = new Promise((resolve, reject) => {
      let given = false, modal = null;
      const input = el("input", { type: "password", id: "admin-token", autocomplete: "current-password", required: true, "aria-describedby": "token-note" });
      const form = el("form", {
        class: "dialog-form",
        onsubmit: (event) => {
          event.preventDefault();
          if (!input.value) return;
          given = true;
          rememberToken(input.value);
          modal.close();
          resolve();
        },
      },
      el("div", { class: "dialog-body" },
        el("p", { class: rejected ? "form-error" : "hint", id: "token-note" },
          rejected ? "That token was not accepted. Try again." : "This console is protected. Enter the admin token to continue."),
        el("div", { class: "field" }, el("label", { for: "admin-token" }, "Admin token"), input)),
      el("div", { class: "dialog-foot" },
        el("button", { class: "btn", type: "button", onclick: () => modal.close() }, "Cancel"),
        el("button", { class: "btn primary", type: "submit" }, "Continue")));
      modal = openModal({
        title: "Admin token", content: form, focus: input,
        onClose: () => { if (!given) reject(new ApiError("The admin token is needed for this.", 401)); },
      });
    }).finally(() => { tokenPrompt = null; });
  }
  return tokenPrompt;
}

// ---- Dialogs ----------------------------------------------------------------------------------------------------------

const layers = [];
let dialogCount = 0;

// With a dialog open, what is behind it can be neither clicked nor tabbed into (and screen readers skip it).
function syncLayers() {
  const open = layers.length > 0;
  [$("top"), $("page")].forEach((node) => (open ? node.setAttribute("inert", "") : node.removeAttribute("inert")));
  layers.forEach((layer, index) => (index < layers.length - 1 ? layer.backdrop.setAttribute("inert", "") : layer.backdrop.removeAttribute("inert")));
}

function openModal(spec) {
  const opener = document.activeElement, titleId = "dialog-title-" + (++dialogCount);
  const modal = { closed: false, heading: el("h2", { id: titleId }, spec.title) };
  modal.dialog = el("div", { class: "dialog" + (spec.wide ? " wide" : ""), role: "dialog", "aria-modal": "true", "aria-labelledby": titleId, tabindex: "-1" },
    el("div", { class: "dialog-head" }, modal.heading,
      el("button", { class: "icon", type: "button", "aria-label": "Close", onclick: () => modal.close() }, "×")),
    spec.content);
  modal.backdrop = el("div", { class: "backdrop" }, modal.dialog);
  // Pressing down on the dim area closes; releasing there after selecting text in a field does not.
  modal.backdrop.addEventListener("mousedown", (event) => { if (event.target === modal.backdrop) modal.close(); });
  modal.close = () => {
    if (modal.closed) return;
    modal.closed = true;
    layers.splice(layers.indexOf(modal), 1);
    modal.backdrop.remove();
    syncLayers();
    if (opener && document.body.contains(opener) && opener.focus) {
      opener.focus();
    } else if (spec.restore) {
      // The control that opened the dialog may have been rebuilt meanwhile (the list is redrawn after a save).
      const target = spec.restore();
      if (target) target.focus();
    }
    if (spec.onClose) spec.onClose();
  };
  $("layer").appendChild(modal.backdrop);
  layers.push(modal);
  (spec.focus || modal.dialog).focus();
  syncLayers();
  return modal;
}

// Tab stays inside the open dialog.
function keepFocus(modal, event) {
  const nodes = Array.prototype.filter.call(modal.dialog.querySelectorAll(FOCUSABLE), (node) => node.offsetParent !== null);
  if (!nodes.length) {
    event.preventDefault();
    modal.dialog.focus();
    return;
  }
  const first = nodes[0], last = nodes[nodes.length - 1], active = document.activeElement;
  if (event.shiftKey && (active === first || active === modal.dialog)) {
    event.preventDefault();
    last.focus();
  } else if (!event.shiftKey && active === last) {
    event.preventDefault();
    first.focus();
  }
}

document.addEventListener("keydown", (event) => {
  const top = layers[layers.length - 1];
  if (!top) return;
  if (event.key === "Escape") {
    event.preventDefault();
    top.close();
  } else if (event.key === "Tab") {
    keepFocus(top, event);
  }
});

function toast(message, kind) {
  const node = el("div", { class: "toast" + (kind === "error" ? " error" : "") }, message);
  $("toasts").appendChild(node);
  setTimeout(() => node.remove(), kind === "error" ? 8000 : 4000);
}

// The clipboard is only open to secure pages, and this one is often served over plain HTTP on a LAN, so the older way is
// the fallback.
async function copyText(text) {
  if (navigator.clipboard && window.isSecureContext) {
    try {
      await navigator.clipboard.writeText(text);
      return true;
    } catch (error) {
      // Fall through to the older way.
    }
  }
  const before = document.activeElement, box = el("textarea", { class: "sr-only", readonly: true, "aria-hidden": "true" });
  box.value = text;
  document.body.appendChild(box);
  box.select();
  let copied = false;
  try {
    copied = document.execCommand("copy");
  } catch (error) {
    copied = false;
  }
  box.remove();
  if (before && before.focus) before.focus();
  return copied;
}

async function copyLink(avatar) {
  const copied = await copyText(location.origin + avatar.url);
  toast(copied ? "Link copied." : "Could not copy the link. Open the settings and copy it by hand.", copied ? "" : "error");
}

// ---- The list ---------------------------------------------------------------------------------------------------------

// How far a new avatar's animations are, on its card: a bar while they are made, the reason and a retry if they failed.
function buildNote(avatar) {
  const build = avatar.build || {};
  if (build.state === "building") {
    const step = build.step && build.steps ? `Step ${build.step}/${build.steps}: ${build.label || "working"}` : "Queued";
    const left = Number.isFinite(build.eta) && build.eta > 0 ? ` · ~${Math.max(1, Math.round(build.eta / 60))} min left` : "";
    const percent = Math.max(0, Math.min(100, Math.round(build.percent || 0)));
    return el("div", { class: "build", role: "status", title: "The avatar can be opened once its animations are made." },
      el("p", { class: "sub" }, `Creating the avatar… ${percent}%`),
      el("div", { class: "build-bar" }, el("div", { class: "build-fill", style: `width: ${percent}%` })),
      el("p", { class: "sub" }, step + left));
  }
  if (build.state === "failed") {
    return el("div", { class: "build failed", role: "alert" },
      el("p", { class: "sub", title: build.error || "" }, "Animations failed: " + (build.error || "unknown error")),
      el("button", { class: "btn small", type: "button", onclick: () => rebuild(avatar) }, "Retry"));
  }
  return null;
}

async function rebuild(avatar) {
  try {
    await api("/api/avatars/" + avatar.key + "/rebuild", { method: "POST" });
    toast("Making the animations again.");
  } catch (error) {
    toast(error.message, "error");
  }
  refresh();
}

function card(avatar) {
  const notes = customised(avatar), badge = motionText(avatar.motion);
  return el("li", { class: "card" + (avatar.key === fresh ? " fresh" : ""), "data-key": avatar.key },
    el("a", { class: "thumb", href: avatar.url, target: "_blank", rel: "noopener", "aria-label": "Open " + avatar.name },
      el("img", { src: avatar.thumb, alt: "Photo of " + avatar.name, loading: "lazy" }),
      el("span", { class: "badge" + (badge.moves ? " moves" : ""), title: badge.title }, badge.text)),
    el("div", { class: "card-body" },
      el("h2", { title: avatar.name }, avatar.name),
      el("p", { class: "sub" }, [sizeText(avatar.size), dateText(avatar.created)].filter(Boolean).join(" · ")),
      notes.length ? el("p", { class: "sub", title: notes.join(", ") }, "Custom: " + notes.join(", ")) : null,
      buildNote(avatar)),
    el("div", { class: "card-actions" },
      el("a", { class: "btn small primary", href: avatar.url, target: "_blank", rel: "noopener" }, "Open"),
      el("button", { class: "btn small", type: "button", onclick: () => copyLink(avatar) }, "Copy link"),
      el("button", { class: "btn small", type: "button", "data-action": "settings", onclick: () => openSettings(avatar) }, "Settings"),
      el("button", { class: "btn small danger", type: "button", "aria-label": "Delete " + avatar.name, onclick: () => deleteDialog(avatar) }, "Delete")));
}

function render() {
  const query = $("search").value.trim().toLowerCase();
  const shown = avatars.filter((avatar) => avatar.name.toLowerCase().indexOf(query) >= 0);
  $("busy").hidden = loaded;
  $("search").hidden = avatars.length === 0;
  $("notice").hidden = !listError;
  $("notice-text").textContent = listError;
  $("empty").hidden = !(loaded && !listError && avatars.length === 0);
  $("none").hidden = !(avatars.length > 0 && shown.length === 0);
  const grid = $("grid");
  grid.textContent = "";
  shown.forEach((avatar) => grid.appendChild(card(avatar)));
}

async function refresh() {
  lastRefresh = Date.now();
  try {
    avatars = (await (await api("/api/avatars")).json()).avatars;
    listError = "";
  } catch (error) {
    listError = error.message;
  }
  loaded = true;
  render();
}

function highlight() {
  $("search").value = "";
  render();
  const node = $("grid").querySelector(".fresh");
  if (node && node.scrollIntoView) node.scrollIntoView({ block: "nearest", behavior: "smooth" });
  setTimeout(() => {
    fresh = "";
    render();
  }, 8000);
}

function showRenderer(state, message) {
  const pill = $("renderer"), canCreate = state === "ready" || state === "busy";
  rendererState = state;
  pill.dataset.state = state;
  pill.textContent = "";
  // On a narrow screen only the state is shown ("ready"), which the colour of the dot supports.
  append(pill, el("span", {}, state === "offline" ? "Service not reachable" : [el("span", { class: "pill-prefix" }, "Renderer "), RENDERER[state]]));
  pill.title = message || "";
  [$("new"), $("empty-new")].forEach((button) => {
    button.disabled = !canCreate;
    button.title = canCreate ? "" : "The renderer is not ready yet.";
  });
}

async function poll() {
  if (document.hidden) return;
  const before = rendererState;
  try {
    const status = await (await fetch("/status", { cache: "no-store" })).json();
    showRenderer(status.state || (status.ready ? "ready" : "warming"), status.message);
    if (before === "offline" || avatars.some((avatar) => avatar.build && avatar.build.state === "building")) refresh();
  } catch (error) {
    showRenderer("offline", "");
  }
}

// ---- New avatar -------------------------------------------------------------------------------------------------------

function createDialog() {
  let file = null, previewUrl = "", working = false, typedName = false, modal = null;
  const input = el("input", { type: "file", id: "photo-file", class: "sr-only", accept: IMAGE_TYPES.join(","), onchange: () => choose(input.files && input.files[0]) });
  const prompt = el("div", { class: "drop-prompt" },
    el("strong", null, "Drop a photo here"), el("span", null, "or click to choose one"),
    el("span", { class: "hint" }, "A clear photo of one person. A full-body photo works too: the picture is kept exactly as it is."));
  const preview = el("div", { class: "drop-preview", hidden: true });
  const zone = el("label", { class: "drop", for: "photo-file" }, input, prompt, preview);
  const name = el("input", { type: "text", id: "avatar-name", maxlength: 60, placeholder: "For example: Front desk host", autocomplete: "off", oninput: () => { typedName = true; } });
  const consent = el("input", { type: "checkbox", id: "consent", onchange: () => update() });
  const problem = el("p", { class: "form-error", role: "alert", hidden: true });
  const progress = el("p", { class: "progress", role: "status" });
  const submit = el("button", { class: "btn primary", type: "submit", disabled: true }, "Create avatar");

  function update() {
    submit.disabled = !(file && consent.checked) || working;
  }

  function fail(message) {
    problem.textContent = message;
    problem.hidden = false;
  }

  function choose(picked) {
    problem.hidden = true;
    if (!picked) return;
    if (IMAGE_TYPES.indexOf(picked.type) < 0) return fail("Use a PNG, JPEG or WebP photo.");
    if (picked.size > MAX_UPLOAD) return fail("That photo is larger than 15 MB.");
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = URL.createObjectURL(picked);
    const image = el("img", { alt: "The chosen photo", src: previewUrl }), detail = el("span", { class: "hint" });
    image.onload = () => {
      const width = image.naturalWidth, height = image.naturalHeight;
      if (width * height > 16000000 || Math.max(width, height) > 8192) {
        file = null;
        fail("Use a photo of up to 16 megapixels and 8192 pixels per side.");
      } else if (Math.max(width, height) < 1024) {
        detail.textContent = width + " × " + height + ". A small photo makes a soft avatar; 1024 pixels or more on the long side is better.";
      } else {
        detail.textContent = width + " × " + height;
      }
      update();
    };
    file = picked;
    preview.textContent = "";
    append(preview, [image, el("strong", null, picked.name), detail, el("span", { class: "hint" }, "Click to choose another photo")]);
    prompt.hidden = true;
    preview.hidden = false;
    if (!typedName) name.value = picked.name.replace(/\.[^.]*$/, "").replace(/[_-]+/g, " ").trim().slice(0, 60);
    update();
  }

  function setWorking(on) {
    working = on;
    [input, name, consent].forEach((control) => { control.disabled = on; });
    update();
  }

  async function create(event) {
    event.preventDefault();
    if (!file || !consent.checked || working) return;
    problem.hidden = true;
    setWorking(true);
    const started = Date.now(), form = new FormData();
    const tick = () => { progress.textContent = "Creating the avatar… " + Math.round((Date.now() - started) / 1000) + " s. This takes up to a minute."; };
    tick();
    const timer = setInterval(tick, 1000);
    form.append("image", file);
    form.append("name", name.value.trim());
    form.append("consent", "true");
    try {
      const created = await (await api("/avatars", { method: "POST", body: form })).json();
      fresh = created.key;
      await refresh();
      modal.close();
      toast(created.build && created.build.state === "building" ? "Avatar uploaded. Its animations are being made; the card shows the progress." : "Avatar created.");
      highlight();
    } catch (error) {
      // Closing the dialog does not stop the work, so a failure is still told to the person.
      if (modal.closed) toast(error.message, "error");
      else fail(error.message);
    } finally {
      clearInterval(timer);
      progress.textContent = "";
      if (!modal.closed) setWorking(false);
    }
  }

  ["dragover", "dragenter"].forEach((type) => zone.addEventListener(type, (event) => {
    event.preventDefault();
    zone.classList.add("over");
  }));
  zone.addEventListener("dragleave", () => zone.classList.remove("over"));
  zone.addEventListener("drop", (event) => {
    event.preventDefault();
    zone.classList.remove("over");
    choose(event.dataTransfer && event.dataTransfer.files[0]);
  });

  const form = el("form", { class: "dialog-form", onsubmit: create },
    el("div", { class: "dialog-body" },
      zone,
      el("div", { class: "field" }, el("label", { for: "avatar-name" }, "Name"), name),
      el("label", { class: "check", for: "consent" }, consent,
        el("span", null, "I have the right to use this photo, and the person in it agrees to being animated.")),
      problem, progress),
    el("div", { class: "dialog-foot" },
      el("button", { class: "btn", type: "button", onclick: () => modal.close() }, "Cancel"), submit));
  modal = openModal({ title: "New avatar", content: form, onClose: () => { if (previewUrl) URL.revokeObjectURL(previewUrl); } });
}

// ---- Settings of one avatar -------------------------------------------------------------------------------------------

// The options are what the settings are built from; if they could not be loaded at the start, they are asked for again.
async function loadSchema() {
  if (schema.length) return true;
  try {
    schema = (await (await fetch("/api/options")).json()).options;
    render();
  } catch (error) {
    schema = [];
  }
  return schema.length > 0;
}

async function openSettings(avatar) {
  if (await loadSchema()) settingsDialog(avatar);
  else toast("The options could not be loaded. Reload the page.", "error");
}

function settingsDialog(avatar) {
  const link = location.origin + avatar.url;
  let values = Object.assign({}, avatar.options), pending = 0, queue = Promise.resolve(), modal = null;
  const saved = el("p", { class: "saved", role: "status" });
  const list = el("div", { class: "options" });
  const embedBox = el("pre", { class: "code" });
  const info = el("p", { class: "hint" });
  const name = el("input", { type: "text", id: "avatar-rename", maxlength: 60, value: avatar.name, autocomplete: "off", onchange: () => persist({ name: name.value }) });
  const linkField = el("input", { type: "text", id: "avatar-link", readonly: true, value: link, "aria-label": "Link to the avatar", onfocus: () => linkField.select() });

  const flag = (text, bad) => {
    saved.textContent = text;
    saved.classList.toggle("error", Boolean(bad));
  };
  const embedCode = () => '<iframe src="' + escapeAttribute(link) + '" title="' + escapeAttribute(avatar.name) +
    '" width="480" height="800" allow="microphone; fullscreen" style="border:0"></iframe>';

  function describeAvatar() {
    const parts = [sizeText(avatar.size), motionText(avatar.motion).text, dateText(avatar.created) && "Made " + dateText(avatar.created)];
    info.textContent = parts.filter(Boolean).join(" · ");
    embedBox.textContent = embedCode();
    modal.heading.textContent = avatar.name;
    name.value = avatar.name;
  }

  // A switch that depends on another (the microphone on the chat box) is dimmed and locked while that one is off.
  function dim() {
    schema.forEach((option) => {
      if (!option.needs) return;
      const row = list.querySelector('[data-name="' + option.name + '"]'), off = !values[option.needs];
      row.classList.toggle("off", off);
      row.querySelector("input, select").disabled = off;
    });
  }

  function optionRow(option) {
    const id = "option-" + option.name, value = values[option.name];
    let control = null;
    if (option.kind === "switch") {
      control = el("input", { type: "checkbox", role: "switch", id: id, checked: Boolean(value), onchange: () => change(option.name, control.checked) });
    } else if (option.kind === "choice") {
      control = el("select", { id: id, onchange: () => change(option.name, control.value) },
        option.choices.map((choice) => el("option", { value: choice.value, selected: choice.value === value }, choice.label)));
    } else {
      const shown = el("output", { for: id }, decimal(value));
      const range = el("input", {
        type: "range", id: id, min: option.min, max: option.max, step: option.step, value: value,
        oninput: () => { shown.textContent = decimal(range.value); },
        onchange: () => change(option.name, Number(range.value)),
      });
      control = el("span", { class: "slider" }, range, shown);
    }
    return el("div", { class: "option" + (option.needs ? " child" : "") + (option.kind === "switch" ? "" : " wide"), "data-name": option.name },
      el("label", { class: "option-text", for: id }, el("span", { class: "option-label" }, option.label), el("span", { class: "option-hint" }, option.hint)),
      control);
  }

  function renderOptions() {
    const focused = document.activeElement && list.contains(document.activeElement) ? document.activeElement.id : "";
    list.textContent = "";
    schema.forEach((option) => list.appendChild(optionRow(option)));
    dim();
    const again = focused && document.getElementById(focused);
    if (again) again.focus();
  }

  const differs = () => schema.some((option) => values[option.name] !== avatar.options[option.name]);

  // Changes are sent one after the other, so the server ends up with the last of them. When the last one has been
  // answered, the page shows what the server has, which matters when one of them was refused.
  function persist(body) {
    pending += 1;
    flag("Saving…");
    queue = queue.then(async () => {
      try {
        Object.assign(avatar, await (await api("/api/avatars/" + avatar.key, {
          method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
        })).json());
        flag("Saved.");
      } catch (error) {
        flag(error.message, true);
      }
      pending -= 1;
      if (pending === 0) {
        describeAvatar();
        if (differs()) {
          values = Object.assign({}, avatar.options);
          renderOptions();
        }
        render();
      }
    });
  }

  function change(optionName, value) {
    values[optionName] = value;
    dim();
    const body = { options: {} };
    body.options[optionName] = value;
    persist(body);
  }

  function preset(chosen) {
    Object.assign(values, chosen.options);
    renderOptions();
    persist({ options: chosen.options });
  }

  const accepted = (option) => (option.kind === "switch" ? "1 or 0" : option.kind === "choice"
    ? option.choices.map((choice) => choice.value).join(" or ") : option.min + " to " + option.max);

  const content = el("div", { class: "dialog-form" },
    el("div", { class: "dialog-body" },
      el("div", { class: "field" }, el("label", { for: "avatar-rename" }, "Name"), name, info),
      el("div", { class: "field" },
        el("h3", { class: "section-title" }, "What viewers see"),
        el("p", { class: "hint" }, "Each control can be switched off for this avatar. Changes save at once and apply the next time its page loads."),
        el("div", { class: "presets", role: "group", "aria-label": "Presets" },
          PRESETS.map((chosen) => el("button", { class: "btn small", type: "button", onclick: () => preset(chosen) }, chosen.label))),
        list, saved),
      el("div", { class: "field" },
        el("label", { for: "avatar-link" }, "Link"),
        el("div", { class: "linkbox" }, linkField,
          el("button", { class: "btn", type: "button", onclick: async () => {
            const copied = await copyText(link);
            if (copied) flag("Link copied.");
            else { linkField.select(); flag("Could not copy. The link is selected: copy it with Ctrl+C.", true); }
          } }, "Copy"),
          el("a", { class: "btn", href: avatar.url, target: "_blank", rel: "noopener" }, "Open")),
        el("details", null,
          el("summary", null, "Set options in the link"),
          el("p", { class: "hint" }, "Add them to the link to change one browser or screen without changing the avatar, for example " + link + "?chat=0&status=0. The link wins over the saved options."),
          el("table", null, el("tbody", null, schema.map((option) => el("tr", null,
            el("td", null, el("code", null, option.name)), el("td", null, option.label), el("td", null, accepted(option)))))))),
      el("div", { class: "field" },
        el("h3", { class: "section-title" }, "Embed in another page"),
        embedBox,
        el("div", null, el("button", { class: "btn small", type: "button", onclick: async () => {
          flag((await copyText(embedCode())) ? "Embed code copied." : "Could not copy the embed code.", false);
        } }, "Copy embed code")))),
    el("div", { class: "dialog-foot" },
      el("button", { class: "btn danger grow", type: "button", onclick: () => deleteDialog(avatar, () => modal.close()) }, "Delete avatar…"),
      el("button", { class: "btn primary", type: "button", onclick: () => modal.close() }, "Done")));
  modal = openModal({
    title: avatar.name, content: content, wide: true,
    restore: () => document.querySelector('[data-key="' + avatar.key + '"] [data-action="settings"]'),
  });
  describeAvatar();
  renderOptions();
}

// ---- Delete -----------------------------------------------------------------------------------------------------------

// The card that was deleted is gone with the button that had the focus: it goes to the card that took its place.
function focusNear(position) {
  const cards = $("grid").children, next = cards[Math.min(position, cards.length - 1)];
  const target = next ? next.querySelector(".card-actions a") : $("new");
  if (target && !target.disabled) target.focus();
}

function deleteDialog(avatar, afterDelete) {
  let modal = null;
  const problem = el("p", { class: "form-error", role: "alert", hidden: true });
  const cancel = el("button", { class: "btn", type: "button", onclick: () => modal.close() }, "Cancel");
  const confirm = el("button", { class: "btn danger", type: "submit" }, "Delete avatar");
  const form = el("form", {
    class: "dialog-form",
    onsubmit: async (event) => {
      event.preventDefault();
      confirm.disabled = true;
      problem.hidden = true;
      try {
        await api("/api/avatars/" + avatar.key, { method: "DELETE" });
      } catch (error) {
        // Already gone (deleted from another window) is the result that was wanted.
        if (error.status !== 404) {
          problem.textContent = error.message;
          problem.hidden = false;
          confirm.disabled = false;
          return;
        }
      }
      const position = Array.prototype.indexOf.call($("grid").children, $("grid").querySelector('[data-key="' + avatar.key + '"]'));
      avatars = avatars.filter((item) => item.key !== avatar.key);
      render();
      modal.close();
      if (afterDelete) afterDelete();
      toast("Deleted " + avatar.name + ".");
      focusNear(position);
    },
  },
  el("div", { class: "dialog-body" },
    el("p", null, "Delete ", el("strong", null, avatar.name), "? Its photo, its settings and its link are removed, and the link stops working. This cannot be undone."),
    problem),
  el("div", { class: "dialog-foot" }, cancel, confirm));
  modal = openModal({ title: "Delete this avatar?", content: form, focus: cancel });
}

// ---- Start ------------------------------------------------------------------------------------------------------------

$("new").onclick = createDialog;
$("empty-new").onclick = createDialog;
$("search").oninput = render;
$("notice-action").onclick = refresh;
document.addEventListener("visibilitychange", () => {
  if (document.hidden) return;
  poll();
  if (Date.now() - lastRefresh > 2000) refresh();
});

(async () => {
  showRenderer("checking", "");
  poll();
  setInterval(poll, 5000);
  await loadSchema();
  await refresh();
})();
