// The dialogs of the library: creating an avatar from a photo, its settings, adding motion to a quick preview, and deleting.
import { api } from "./api.js";
import { formatDuration, h } from "./dom.js";
import { icon } from "./icons.js";
import { confirmDialog, modal, toast } from "./ui.js";

const TYPES = ["image/png", "image/jpeg", "image/webp"];
const ESTIMATE_S = { hd: 13 * 60, standard: 8 * 60, quick: 60 };  // what the job runner measured on this GPU
const DEFAULTS = { expression: 0.6, head_motion: 0.5 };
const RANGES = { expression: [0.2, 1.2], head_motion: [0, 1] };

let sequence = 0;

/** Radio buttons drawn as a segmented control. */
function segmented(name, label, options, value, onChange) {
  const group = `${name}-${++sequence}`;
  const el = h("fieldset", { class: "seg" }, h("legend", { class: "sr-only" }, label));
  for (const option of options) {
    const input = h("input", {
      type: "radio", name: group, value: option.value, checked: option.value === value,
      onChange: () => onChange(option.value),
    });
    el.append(h("label", null, input, h("span", null, option.label)));
  }
  return el;
}

function formError() {
  return h("p", { class: "form-error", role: "alert", hidden: true });
}

function show(el, message) {
  el.textContent = message || "";
  el.hidden = !message;
}

// ---------------------------------------------------------------------------------------------------------------- a new avatar
function readSize(url) {
  return new Promise((resolve) => {
    const image = new Image();
    image.onload = () => resolve({ width: image.naturalWidth, height: image.naturalHeight });
    image.onerror = () => resolve(null);
    image.src = url;
  });
}

export function openNewAvatar({ limits = {}, onCreated }) {
  const maxMb = limits.upload_mb || 20;
  const minSide = limits.min_side || 256;
  let file = null;
  let previewUrl = "";
  let mode = "full";
  let quality = "hd";

  const dialog = modal({ title: "New avatar", subtitle: "Upload one photo of the person. A full-body or a portrait photo both work.", className: "wide" });
  const picker = h("input", { type: "file", class: "sr-only", accept: TYPES.join(","), tabindex: -1, "aria-hidden": "true" });
  const preview = h("img", { class: "drop-preview", alt: "Preview of the chosen photo", hidden: true });
  const empty = h("div", { class: "drop-empty" },
    icon("upload", 30),
    h("strong", null, "Drop a photo here or click to choose one"),
    h("span", null, `PNG, JPEG or WebP, up to ${maxMb} MB, at least ${minSide} pixels on each side`));
  const change = h("span", { class: "drop-change", hidden: true }, icon("refresh", 14), "Choose a different photo");
  const drop = h("div", { class: "drop", role: "button", tabindex: 0, "aria-label": "Choose a photo to upload" }, h("i", { class: "corners" }), empty, preview, change);
  const photoError = formError();
  const nameInput = h("input", { type: "text", class: "input", maxlength: 80, autocomplete: "off", placeholder: "For example: Aisha" });
  const subjectInput = h("textarea", {
    class: "input", rows: 2, maxlength: 600,
    placeholder: "For example: A woman in a navy blazer standing in a bright office",
  });
  const estimate = h("span", { class: "estimate" });
  const qualityRow = h("div", { class: "field" },
    h("span", null, "Quality"),
    segmented("quality", "Quality", [{ value: "hd", label: "HD" }, { value: "standard", label: "Standard" }], quality, (value) => { quality = value; refresh(); }));
  const choice = (value, title, text, badge) => h("label", { class: "choice" },
    h("input", { type: "radio", name: "mode", value, checked: value === mode, onChange: () => { mode = value; refresh(); } }),
    h("span", { class: "choice-body" },
      h("span", { class: "choice-title" }, title, badge ? h("em", null, badge) : null),
      h("span", { class: "choice-text" }, text)));
  const problem = formError();
  const cancel = h("button", { class: "btn btn-ghost", type: "button", onClick: () => dialog.close("") }, "Cancel");
  const submit = h("button", { class: "btn btn-primary", type: "submit", disabled: true }, icon("bolt", 16), h("span", null, "Create avatar"));
  const form = h("form", { novalidate: true, onSubmit: send },
    drop, photoError,
    h("label", { class: "field" }, h("span", null, "Name"), nameInput, h("small", null, "Shown in the library. Leave it empty to use the file name.")),
    h("label", { class: "field" }, h("span", null, "Description"), subjectInput,
      h("small", null, "Describe the person in English: clothing and setting. It guides the generated body motion.")),
    h("fieldset", { class: "choices" },
      h("legend", null, "Motion"),
      choice("full", "Full motion", "Natural breathing and posture while idle, and hand and body gestures while speaking.", "Recommended"),
      choice("quick", "Quick preview", "Only the face is animated on the still photo. Ready in about a minute; add full motion later.")),
    qualityRow,
    h("p", { class: "note" }, icon("clock", 16), estimate),
    h("p", { class: "note warn" }, icon("alert", 16), h("span", null, "Making motion uses the same GPU as live conversations, so they may respond more slowly until it finishes.")),
    problem,
    h("div", { class: "modal-foot" }, cancel, submit));

  function refresh() {
    qualityRow.hidden = mode === "quick";
    const seconds = mode === "quick" ? ESTIMATE_S.quick : ESTIMATE_S[quality];
    estimate.textContent = `Takes ${formatDuration(seconds)} once it starts. You can leave this page; it keeps going.`;
    submit.disabled = !file;
  }

  function clearPhoto() {
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = "";
    file = null;
    preview.hidden = true;
    preview.removeAttribute("src");
    empty.hidden = false;
    change.hidden = true;
    drop.classList.remove("has-photo");
  }

  async function choose(picked) {
    show(photoError, "");
    show(problem, "");
    if (!picked) return;
    if (!TYPES.includes(picked.type)) return reject("Use a PNG, JPEG or WebP photo.");
    if (picked.size > maxMb * 2 ** 20) return reject(`That photo is larger than ${maxMb} MB.`);
    const url = URL.createObjectURL(picked);
    const size = await readSize(url);
    if (!size) {
      URL.revokeObjectURL(url);
      return reject("That file could not be read as a photo.");
    }
    if (Math.min(size.width, size.height) < minSide) {
      URL.revokeObjectURL(url);
      return reject(`That photo is too small. Each side needs at least ${minSide} pixels.`);
    }
    if (Math.max(size.width, size.height) / Math.min(size.width, size.height) > 4) {
      URL.revokeObjectURL(url);
      return reject("That photo is too narrow. Its long side can be at most four times its short side.");
    }
    clearPhoto();
    file = picked;
    previewUrl = url;
    preview.src = url;
    preview.hidden = false;
    empty.hidden = true;
    change.hidden = false;
    drop.classList.add("has-photo");
    const fromFile = picked.name.replace(/\.[^.]+$/, "").replace(/[-_.]+/g, " ").trim();
    if (fromFile) nameInput.placeholder = `Leave empty to use: ${fromFile}`;
    refresh();
  }

  function reject(message) {
    show(photoError, message);
  }

  async function send(event) {
    event.preventDefault();
    if (!file) return;
    show(problem, "");
    const data = new FormData();
    data.append("image", file, file.name);
    data.append("name", nameInput.value.trim());
    data.append("subject", subjectInput.value.trim());
    data.append("mode", mode);
    data.append("quality", quality);
    dialog.lock(true);
    submit.disabled = true;
    cancel.disabled = true;
    submit.replaceChildren(h("span", { class: "spinner" }), h("span", null, "Uploading"));
    try {
      const result = await api.create(data);
      dialog.lock(false);
      dialog.close("created");
      toast(`${result.job.title || "The avatar"} is queued. Follow its progress in the library.`, "success");
      onCreated(result);
    } catch (error) {
      dialog.lock(false);
      show(problem, error.message);
      cancel.disabled = false;
      submit.disabled = false;
      submit.replaceChildren(icon("bolt", 16), h("span", null, "Create avatar"));
    }
  }

  picker.addEventListener("change", () => { choose(picker.files[0]); picker.value = ""; });
  drop.addEventListener("click", () => picker.click());
  drop.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      picker.click();
    }
  });
  for (const name of ["dragenter", "dragover"]) {
    drop.addEventListener(name, (event) => { event.preventDefault(); drop.classList.add("over"); });
  }
  for (const name of ["dragleave", "drop"]) {
    drop.addEventListener(name, () => drop.classList.remove("over"));
  }
  drop.addEventListener("drop", (event) => {
    event.preventDefault();
    choose(event.dataTransfer.files[0]);
  });
  dialog.dialog.addEventListener("close", clearPhoto);

  dialog.body.append(picker, form);
  refresh();
  dialog.open();
  drop.focus();
  return dialog.closed;
}

// ---------------------------------------------------------------------------------------------------------------- settings
function describe(value, bands) {
  return bands.find(([limit]) => value < limit)[1];
}

function slider({ name, label, hint, bands }) {
  const [min, max] = RANGES[name];
  const readout = h("output", { class: "readout" });
  const input = h("input", { type: "range", min, max, step: 0.05, "aria-label": label });
  const paint = () => {
    const value = Number(input.value);
    readout.textContent = `${value.toFixed(2)} · ${describe(value, bands)}`;
    input.style.setProperty("--fill", `${((value - min) / (max - min)) * 100}%`);
  };
  input.addEventListener("input", paint);
  const el = h("div", { class: "field" }, h("span", { class: "slider-head" }, h("span", null, label), readout), input, h("small", null, hint));
  return { el, input, set(value) { input.value = value; paint(); }, get: () => Number(input.value) };
}

export function openSettings(avatar, { onSaved }) {
  const dialog = modal({ title: `${avatar.title} settings`, subtitle: "Fine-tune how this avatar looks and moves while it talks.", className: "wide" });
  const nameInput = h("input", { type: "text", class: "input", maxlength: 80, value: avatar.title, autocomplete: "off" });
  const subjectInput = h("textarea", { class: "input", rows: 2, maxlength: 600, value: avatar.subject || "" });
  const expression = slider({
    name: "expression", label: "Facial expression",
    hint: "Lower values give calmer lips, jaw and face. Raise it for a more animated face.",
    bands: [[0.45, "Very subtle"], [0.75, "Subtle"], [1.0, "Natural"], [Infinity, "Expressive"]],
  });
  const motion = slider({
    name: "head_motion", label: "Head movement",
    hint: "How much the head sways while the avatar speaks.",
    bands: [[0.2, "Still"], [0.65, "Gentle"], [Infinity, "Lively"]],
  });
  // With a speaking clip the head follows the generated body, so only a still photo has a head movement to set.
  const followsBody = (avatar.states || []).includes("speaking");
  expression.set(avatar.settings.expression);
  motion.set(avatar.settings.head_motion);
  const current = { expression: expression.get(), head_motion: motion.get() };  // as the sliders show them: they snap to their step
  const problem = formError();
  const reset = h("button", {
    class: "btn btn-ghost", type: "button", onClick: () => { expression.set(DEFAULTS.expression); if (!followsBody) motion.set(DEFAULTS.head_motion); },
  }, icon("refresh", 16), "Reset to defaults");
  const cancel = h("button", { class: "btn btn-ghost", type: "button", onClick: () => dialog.close("") }, "Cancel");
  const save = h("button", { class: "btn btn-primary", type: "submit" }, icon("check", 16), "Save changes");

  const form = h("form", {
    novalidate: true,
    async onSubmit(event) {
      event.preventDefault();
      show(problem, "");
      save.disabled = true;
      try {
        const title = nameInput.value.trim();
        const subject = subjectInput.value.trim();
        if (title !== avatar.title || subject !== (avatar.subject || "")) await api.rename(avatar.id, { title, subject });
        const changes = {};  // only what the person could see and moved; the renderer keeps the rest
        if (Math.abs(expression.get() - current.expression) > 1e-6) changes.expression = expression.get();
        if (!followsBody && Math.abs(motion.get() - current.head_motion) > 1e-6) changes.head_motion = motion.get();
        if (Object.keys(changes).length) {
          await api.saveSettings(avatar.id, changes);
          api.warm(avatar.id).catch(() => {});  // load the new look now so the next conversation starts at once
        }
        dialog.close("saved");
        toast("Saved. Changes to the face apply from the next conversation.", "success");
        onSaved();
      } catch (error) {
        show(problem, error.message);
        save.disabled = false;
      }
    },
  },
  h("label", { class: "field" }, h("span", null, "Name"), nameInput),
  h("label", { class: "field" }, h("span", null, "Description"), subjectInput, h("small", null, "In English. Used when motion is generated for this avatar.")),
  h("div", { class: "divider" }),
  expression.el,
  followsBody
    ? h("p", { class: "note" }, icon("info", 16), h("span", null, "The head follows this avatar's generated body motion, so only the facial expression can be adjusted."))
    : motion.el,
  h("p", { class: "note" }, icon("info", 16), h("span", null, "A conversation that is already running keeps its look. The next one loads the new settings, which takes about 20 seconds.")),
  problem,
  h("div", { class: "modal-foot split" }, reset, h("span", { class: "spacer" }), cancel, save));
  dialog.body.append(form);
  dialog.open();
  nameInput.focus();
  return dialog.closed;
}

// ---------------------------------------------------------------------------------------------------------------- adding motion
export function openUpgrade(avatar, { onQueued }) {
  const dialog = modal({ title: "Add full motion", subtitle: `Generate idle and speaking motion for ${avatar.title}.`, className: "narrow" });
  let quality = "hd";
  const estimate = h("span", null);
  const problem = formError();
  const paint = () => { estimate.textContent = `Takes ${formatDuration(ESTIMATE_S[quality])}. ${avatar.title} stays available meanwhile.`; };
  const cancel = h("button", { class: "btn btn-ghost", type: "button", onClick: () => dialog.close("") }, "Cancel");
  const start = h("button", { class: "btn btn-primary", type: "submit" }, icon("bolt", 16), "Generate motion");
  const form = h("form", {
    async onSubmit(event) {
      event.preventDefault();
      show(problem, "");
      start.disabled = true;
      try {
        const result = await api.upgrade(avatar.id, quality);
        dialog.close("queued");
        toast(`Motion for ${avatar.title} is queued.`, "success");
        onQueued(result);
      } catch (error) {
        show(problem, error.message);
        start.disabled = false;
      }
    },
  },
  h("div", { class: "field" }, h("span", null, "Quality"),
    segmented("upgrade-quality", "Quality", [{ value: "hd", label: "HD" }, { value: "standard", label: "Standard" }], quality, (value) => { quality = value; paint(); })),
  h("p", { class: "note" }, icon("clock", 16), estimate),
  h("p", { class: "note warn" }, icon("alert", 16), h("span", null, "Making motion uses the same GPU as live conversations, so they may respond more slowly until it finishes.")),
  problem,
  h("div", { class: "modal-foot" }, cancel, start));
  paint();
  dialog.body.append(form);
  dialog.open();
  return dialog.closed;
}

// ---------------------------------------------------------------------------------------------------------------- deleting
/** Asks, then deletes. Resolves true when the avatar is gone. */
export async function deleteAvatar(avatar) {
  const sure = await confirmDialog({
    title: `Delete ${avatar.title}?`,
    message: "This permanently deletes the avatar and its generated motion clips. It cannot be undone.",
    confirmLabel: "Delete avatar",
    danger: true,
  });
  if (!sure) return false;
  try {
    await api.remove(avatar.id);
    toast(`${avatar.title} was deleted.`, "success");
    return true;
  } catch (error) {
    toast(error.message, "error", 8000);
    return false;
  }
}
