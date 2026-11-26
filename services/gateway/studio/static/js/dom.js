// Helpers for building the page. Text always goes in as text, never as HTML, because avatar names and descriptions are typed by users.

const PROPERTIES = new Set(["value", "checked", "disabled", "hidden", "selected", "open", "required", "readOnly", "multiple"]);

/** h("button", {class: "btn", onClick: fn, "aria-label": "Close"}, "Text", otherNode) */
export function h(tag, props, ...children) {
  const el = document.createElement(tag);
  for (const [name, value] of Object.entries(props || {})) {
    if (value === undefined || value === null || value === false) continue;
    if (name === "class") el.className = value;
    else if (name === "dataset") Object.assign(el.dataset, value);
    else if (name.startsWith("on") && typeof value === "function") el.addEventListener(name.slice(2).toLowerCase(), value);
    else if (PROPERTIES.has(name)) el[name] = value;
    else el.setAttribute(name, value === true ? "" : String(value));
  }
  return add(el, children);
}

export function add(parent, children) {
  for (const child of children.flat(Infinity)) {
    if (child === null || child === undefined || child === false) continue;
    parent.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return parent;
}

export function setText(el, text) {
  if (el.textContent !== text) el.textContent = text;
}

export function setVars(el, vars) {
  for (const [name, value] of Object.entries(vars)) el.style.setProperty(name, String(value));
}

/** Keep `parent`'s children in step with `items` without rebuilding them: new items are created, known ones updated in place, gone ones removed. */
export function reconcile(parent, items, key, create, update) {
  const known = new Map();
  for (const child of parent.children) if (child.dataset.key) known.set(child.dataset.key, child);
  let previous = null;
  const used = new Set();
  for (const item of items) {
    const id = key(item);
    let el = known.get(id);
    if (!el) {
      el = create(item);
      el.dataset.key = id;
    }
    update(el, item);
    const wanted = previous ? previous.nextSibling : parent.firstChild;
    if (el !== wanted) parent.insertBefore(el, wanted);
    previous = el;
    used.add(id);
  }
  for (const [id, el] of known) if (!used.has(id)) el.remove();
}

export function formatBytes(n) {
  if (!Number.isFinite(n) || n < 0) return "";
  if (n >= 1e9) return `${(n / 1e9).toFixed(n >= 1e10 ? 0 : 1)} GB`;
  if (n >= 1e6) return `${Math.round(n / 1e6)} MB`;
  return `${Math.max(1, Math.round(n / 1e3))} KB`;
}

export function formatDate(iso) {
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return "";
  return date.toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" });
}

export function formatDuration(seconds) {
  if (!Number.isFinite(seconds)) return "";
  if (seconds < 45) return "under a minute";
  const minutes = Math.round(seconds / 60);
  if (minutes < 60) return `about ${minutes} min`;
  const hours = Math.floor(minutes / 60);
  const rest = minutes % 60;
  return rest ? `about ${hours} h ${rest} min` : `about ${hours} h`;
}

export function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

export async function copyText(text) {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    // Clipboard access needs a secure page; fall back to selecting a hidden field.
  }
  const field = h("textarea", { class: "offscreen", readOnly: true, "aria-hidden": "true", value: text });
  document.body.append(field);
  field.select();
  let copied = false;
  try {
    copied = document.execCommand("copy");
  } catch {
    copied = false;
  }
  field.remove();
  return copied;
}
