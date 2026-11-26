// The two kinds of card in the library: an avatar, and a job that is still making one. Each has a create function that builds the card once
// and an update function that brings it up to date, so the 2-second progress refresh changes text and bars without rebuilding anything.
import { formatBytes, formatDate, formatDuration, h, reconcile, setText, setVars } from "./dom.js";
import { icon } from "./icons.js";

export const ACTIVE = new Set(["queued", "generating", "waiting", "registering"]);

const MODE_BADGES = {
  full: { text: "Full motion", kind: "full" },
  partial: { text: "Partial motion", kind: "partial" },
  quick: { text: "Quick preview", kind: "quick" },
};

const JOB_BADGES = {
  queued: { text: "Queued", kind: "wait" },
  waiting: { text: "Waiting", kind: "wait" },
  generating: { text: "Generating", kind: "work" },
  registering: { text: "Finishing", kind: "work" },
  failed: { text: "Failed", kind: "failed" },
  cancelled: { text: "Cancelled", kind: "quiet" },
};

function setAttr(el, name, value) {
  if (el.getAttribute(name) !== value) el.setAttribute(name, value);
}

function button(label, iconName, extra = "") {
  return h("button", { class: `btn btn-small ${extra}`.trim(), type: "button" }, icon(iconName, 15), h("span", null, label));
}

// ---------------------------------------------------------------------------------------------------------------- avatars
export function createAvatarCard(actions) {
  const img = h("img", { class: "thumb-img", alt: "", loading: "lazy", decoding: "async" });
  const badge = h("span", { class: "badge" });
  const onStage = h("span", { class: "onstage", hidden: true }, h("i", { class: "dot" }), "On stage");
  const thumb = h("a", { class: "thumb", tabindex: -1, "aria-hidden": "true" }, img, h("i", { class: "corners" }), badge, onStage);
  const name = h("h3", { class: "name" });
  const subject = h("p", { class: "subject" });
  const size = h("span");
  const date = h("span");
  const upgrade = button("Add full motion", "upgrade", "btn-accent");
  const open = h("a", { class: "btn btn-small btn-primary" }, icon("open", 15), h("span", null, "Open"));
  const copy = button("Copy link", "link");
  const settings = button("Settings", "sliders");
  const remove = button("Delete", "trash", "btn-danger");
  const el = h("article", { class: "card avatar" },
    thumb,
    h("div", { class: "meta" }, name, subject, h("p", { class: "facts" }, size, date), upgrade),
    h("div", { class: "actions" }, open, copy, settings, remove));
  el.refs = { img, badge, onStage, thumb, name, subject, size, date, upgrade, open };
  el.avatar = null;
  copy.addEventListener("click", () => actions.copy(el.avatar));
  settings.addEventListener("click", () => actions.settings(el.avatar));
  remove.addEventListener("click", () => actions.remove(el.avatar));
  upgrade.addEventListener("click", () => actions.upgrade(el.avatar));
  tilt(el);
  return el;
}

export function updateAvatarCard(el, avatar) {
  el.avatar = avatar;
  const r = el.refs;
  const href = `/a/${encodeURIComponent(avatar.id)}`;
  setAttr(r.thumb, "href", href);
  setAttr(r.open, "href", href);
  setAttr(r.open, "aria-label", `Open ${avatar.title}`);
  setAttr(r.img, "src", avatar.thumbnail);
  setText(r.name, avatar.title);
  setText(r.subject, avatar.subject || "No description yet");
  r.subject.classList.toggle("none", !avatar.subject);
  setText(r.size, formatBytes(avatar.size_bytes));
  setText(r.date, formatDate(avatar.created));
  const badge = MODE_BADGES[avatar.mode] || MODE_BADGES.quick;
  setText(r.badge, badge.text);
  r.badge.className = `badge ${badge.kind}`;
  r.onStage.hidden = !avatar.selected;
  r.upgrade.hidden = !avatar.upgradable;
}

// ---------------------------------------------------------------------------------------------------------------- jobs
export function createJobCard(actions) {
  const img = h("img", { class: "thumb-img", alt: "", loading: "lazy", decoding: "async" });
  const badge = h("span", { class: "badge" });
  const thumb = h("div", { class: "thumb" }, img, h("i", { class: "corners" }), badge);
  const name = h("h3", { class: "name" });
  const phase = h("p", { class: "phase" });
  const fill = h("i");
  const bar = h("div", { class: "bar", role: "progressbar", "aria-valuemin": 0, "aria-valuemax": 100 }, fill);
  const percent = h("span");
  const left = h("span");
  const steps = h("ol", { class: "steps" });
  const failure = h("div", { class: "failure", role: "alert", hidden: true });
  const cancel = button("Cancel", "close");
  const retry = button("Try again", "refresh", "btn-primary");
  const retryStandard = button("Try again at Standard quality", "bolt");
  const discard = button("Remove", "trash", "btn-danger");
  const el = h("article", { class: "card job" },
    thumb,
    h("div", { class: "meta" }, name, phase, bar, h("p", { class: "facts" }, percent, left), steps, failure),
    h("div", { class: "actions single" }, cancel, retry, retryStandard, discard));
  el.refs = { img, badge, name, phase, bar, fill, percent, left, steps, failure, cancel, retry, retryStandard, discard };
  el.job = null;
  cancel.addEventListener("click", () => actions.cancel(el.job));
  retry.addEventListener("click", () => actions.retry(el.job));
  retryStandard.addEventListener("click", () => actions.retry(el.job, "standard"));
  discard.addEventListener("click", () => actions.discard(el.job));
  return el;
}

export function updateJobCard(el, job, { upgrade }) {
  el.job = job;
  const r = el.refs;
  const active = ACTIVE.has(job.status);
  setAttr(r.img, "src", job.thumbnail);
  const title = job.title || job.avatar;
  setText(r.name, upgrade ? `Adding full motion to ${title}` : title);
  const badge = job.status === "generating" && upgrade ? { text: "Adding motion", kind: "work" } : JOB_BADGES[job.status] || JOB_BADGES.queued;
  setText(r.badge, badge.text);
  r.badge.className = `badge ${badge.kind}`;
  setText(r.phase, job.status === "queued" && job.queue_ahead ? `Waiting in the queue: ${job.queue_ahead} ahead` : job.phase || "");
  r.bar.hidden = !active;
  r.bar.classList.toggle("live", job.status === "generating" || job.status === "registering");
  setVars(r.bar, { "--p": job.progress || 0 });
  setAttr(r.bar, "aria-valuenow", String(Math.round((job.progress || 0) * 100)));
  setAttr(r.bar, "aria-label", `${title}: ${job.phase || "in progress"}`);
  setText(r.percent, active ? `${Math.round((job.progress || 0) * 100)}%` : "");
  setText(r.left, active && job.eta_s ? `${formatDuration(job.eta_s)} left` : "");
  r.steps.hidden = !(job.stages && job.stages.length > 1);
  reconcile(r.steps, job.stages || [], (s) => s.key,
    (s) => h("li", null, h("span", { class: "mark" }), h("span", { class: "step-label" }), h("span", { class: "step-pct" })),
    (li, s) => {
      li.className = s.state;
      setText(li.children[1], s.label);
      setText(li.children[2], s.state === "running" && s.fraction > 0 ? `${Math.round(s.fraction * 100)}%` : "");
    });
  const failed = job.status === "failed";
  r.failure.hidden = !(failed && job.error);
  if (failed && job.error) {
    r.failure.replaceChildren(h("strong", null, job.error), job.hint ? h("span", null, job.hint) : null);
  }
  r.cancel.hidden = !active;
  r.retry.hidden = active;
  r.retryStandard.hidden = !(failed && job.mode === "full" && job.quality === "hd");
  r.discard.hidden = active;
}

// ---------------------------------------------------------------------------------------------------------------- effects
/** The card leans towards the pointer. Only on devices with a real pointer, and never when the person prefers reduced motion. */
function tilt(el) {
  if (!window.matchMedia("(hover: hover) and (prefers-reduced-motion: no-preference)").matches) return;
  el.addEventListener("pointermove", (event) => {
    const box = el.getBoundingClientRect();
    const x = (event.clientX - box.left) / box.width - 0.5;
    const y = (event.clientY - box.top) / box.height - 0.5;
    setVars(el, { "--tilt-x": `${(x * 5).toFixed(2)}deg`, "--tilt-y": `${(-y * 5).toFixed(2)}deg`, "--glow-x": `${((x + 0.5) * 100).toFixed(0)}%`, "--glow-y": `${((y + 0.5) * 100).toFixed(0)}%` });
  });
  el.addEventListener("pointerleave", () => setVars(el, { "--tilt-x": "0deg", "--tilt-y": "0deg" }));
}
