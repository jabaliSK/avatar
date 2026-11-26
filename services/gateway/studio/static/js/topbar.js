// The top bar shared by every page, and the service status it shows.
import { api } from "./api.js";
import { h, setText } from "./dom.js";

const HEALTH_INTERVAL_MS = 5000;

/** What to say about the renderer and how to colour it: `text` for the pill, `value` for a tile that already says "Renderer". */
export function rendererStatus(health) {
  if (!health) return { kind: "bad", value: "Unreachable", text: "Studio unreachable" };
  const renderer = health.renderer;
  if (!renderer || renderer.state === "offline") return { kind: "bad", value: "Offline", text: "Renderer offline" };
  if (renderer.state === "loading") return { kind: "warn", value: "Starting", text: "Renderer starting" };
  if (renderer.state === "error") return { kind: "bad", value: "Error", text: "Renderer error" };
  if (renderer.registering) return { kind: "warn", value: "Preparing an avatar", text: "Renderer preparing an avatar" };
  if (renderer.busy) return { kind: "ok", value: "In conversation", text: "Renderer in conversation" };
  return { kind: "ok", value: "Ready", text: "Renderer ready" };
}

/** Polls /api/health while the page is visible and tells every listener the latest result (null when the Studio cannot be reached). */
export function watchHealth() {
  const listeners = new Set();
  let latest;  // undefined until the first answer, then the health or null
  let timer = 0;
  let stopped = false;

  async function poll() {
    clearTimeout(timer);
    try {
      latest = await api.health();
    } catch {
      latest = null;
    }
    for (const listener of listeners) listener(latest);
    if (!stopped) timer = setTimeout(poll, document.hidden ? HEALTH_INTERVAL_MS * 3 : HEALTH_INTERVAL_MS);
  }

  const onVisible = () => { if (!document.hidden) poll(); };
  document.addEventListener("visibilitychange", onVisible);
  poll();
  return {
    latest: () => latest,
    refresh: poll,
    subscribe(listener) {
      listeners.add(listener);
      if (latest !== undefined) listener(latest);
      return () => listeners.delete(listener);
    },
    stop() {
      stopped = true;
      clearTimeout(timer);
      document.removeEventListener("visibilitychange", onVisible);
    },
  };
}

/** The bar itself. `actions` is where a page puts its own controls (search and "New avatar", or the conversation's links). */
export function topbar(watcher) {
  const dot = h("span", { class: "dot" });
  const label = h("span", null, "Checking the renderer");
  const pill = h("div", { class: "pill", role: "status" }, dot, label);
  const actions = h("div", { class: "topbar-actions" });
  const el = h("header", { class: "topbar" },
    h("a", { class: "brand", href: "/", "aria-label": "Avatar Studio, all avatars" },
      h("img", { src: "/ui/brand/mark.svg", alt: "", width: 36, height: 36 }),
      h("span", { class: "brand-text" }, h("span", { class: "brand-name" }, "Avatar Studio"), h("span", { class: "brand-sub" }, "Live avatars"))),
    pill,
    actions);
  watcher.subscribe(function show(health) {
    const status = rendererStatus(health);
    pill.className = `pill ${status.kind}`;
    setText(label, status.text);
  });
  return { el, actions };
}
