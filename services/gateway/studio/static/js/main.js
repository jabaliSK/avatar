// Entry point: puts the shared top bar and the page for the address on the screen, and starts the animated backdrop.
import { startBackground } from "./background.js";
import { h } from "./dom.js";
import { libraryPage } from "./library.js";
import { stagePage } from "./stage.js";
import { topbar, watchHealth } from "./topbar.js";

const app = document.getElementById("app");

/** The avatar id in an address like /a/neri, or null for the library. A malformed address is shown as an avatar that does not exist. */
function avatarIdFromAddress() {
  const match = location.pathname.match(/^\/a\/([^/]+)\/?$/);
  if (!match) return null;
  try {
    return decodeURIComponent(match[1]);
  } catch {
    return match[1];
  }
}

function boot() {
  const avatarId = avatarIdFromAddress();
  document.body.dataset.page = avatarId === null ? "library" : "stage";
  startBackground(document.getElementById("bg"), { density: avatarId === null ? 1 : 0.55 });
  const watcher = watchHealth();
  const bar = topbar(watcher);
  const page = avatarId === null ? libraryPage({ watcher, actions: bar.actions }) : stagePage({ watcher, actions: bar.actions, avatarId });
  app.append(h("a", { class: "skip", href: "#main" }, "Skip to content"), bar.el, page.el);
}

try {
  boot();
} catch (error) {
  console.error(error);
  app.replaceChildren(h("main", { class: "page fatal", id: "main" },
    h("div", { class: "empty glass" },
      h("h1", null, "Avatar Studio could not start"),
      h("p", null, "Reload the page. If this keeps happening, ask an administrator to check the Studio."))));
}
