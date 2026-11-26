// The library page: a card for every avatar, the cards of avatars that are still being made, and the state of the services behind them.
import { api } from "./api.js";
import { ACTIVE, createAvatarCard, createJobCard, updateAvatarCard, updateJobCard } from "./cards.js";
import { deleteAvatar, openNewAvatar, openSettings, openUpgrade } from "./dialogs.js";
import { copyText, formatBytes, h, reconcile, setText } from "./dom.js";
import { icon } from "./icons.js";
import { rendererStatus } from "./topbar.js";
import { confirmDialog, toast } from "./ui.js";

const JOBS_BUSY_MS = 2000;
const JOBS_IDLE_MS = 10000;
const AVATARS_MS = 10000;
const SKELETONS = 4;

/** Slower polling for a page nobody is looking at. */
const every = (ms) => (document.hidden ? ms * 4 : ms);

export function libraryPage({ watcher, actions }) {
  const state = { avatars: null, jobs: [], query: "", signature: null, problem: "" };
  let health = watcher.latest();
  let jobsTimer = 0;
  let avatarsTimer = 0;
  let jobsTicket = 0;
  let avatarsTicket = 0;
  const working = new Set();  // jobs with a request under way, so a double click does nothing twice

  // ------------------------------------------------------------------------------------------------------------ top bar
  const search = h("input", { type: "search", class: "search-input", placeholder: "Search avatars", "aria-label": "Search avatars", autocomplete: "off", spellcheck: "false" });
  actions.append(
    h("label", { class: "search" }, icon("search", 16), search, h("kbd", { "aria-hidden": "true" }, "/")),
    h("button", { class: "btn btn-primary", type: "button", onClick: newAvatar }, icon("plus", 16), h("span", null, "New avatar")));

  // ------------------------------------------------------------------------------------------------------------ page
  const strip = h("div", { class: "strip", role: "list", "aria-label": "Services" });
  const notices = h("div", { class: "notices" });
  const jobsGrid = h("div", { class: "grid" });
  const jobsCount = h("span", { class: "count" });
  const jobsSection = h("section", { class: "section", "aria-label": "Avatars being made", hidden: true },
    h("h2", { class: "section-title" }, h("span", null, "In progress"), jobsCount), jobsGrid);
  const avatarsGrid = h("div", { class: "grid" });
  const avatarsCount = h("span", { class: "count" });
  const skeletons = h("div", { class: "grid", "aria-hidden": "true" },
    Array.from({ length: SKELETONS }, () => h("div", { class: "card skeleton" }, h("div", { class: "thumb" }), h("div", { class: "meta" }, h("i"), h("i"), h("i")))));
  const empty = h("div", { class: "empty glass", hidden: true });
  const avatarsSection = h("section", { class: "section", "aria-label": "Avatars" },
    h("h2", { class: "section-title" }, h("span", null, "Avatars"), avatarsCount), skeletons, avatarsGrid, empty);

  const el = h("main", { class: "page library", id: "main" },
    h("section", { class: "hero" },
      h("div", { class: "hero-text" },
        h("p", { class: "eyebrow" }, h("i", { class: "dot" }), "Live avatar studio"),
        h("h1", null, "Your avatars"),
        h("p", { class: "lede" }, "Turn one photo into a live, talking avatar. Open an avatar to start a conversation.")),
      strip),
    notices, jobsSection, avatarsSection);

  // ------------------------------------------------------------------------------------------------------------ what the cards do
  const avatarActions = {
    async copy(avatar) {
      const copied = await copyText(`${location.origin}/a/${encodeURIComponent(avatar.id)}`);
      toast(copied ? "Link copied." : "The link could not be copied. Copy it from the address bar of the avatar's page instead.", copied ? "success" : "error");
    },
    settings: (avatar) => openSettings(avatar, { onSaved: loadAvatars }),
    upgrade: (avatar) => openUpgrade(avatar, { onQueued: queued }),
    async remove(avatar) {
      if (!(await deleteAvatar(avatar))) return;
      state.avatars = (state.avatars || []).filter((a) => a.id !== avatar.id);
      render();
      loadAvatars();
      loadJobs();
    },
  };

  /** Runs one job action at a time per job and shows what went wrong in English. */
  async function onJob(job, action, success) {
    if (working.has(job.id)) return;
    working.add(job.id);
    try {
      await action();
      if (success) toast(success, "success");
    } catch (error) {
      toast(error.message, "error", 8000);
    } finally {
      working.delete(job.id);
    }
    loadJobs();
    loadAvatars();
  }

  const jobActions = {
    async cancel(job) {
      const sure = await confirmDialog({
        title: `Cancel ${job.title || "this avatar"}?`,
        message: "The work done so far is lost. You can start again at any time.",
        confirmLabel: "Cancel job",
        cancelLabel: "Keep going",
        danger: true,
      });
      if (sure) onJob(job, () => api.cancelJob(job.id), "Cancelled.");
    },
    retry: (job, quality) => onJob(job, () => api.retryJob(job.id, quality), `${job.title || "The avatar"} is queued again.`),
    discard: (job) => onJob(job, () => api.removeJob(job.id)),
  };

  function newAvatar() {
    health = watcher.latest() ?? health;
    if (health && health.worker.state === "offline") {
      toast("The motion generator is not running, so new avatars cannot be made right now.", "error");
    } else if (health && health.disk.full) {
      toast("The server is almost out of disk space. Delete an avatar you no longer need and try again.", "error", 8000);
    } else {
      openNewAvatar({ limits: health ? health.limits : {}, onCreated: queued });
    }
  }

  /** A job was just queued: show it at once instead of waiting for the next refresh. */
  function queued(result) {
    if (result && result.job) {
      state.jobs = [result.job, ...state.jobs.filter((j) => j.id !== result.job.id)];
      renderJobs();
    }
    loadJobs();
    watcher.refresh();
  }

  // ------------------------------------------------------------------------------------------------------------ loading
  function note(problem) {
    if (state.problem === problem) return;
    state.problem = problem;
    renderNotices();
  }

  async function loadAvatars() {
    clearTimeout(avatarsTimer);
    const ticket = ++avatarsTicket;
    try {
      const data = await api.avatars();
      if (ticket !== avatarsTicket) return;
      state.avatars = data.avatars;
      note("");
      render();
    } catch (error) {
      if (ticket !== avatarsTicket) return;
      note(error.message);
      if (state.avatars === null) render();
    }
    avatarsTimer = setTimeout(loadAvatars, every(AVATARS_MS));
  }

  async function loadJobs() {
    clearTimeout(jobsTimer);
    const ticket = ++jobsTicket;
    let delay = JOBS_IDLE_MS;
    try {
      const data = await api.jobs();
      if (ticket !== jobsTicket) return;
      const signature = data.jobs.map((j) => `${j.id}:${j.status}`).sort().join("|");
      const changed = state.signature !== null && signature !== state.signature;
      state.signature = signature;
      state.jobs = data.jobs;
      if (changed) await loadAvatars();  // a finished avatar appears as a card in the same moment its job card goes
      renderJobs();
      renderNotices();
      if (data.jobs.some((j) => ACTIVE.has(j.status))) delay = JOBS_BUSY_MS;
    } catch {
      if (ticket !== jobsTicket) return;
    }
    jobsTimer = setTimeout(loadJobs, every(delay));
  }

  // ------------------------------------------------------------------------------------------------------------ drawing
  function renderStrip() {
    const renderer = rendererStatus(health);
    const tiles = [{ key: "renderer", icon: "cpu", label: "Renderer", value: health === undefined ? "Checking" : renderer.value, kind: health === undefined ? "wait" : renderer.kind }];
    if (health) {
      const { llm, worker, disk } = health;
      tiles.push(
        { key: "llm", icon: "spark", label: "Language model", value: llm.state === "ready" ? "Ready" : "Offline", kind: llm.state === "ready" ? "ok" : "bad" },
        {
          key: "worker", icon: "film", label: "Motion generator",
          value: worker.state !== "ready" ? "Offline" : worker.active ? `${worker.active} running or waiting` : "Idle",
          kind: worker.state !== "ready" ? "bad" : "ok",
        },
        { key: "disk", icon: "disk", label: "Free disk space", value: formatBytes(disk.free_bytes), kind: disk.full ? "bad" : disk.low ? "warn" : "ok" });
    }
    reconcile(strip, tiles, (t) => t.key,
      (t) => {
        const value = h("span", { class: "tile-value" });
        const tile = h("div", { class: "tile", role: "listitem" },
          h("span", { class: "tile-icon" }, icon(t.icon, 18)),
          h("span", { class: "tile-text" }, h("span", { class: "tile-label" }, t.label), value));
        tile.value = value;
        return tile;
      },
      (tile, t) => {
        tile.className = `tile ${t.kind}`;
        setText(tile.value, t.value);
      });
    renderNotices();
  }

  function renderNotices() {
    const items = [];
    if (state.problem) items.push({ key: "problem", kind: "bad", text: `${state.problem} Trying again automatically.` });
    if (health && health.disk.full) items.push({ key: "disk", kind: "bad", text: `The server is almost out of disk space (${formatBytes(health.disk.free_bytes)} free). New avatars cannot be made until you delete one you no longer need.` });
    else if (health && health.disk.low) items.push({ key: "disk", kind: "warn", text: `Disk space is running low (${formatBytes(health.disk.free_bytes)} free). Each new avatar needs about 1 GB.` });
    if (health && health.worker.state === "offline") items.push({ key: "worker", kind: "warn", text: "The motion generator is not running, so new avatars cannot be made right now." });
    reconcile(notices, items, (n) => n.key,
      () => h("div", { class: "banner", role: "status" }, icon("alert", 18), h("span")),
      (banner, n) => {
        banner.className = `banner ${n.kind}`;
        setText(banner.lastChild, n.text);
      });
  }

  function render() {
    const first = state.avatars === null;
    skeletons.hidden = !first || Boolean(state.problem);
    const query = state.query.trim().toLowerCase();
    const all = state.avatars || [];
    const shown = query ? all.filter((a) => `${a.title} ${a.subject || ""} ${a.id}`.toLowerCase().includes(query)) : all;
    reconcile(avatarsGrid, shown, (a) => a.id, () => createAvatarCard(avatarActions), updateAvatarCard);
    avatarsGrid.hidden = shown.length === 0;
    setText(avatarsCount, first ? "" : query ? `${shown.length} of ${all.length}` : String(all.length));
    drawEmpty(first, all.length, shown.length, query);
  }

  function drawEmpty(first, total, shown, query) {
    const nothingMade = total === 0 && !visibleJobs().length;
    empty.hidden = first || shown > 0 || (total === 0 && !nothingMade);
    if (empty.hidden) return;
    if (query && total > 0) {
      empty.replaceChildren(
        h("span", { class: "empty-icon" }, icon("search", 34)),
        h("h3", null, "No avatars match your search"),
        h("p", null, "Check the spelling or clear the search to see every avatar."),
        h("button", { class: "btn btn-ghost", type: "button", onClick: () => { search.value = ""; state.query = ""; render(); } }, "Clear search"));
    } else {
      empty.replaceChildren(
        h("span", { class: "empty-icon" }, icon("person", 34)),
        h("h3", null, "No avatars yet"),
        h("p", null, "Upload one photo, full-body or portrait, and the Studio makes a live avatar you can talk to."),
        h("button", { class: "btn btn-primary", type: "button", onClick: newAvatar }, icon("plus", 16), h("span", null, "Create your first avatar")));
    }
  }

  const visibleJobs = () => state.jobs.filter((j) => j.status !== "ready");  // a finished job is already a card of its own

  function renderJobs() {
    const jobs = visibleJobs().sort((a, b) => {
      const activeA = ACTIVE.has(a.status);
      const activeB = ACTIVE.has(b.status);
      if (activeA !== activeB) return activeA ? -1 : 1;
      return activeA ? a.created - b.created : b.created - a.created;
    });
    const known = new Map((state.avatars || []).map((a) => [a.id, a]));
    reconcile(jobsGrid, jobs, (j) => j.id, () => createJobCard(jobActions), (card, job) => {
      const avatar = known.get(job.avatar);
      updateJobCard(card, job, { upgrade: job.mode === "full" && Boolean(avatar) && avatar.mode !== "full" });
    });
    jobsSection.hidden = jobs.length === 0;
    setText(jobsCount, String(jobs.length));
    if (state.avatars !== null) drawEmpty(false, state.avatars.length, avatarsGrid.children.length, state.query.trim().toLowerCase());
  }

  // ------------------------------------------------------------------------------------------------------------ wiring
  search.addEventListener("input", () => { state.query = search.value; render(); });
  search.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && search.value) {
      search.value = "";
      state.query = "";
      render();
    }
  });
  const onKey = (event) => {
    if (event.key !== "/" || event.ctrlKey || event.metaKey || event.altKey) return;
    const target = event.target;
    if (target instanceof HTMLElement && (target.isContentEditable || /^(input|textarea|select)$/i.test(target.tagName))) return;
    if (document.querySelector("dialog[open]")) return;
    event.preventDefault();
    search.focus();
  };
  const onVisible = () => {
    if (document.hidden) return;
    loadJobs();
    loadAvatars();
  };
  document.addEventListener("keydown", onKey);
  document.addEventListener("visibilitychange", onVisible);
  const unsubscribe = watcher.subscribe((latest) => { health = latest; renderStrip(); });
  renderStrip();
  render();
  loadJobs();
  loadAvatars();

  return {
    el,
    destroy() {
      clearTimeout(jobsTimer);
      clearTimeout(avatarsTimer);
      jobsTicket++;
      avatarsTicket++;
      unsubscribe();
      document.removeEventListener("keydown", onKey);
      document.removeEventListener("visibilitychange", onVisible);
    },
  };
}
