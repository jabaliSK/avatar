// The conversation page of one avatar: get the avatar ready in the renderer, then talk to it by voice or by typing.
import { api } from "./api.js";
import { openSettings } from "./dialogs.js";
import { copyText, h, reconcile, setText, setVars, sleep } from "./dom.js";
import { icon } from "./icons.js";
import { startConversation } from "./rtc.js";
import { toast } from "./ui.js";

const DEFAULT_SIZE = [704, 1280];
const POLL_MS = 1000;
const REWARM_MS = 3000;
const LOADING_LIMIT_MS = 120000;

const TURNS = { listening: "Listening", thinking: "Thinking", speaking: "Speaking" };
const MODES = { full: "Full motion", partial: "Partial motion", quick: "Quick preview" };

export function stagePage({ watcher, actions, avatarId }) {
  const state = { avatar: null, phase: "loading", turn: "listening", conversation: null, connected: false, muted: false, rewarm: false };
  let destroyed = false;
  let attempt = 0;  // each try to get ready or to start has a number; an older one stops as soon as a newer begins

  // ------------------------------------------------------------------------------------------------------------ top bar
  const settingsButton = h("button", { class: "btn btn-ghost btn-small", type: "button", disabled: true, onClick: openAvatarSettings }, icon("sliders", 15), h("span", null, "Settings"));
  const copyButton = h("button", { class: "btn btn-ghost btn-small", type: "button", onClick: copyLink }, icon("link", 15), h("span", null, "Copy link"));
  actions.append(h("a", { class: "btn btn-ghost btn-small", href: "/" }, icon("back", 15), h("span", null, "All avatars")), settingsButton, copyButton);

  // ------------------------------------------------------------------------------------------------------------ the picture
  const poster = h("img", { class: "poster", alt: "" });
  const video = h("video", { class: "video", playsinline: true, autoplay: true, "aria-label": "Live video of the avatar" });

  const spinner = h("span", { class: "spinner large", hidden: true });
  const overlayIcon = h("span", { class: "overlay-icon", hidden: true });
  const overlayTitle = h("h2", { class: "overlay-title" });
  const overlayText = h("p", { class: "overlay-text" });
  const overlayActions = h("div", { class: "overlay-actions" });
  const overlay = h("div", { class: "overlay" },
    h("div", { class: "overlay-card", role: "status" }, h("i", { class: "corners" }), overlayIcon, spinner, overlayTitle, overlayText, overlayActions));

  const liveBadge = h("span", { class: "live-badge" }, h("i", { class: "dot" }), "LIVE");
  const readout = h("span", { class: "hud-readout" });
  const chipText = h("span", null, TURNS.listening);
  const chip = h("span", { class: "chip listening" }, h("span", { class: "eq", "aria-hidden": "true" }, h("i"), h("i"), h("i"), h("i")), chipText);
  const hud = h("div", { class: "hud", hidden: true }, h("i", { class: "corners" }), h("div", { class: "hud-top" }, liveBadge, readout), h("div", { class: "hud-bottom" }, chip));
  const frame = h("div", { class: "frame", "data-phase": "loading" }, poster, video, hud, overlay);

  // ------------------------------------------------------------------------------------------------------------ the controls
  const micButton = h("button", { class: "ctl mic", type: "button", disabled: true, "aria-pressed": "false", onClick: toggleMute }, icon("mic", 20), h("span", null, "Mute"));
  const interruptButton = h("button", { class: "ctl", type: "button", disabled: true, onClick: () => state.conversation && state.conversation.interrupt() }, icon("stop", 20), h("span", null, "Interrupt"));
  const endButton = h("button", { class: "ctl end", type: "button", disabled: true, onClick: () => state.conversation && state.conversation.stop() }, icon("close", 20), h("span", null, "End"));
  const controls = h("div", { class: "controls", role: "group", "aria-label": "Conversation controls" }, micButton, interruptButton, endButton);

  // ------------------------------------------------------------------------------------------------------------ the text of the conversation
  const title = h("h1", { class: "avatar-title" });
  const subject = h("p", { class: "avatar-subject" });
  const mode = h("span", { class: "badge" });
  const about = h("section", { class: "about" },
    h("p", { class: "eyebrow" }, h("i", { class: "dot" }), "Now talking with"), title, subject, h("div", { class: "about-meta" }, mode));
  const hint = h("p", { class: "captions-empty" }, "The conversation appears here. Say hello, or type a message below.");
  const log = h("div", { class: "captions", role: "log", "aria-live": "polite", "aria-label": "Conversation" });
  const input = h("input", { class: "input", type: "text", maxlength: 1000, autocomplete: "off", placeholder: "Start the conversation first", "aria-label": "Message to the avatar", disabled: true });
  const send = h("button", { class: "btn btn-primary", type: "submit", disabled: true }, icon("send", 16), h("span", null, "Send"));
  const composer = h("form", { class: "composer", onSubmit: submitText }, input, send);
  const side = h("aside", { class: "side" }, about, h("section", { class: "talk" }, hint, log, composer));

  const el = h("main", { class: "page stage", id: "main" }, h("section", { class: "stage-main" }, frame, controls), side);

  // ------------------------------------------------------------------------------------------------------------ overlay
  /** Says what is going on in the middle of the picture. Text changes in place so a spinner is not restarted every second. */
  function showOverlay({ heading, text = "", busy = false, iconName = "", buttons = [] }) {
    overlay.hidden = false;
    setText(overlayTitle, heading);
    setText(overlayText, text);
    overlayText.hidden = !text;
    spinner.hidden = !busy;
    overlayIcon.hidden = !iconName;
    if (iconName && overlayIcon.dataset.icon !== iconName) {
      overlayIcon.dataset.icon = iconName;
      overlayIcon.replaceChildren(icon(iconName, 34));
    }
    const key = buttons.map((b) => b.label).join("|");
    if (overlayActions.dataset.key !== key) {
      overlayActions.dataset.key = key;
      overlayActions.replaceChildren(...buttons.map((b) => (b.href
        ? h("a", { class: `btn ${b.primary ? "btn-primary btn-large" : "btn-ghost"}`, href: b.href }, b.icon ? icon(b.icon, 18) : null, b.label)
        : h("button", { class: `btn ${b.primary ? "btn-primary btn-large" : "btn-ghost"}`, type: "button", onClick: b.onClick }, b.icon ? icon(b.icon, 18) : null, b.label))));
    }
  }

  function setPhase(phase) {
    state.phase = phase;
    frame.dataset.phase = phase;
    syncControls();
  }

  function fail(heading, text, retry) {
    setPhase("error");
    showOverlay({
      heading, text, iconName: "alert",
      buttons: [...(retry ? [{ label: "Try again", primary: true, icon: "refresh", onClick: retry }] : []), { label: "All avatars", href: "/", icon: "back" }],
    });
  }

  function syncControls() {
    const conversation = state.conversation;
    micButton.disabled = !conversation || !conversation.micAvailable;
    micButton.title = conversation && !conversation.micAvailable ? "No microphone is connected. Type your messages instead." : "";
    interruptButton.disabled = !conversation || state.turn === "listening";
    endButton.disabled = !conversation;
    const typing = Boolean(conversation) && state.connected;
    input.disabled = !typing;
    send.disabled = !typing;
    input.placeholder = typing ? "Type a message" : "Start the conversation first";
    settingsButton.disabled = !state.avatar;
  }

  function setTurn(turn) {
    state.turn = turn;
    chip.className = `chip ${turn}`;
    setText(chipText, TURNS[turn]);
    syncControls();
  }

  // ------------------------------------------------------------------------------------------------------------ getting ready
  async function load() {
    const mine = ++attempt;
    setPhase("loading");
    showOverlay({ heading: "Loading the avatar", busy: true });
    try {
      state.avatar = await api.avatar(avatarId);
    } catch (error) {
      if (mine !== attempt || destroyed) return;
      if (error.status === 404) fail("Avatar not found", "This avatar does not exist. It may have been deleted.");
      else fail("Cannot load this avatar", error.message, load);
      return;
    }
    if (mine !== attempt || destroyed) return;
    drawAvatar();
    prepare(false);
  }

  function drawAvatar() {
    const avatar = state.avatar;
    const [width, height] = avatar.frame_size && avatar.frame_size.length === 2 ? avatar.frame_size : DEFAULT_SIZE;
    setVars(frame, { "--ar": `${width} / ${height}`, "--ratio": width / height });
    if (poster.getAttribute("src") !== avatar.thumbnail) poster.setAttribute("src", avatar.thumbnail);
    setText(title, avatar.title);
    setText(subject, avatar.subject || "");
    subject.hidden = !avatar.subject;
    const badge = MODES[avatar.mode] || MODES.quick;
    setText(mode, badge);
    mode.className = `badge ${avatar.mode || "quick"}`;
    document.title = `${avatar.title} · Avatar Studio`;
    syncControls();
  }

  /** Selects this avatar for the next conversation and waits until the renderer has it in memory. */
  async function prepare(thenStart) {
    const mine = ++attempt;
    const avatar = state.avatar;
    const name = avatar.title;
    setPhase("preparing");
    showOverlay({ heading: `Preparing ${name}`, busy: true });
    try {
      await api.select(avatar.id);
    } catch (error) {
      if (mine === attempt && !destroyed) fail("Cannot select this avatar", error.message, () => prepare(thenStart));
      return;
    }
    let warmed = false;
    let lastWarm = 0;
    let loadingSince = 0;
    for (;;) {
      if (mine !== attempt || destroyed) return;
      let status;
      try {
        const due = !warmed || Date.now() - lastWarm >= REWARM_MS;
        if (due && (!warmed || state.rewarm)) {
          status = await api.warm(avatar.id);
          warmed = true;
          lastWarm = Date.now();
        } else {
          status = await api.status(avatar.id);
        }
      } catch (error) {
        if (mine !== attempt || destroyed) return;
        if (error.status === 404) return fail("Avatar not found", "This avatar does not exist. It may have been deleted.");
        showOverlay({ heading: `Preparing ${name}`, text: `${error.message} Trying again…`, busy: true });
        await sleep(REWARM_MS);
        continue;
      }
      if (mine !== attempt || destroyed) return;
      if (status.loaded) {
        ready(thenStart);
        return;
      }
      if (status.renderer === "error") return fail("The renderer has a problem", "Ask an administrator to check the renderer, then try again.", () => prepare(thenStart));
      // While busy the renderer cannot have restarted the load, so an error it still shows is an old one.
      if (status.error && !status.loading && !status.busy) return fail(`${name} could not be prepared`, "Try again. If it keeps failing, ask an administrator to check the avatar.", () => prepare(thenStart));
      state.rewarm = false;
      if (status.renderer !== "ready") {
        state.rewarm = true;
        showOverlay({ heading: "The renderer is starting", text: "This can take a minute after a restart.", busy: true });
      } else if (status.loading) {
        loadingSince = loadingSince || Date.now();
        if (Date.now() - loadingSince > LOADING_LIMIT_MS) return fail(`${name} is taking too long`, "Try again in a moment.", () => prepare(thenStart));
        showOverlay({ heading: `Preparing ${name}`, text: "Loading the face model. This takes up to half a minute.", busy: true });
      } else if (status.busy) {
        state.rewarm = true;
        const health = watcher.latest();
        const registering = health && health.renderer && health.renderer.registering;
        showOverlay({
          heading: registering ? "The renderer is preparing another avatar" : "Another conversation is in progress",
          text: registering ? `${name} will be ready right after.` : `${name} will be ready as soon as it ends.`,
          busy: true,
        });
      } else {
        state.rewarm = true;
      }
      await sleep(status.loading ? POLL_MS : state.rewarm ? 1500 : POLL_MS);
    }
  }

  function ready(thenStart) {
    setPhase("ready");
    if (thenStart) {
      start();
      return;
    }
    showOverlay({
      heading: `Talk with ${state.avatar.title}`,
      text: "Your browser asks to use the microphone. Allow it to speak, or type your messages instead.",
      iconName: "wave",
      buttons: [{ label: "Start conversation", primary: true, icon: "mic", onClick: start }],
    });
  }

  // ------------------------------------------------------------------------------------------------------------ talking
  async function start() {
    if (state.conversation || (state.phase !== "ready" && state.phase !== "ended")) return;
    const mine = ++attempt;
    setPhase("connecting");
    showOverlay({ heading: "Connecting", text: "Your browser may ask to use the microphone.", busy: true });
    try {
      await api.select(state.avatar.id);  // another tab may have chosen a different avatar meanwhile
      const conversation = await startConversation(handlers);
      if (mine !== attempt || destroyed) {
        conversation.stop();
        return;
      }
      state.conversation = conversation;
      state.muted = false;
      state.connected = false;
      setMuted(false);
      hud.hidden = false;
      syncControls();
      if (conversation.notice) toast(conversation.notice, "warn", 10000);
    } catch (error) {
      if (mine !== attempt || destroyed) return;
      setPhase("ready");
      showOverlay({
        heading: "The conversation could not start", text: error.message, iconName: "alert",
        buttons: [{ label: "Try again", primary: true, icon: "refresh", onClick: start }, { label: "All avatars", href: "/", icon: "back" }],
      });
    }
  }

  const handlers = {
    onStream(stream) {
      if (video.srcObject !== stream) video.srcObject = stream;
      video.play().catch(() => {
        showOverlay({ heading: "Tap to start the video", text: "Your browser blocked automatic playback.", iconName: "wave", buttons: [{ label: "Play", primary: true, icon: "open", onClick: () => { video.play(); overlay.hidden = true; } }] });
      });
    },
    onStatus(status) {
      state.connected = status === "connected";
      setText(readout, status === "connected" ? readout.textContent : "Reconnecting…");
      syncControls();
      if (state.connected) input.focus({ preventScroll: true });
    },
    onTurn: setTurn,
    onCaptions: drawCaptions,
    onLevel(level) {
      micButton.style.setProperty("--level", level.toFixed(2));
    },
    onStats(stats) {
      const parts = [];
      if (stats.width && stats.height) parts.push(`${stats.width}×${stats.height}`);
      if (Number.isFinite(stats.fps)) parts.push(`${stats.fps} fps`);
      if (Number.isFinite(stats.rttMs)) parts.push(`${stats.rttMs} ms`);
      if (stats.route) parts.push(stats.route === "relay" ? "relayed" : "direct");
      setText(readout, parts.join(" · "));
    },
    onEnded(message) {
      state.conversation = null;
      state.connected = false;
      video.pause();
      video.srcObject = null;
      frame.classList.remove("has-video");
      hud.hidden = true;
      setText(readout, "");
      micButton.style.setProperty("--level", "0");
      setMuted(false);
      setTurn("listening");
      setPhase("ended");
      showOverlay({
        heading: message ? "The conversation ended" : "Conversation ended",
        text: message || "You can start another one at any time.",
        iconName: message ? "alert" : "check",
        buttons: [{ label: "Start a new conversation", primary: true, icon: "mic", onClick: () => prepare(true) }, { label: "All avatars", href: "/", icon: "back" }],
      });
    },
  };

  video.addEventListener("playing", () => {
    if (!state.conversation && state.phase !== "connecting") return;
    frame.classList.add("has-video");
    if (state.phase === "connecting") {
      setPhase("live");
      overlay.hidden = true;
    }
  });

  function setMuted(value) {
    state.muted = value;
    if (state.conversation) state.conversation.setMuted(value);
    micButton.setAttribute("aria-pressed", String(value));
    micButton.classList.toggle("off", value);
    micButton.replaceChildren(icon(value ? "micoff" : "mic", 20), h("span", null, value ? "Unmute" : "Mute"));
  }

  function toggleMute() {
    setMuted(!state.muted);
  }

  function submitText(event) {
    event.preventDefault();
    const text = input.value.trim();
    if (!text) return;
    if (state.conversation && state.conversation.sendText(text)) {
      input.value = "";
    } else {
      toast("Not connected yet. Try again in a moment.", "warn");
    }
  }

  // ------------------------------------------------------------------------------------------------------------ captions
  function drawCaptions(records) {
    const nearBottom = log.scrollHeight - log.scrollTop - log.clientHeight < 48;
    hint.hidden = records.length > 0;
    const name = state.avatar ? state.avatar.title : "Avatar";
    reconcile(log, records, (record) => record.id,
      (record) => {
        const who = h("span", { class: "who" });
        const text = h("p", { class: "text" });
        const message = h("div", { class: "msg" }, who, text);
        message.refs = { who, text };
        return message;
      },
      (message, record) => {
        message.className = `msg ${record.role}${record.cancelled ? " cancelled" : ""}${!record.final && record.role === "avatar" && !record.cancelled ? " typing" : ""}`;
        setText(message.refs.who, record.role === "human" ? "You" : name);
        setText(message.refs.text, record.text);
      });
    if (nearBottom) log.scrollTop = log.scrollHeight;
  }

  // ------------------------------------------------------------------------------------------------------------ the rest
  async function copyLink() {
    const copied = await copyText(location.href);
    toast(copied ? "Link copied." : "The link could not be copied. Copy it from the address bar instead.", copied ? "success" : "error");
  }

  function openAvatarSettings() {
    if (!state.avatar) return;
    openSettings(state.avatar, {
      async onSaved() {
        try {
          state.avatar = await api.avatar(avatarId);
          drawAvatar();
        } catch {
          // The page keeps showing what it had; the change still applies from the next conversation.
        }
      },
    });
  }

  const onHide = () => { if (state.conversation) state.conversation.stop(); };
  window.addEventListener("pagehide", onHide);
  load();

  return {
    el,
    destroy() {
      destroyed = true;
      attempt++;
      window.removeEventListener("pagehide", onHide);
      if (state.conversation) state.conversation.stop();
    },
  };
}
