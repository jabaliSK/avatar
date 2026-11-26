// Notifications and dialogs shared by the pages.
import { h } from "./dom.js";
import { icon } from "./icons.js";

const MAX_TOASTS = 4;
const TOAST_ICONS = { success: "check", error: "alert", warn: "alert", info: "info" };

export function toast(message, kind = "info", timeoutMs = 5200) {
  const host = document.getElementById("toasts");
  let timer = null;
  const dismiss = () => {
    clearTimeout(timer);
    item.classList.add("leaving");
    setTimeout(() => item.remove(), 180);
  };
  const item = h("div", { class: `toast ${kind}`, role: kind === "error" ? "alert" : "status" },
    icon(TOAST_ICONS[kind] || "info", 18),
    h("p", null, message),
    h("button", { class: "toast-close", type: "button", "aria-label": "Dismiss", onClick: dismiss }, icon("close", 14)));
  host.append(item);
  while (host.children.length > MAX_TOASTS) host.firstChild.remove();
  if (timeoutMs) timer = setTimeout(dismiss, timeoutMs);
  return dismiss;
}

let counter = 0;

/** A dialog built on <dialog>: focus stays inside, Escape and a click on the backdrop close it unless it is locked (e.g. while uploading). */
export function modal({ title, subtitle, className = "" }) {
  const id = `dialog-title-${++counter}`;
  const body = h("div", { class: "modal-body" });
  const footer = h("div", { class: "modal-foot" });
  const closeButton = h("button", { class: "btn btn-ghost btn-icon modal-x", type: "button", "aria-label": "Close", onClick: () => handle.close("") }, icon("close"));
  const card = h("div", { class: "modal-card" },
    h("i", { class: "corners" }),
    h("header", { class: "modal-head" },
      h("div", null, h("h2", { class: "modal-title", id }, title), subtitle ? h("p", { class: "modal-sub" }, subtitle) : null),
      closeButton),
    body,
    footer);
  const dialog = h("dialog", { class: `modal ${className}`.trim(), "aria-labelledby": id }, card);
  let resolveClosed;
  const closed = new Promise((resolve) => { resolveClosed = resolve; });
  const handle = {
    dialog, body, footer, closed, locked: false,
    open() {
      document.body.append(dialog);
      dialog.showModal();
    },
    close(value = "") {
      if (dialog.open) dialog.close(value);
    },
    lock(locked) {
      handle.locked = locked;
      closeButton.disabled = locked;
    },
  };
  let pressedOnBackdrop = false;
  dialog.addEventListener("cancel", (event) => { if (handle.locked) event.preventDefault(); });
  dialog.addEventListener("mousedown", (event) => { pressedOnBackdrop = event.target === dialog; });
  dialog.addEventListener("click", (event) => {
    if (event.target === dialog && pressedOnBackdrop && !handle.locked) handle.close("");
  });
  dialog.addEventListener("close", () => {
    dialog.remove();
    resolveClosed(dialog.returnValue);
  });
  return handle;
}

/** Resolves true when the person confirms. The safe choice has the focus. */
export function confirmDialog({ title, message, confirmLabel = "Confirm", cancelLabel = "Cancel", danger = false }) {
  const dialog = modal({ title, className: "narrow" });
  dialog.body.append(h("p", { class: "confirm-text" }, message));
  const cancel = h("button", { class: "btn btn-ghost", type: "button", onClick: () => dialog.close("no") }, cancelLabel);
  const confirm = h("button", { class: `btn ${danger ? "btn-danger-solid" : "btn-primary"}`, type: "button", onClick: () => dialog.close("yes") }, confirmLabel);
  dialog.footer.append(cancel, confirm);
  dialog.open();
  cancel.focus();
  return dialog.closed.then((value) => value === "yes");
}

/** Asks for the Studio key. `verify(key)` returns null when the key works, otherwise the message to show. Resolves the key, or null if cancelled. */
export function askKey(verify) {
  const dialog = modal({ title: "Enter the Studio key", subtitle: "Creating, changing and deleting avatars needs the key your administrator set.", className: "narrow" });
  let accepted = null;
  const input = h("input", { type: "password", class: "input", autocomplete: "current-password", "aria-label": "Studio key", required: true });
  const error = h("p", { class: "form-error", role: "alert", hidden: true });
  const submit = h("button", { class: "btn btn-primary", type: "submit" }, icon("lock", 16), "Unlock");
  const form = h("form", {
    onSubmit: async (event) => {
      event.preventDefault();
      if (!input.value) return;
      submit.disabled = true;
      error.hidden = true;
      const problem = await verify(input.value);
      submit.disabled = false;
      if (problem) {
        error.textContent = problem;
        error.hidden = false;
        input.select();
        return;
      }
      accepted = input.value;
      dialog.close("ok");
    },
  },
  h("label", { class: "field" }, h("span", null, "Studio key"), input),
  error,
  h("div", { class: "modal-foot" }, h("button", { class: "btn btn-ghost", type: "button", onClick: () => dialog.close("") }, "Cancel"), submit));
  dialog.body.append(form);
  dialog.open();
  input.focus();
  return dialog.closed.then(() => accepted);
}
