// The Studio's API. Everything the page does goes through request(): it adds the Studio key when one is stored, turns every failure into
// an ApiError with a message written for the person using the page, and asks for the key when the server says one is needed.
import { askKey } from "./ui.js";

const KEY_STORE = "avatar-studio-key";
const TIMEOUT_MS = 12000;

export class ApiError extends Error {
  constructor(status, message) {
    super(message);
    this.status = status;
  }
}

const FALLBACK = {
  0: "Cannot reach the Studio. Check your network connection and try again.",
  401: "The Studio key is missing or wrong.",
  403: "This request was blocked. Reload the page and try again.",
  404: "That item was not found. It may have been deleted.",
  413: "The photo is too large.",
  429: "Too many attempts. Wait a minute and try again.",
};

function messageFor(status, body) {
  const detail = body && body.detail;
  if (typeof detail === "string" && detail) return detail;
  if (Array.isArray(detail)) return "Check the values and try again.";
  if (FALLBACK[status]) return FALLBACK[status];
  return status >= 500 ? "The Studio ran into a problem. Try again in a moment." : "The request could not be completed.";
}

function storedKey() {
  try {
    return sessionStorage.getItem(KEY_STORE) || "";
  } catch {
    return "";
  }
}

function storeKey(key) {
  try {
    sessionStorage.setItem(KEY_STORE, key);
  } catch {
    // Without storage the key is asked for again next time.
  }
}

async function verify(key) {
  try {
    const response = await fetch("/api/unlock", { method: "POST", headers: { "X-Studio-Key": key }, cache: "no-store" });
    if (response.ok) return null;
    if (response.status === 429) return FALLBACK[429];
    if (response.status === 401) return "That key was not accepted.";
  } catch {
    // Handled below with the other failures.
  }
  return "The key could not be checked. Try again.";
}

let asking = null;

function obtainKey() {
  if (!asking) {
    asking = askKey(verify).finally(() => { asking = null; });
  }
  return asking;
}

export async function request(method, path, { json, form, timeoutMs = TIMEOUT_MS, retried = false } = {}) {
  const headers = {};
  const key = storedKey();
  if (key) headers["X-Studio-Key"] = key;
  let body;
  if (json !== undefined) {
    headers["Content-Type"] = "application/json";
    body = JSON.stringify(json);
  } else if (form) {
    body = form;
  }
  let response;
  try {
    const signal = typeof AbortSignal.timeout === "function" ? AbortSignal.timeout(timeoutMs) : undefined;
    response = await fetch(path, { method, headers, body, cache: "no-store", signal });
  } catch {
    throw new ApiError(0, FALLBACK[0]);
  }
  let data = null;
  if ((response.headers.get("content-type") || "").includes("json")) {
    data = await response.json().catch(() => null);
  }
  if (response.ok) return data;
  if (response.status === 401 && !retried) {
    const given = await obtainKey();
    if (given) {
      storeKey(given);
      return request(method, path, { json, form, timeoutMs, retried: true });
    }
  }
  throw new ApiError(response.status, messageFor(response.status, data));
}

const id = encodeURIComponent;

export const api = {
  health: () => request("GET", "/api/health", { timeoutMs: 6000 }),
  avatars: () => request("GET", "/api/avatars"),
  avatar: (name) => request("GET", `/api/avatars/${id(name)}`),
  create: (form) => request("POST", "/api/avatars", { form, timeoutMs: 120000 }),
  upgrade: (name, quality) => request("POST", `/api/avatars/${id(name)}/upgrade`, { json: { quality } }),
  rename: (name, fields) => request("PATCH", `/api/avatars/${id(name)}`, { json: fields }),
  saveSettings: (name, settings) => request("PUT", `/api/avatars/${id(name)}/settings`, { json: settings }),
  remove: (name) => request("DELETE", `/api/avatars/${id(name)}`),
  select: (name) => request("POST", "/api/stage", { json: { avatar: name } }),
  warm: (name) => request("POST", `/api/avatars/${id(name)}/warm`),
  status: (name) => request("GET", `/api/avatars/${id(name)}/status`),
  jobs: () => request("GET", "/api/jobs"),
  cancelJob: (job) => request("POST", `/api/jobs/${id(job)}/cancel`, { json: {} }),
  retryJob: (job, quality) => request("POST", `/api/jobs/${id(job)}/retry`, { json: quality ? { quality } : {} }),
  removeJob: (job) => request("DELETE", `/api/jobs/${id(job)}`),
};
