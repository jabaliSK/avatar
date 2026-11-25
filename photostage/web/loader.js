"use strict";
// Downloads that report their progress, and that do not wait for ever on a connection which has gone quiet (a dropped
// VPN or a flaky link leaves a transfer stuck without any error): a transfer that delivers nothing for `stallMs` is
// abandoned and started again.

class DownloadError extends Error {
  constructor(message, status) {
    super(message);
    this.status = status || 0;
  }
}

// Collects the progress of several downloads running side by side into one report.
class Progress {
  constructor(report) {
    this.report = report;
    this.parts = new Map();
  }

  update(name, loaded, total) {
    this.parts.set(name, { loaded, total });
    let loadedAll = 0, totalAll = 0;
    for (const part of this.parts.values()) {
      loadedAll += part.loaded;
      totalAll += part.total || part.loaded;
    }
    this.report({ loaded: loadedAll, total: totalAll, fraction: totalAll ? Math.min(1, loadedAll / totalAll) : 0 });
  }
}

async function download(url, { onProgress = () => {}, stallMs = 20000, attempts = 3, pause = 400, fetcher = fetch } = {}) {
  for (let attempt = 1; ; attempt++) {
    const abort = new AbortController();
    let timer = null, stalled = false;
    const wake = () => {
      clearTimeout(timer);
      timer = setTimeout(() => {
        stalled = true;
        abort.abort();
      }, stallMs);
    };
    try {
      wake();
      const response = await fetcher(url, { signal: abort.signal });
      if (!response.ok) throw new DownloadError(`${url} answered ${response.status}.`, response.status);
      const total = Number(response.headers.get("content-length")) || 0;
      const type = response.headers.get("content-type") || "";
      onProgress(0, total);
      if (!response.body || !response.body.getReader) {
        const blob = await response.blob();
        onProgress(blob.size, blob.size);
        return blob;
      }
      const reader = response.body.getReader(), chunks = [];
      let loaded = 0;
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        chunks.push(value);
        loaded += value.length;
        onProgress(loaded, total);
        wake();
      }
      return new Blob(chunks, { type });
    } catch (error) {
      if (error instanceof DownloadError) throw error;
      if (attempt >= attempts) {
        throw new DownloadError(stalled
          ? "The connection to the avatar service stalled. Reload the page to try again."
          : "Cannot reach the avatar service. Check the network and reload the page.");
      }
    } finally {
      clearTimeout(timer);
    }
    await new Promise((resolve) => setTimeout(resolve, pause));
  }
}
