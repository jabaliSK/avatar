"""Real-browser check of the avatar page: headless Chromium (Playwright) with fake camera and microphone devices.

It behaves like a user: opens the page, allows camera and microphone, starts the conversation, and then records what the browser's own
WebRTC stack reports while the avatar answers a question that the fake microphone plays. This covers what e2e_client.py (aiortc) cannot:
the page and its TURN settings as served by the gateway, Chrome's jitter buffer and H.264 decoder, and a camera stream going up.

    pip install playwright && playwright install chromium
    python browser_e2e.py --url https://10.25.73.101:8282 --mic mic.wav --out run1 [--relay-only]

The fake microphone starts playing the moment the page asks for it, so --mic needs several seconds of silence before the speech
(the question ends up in the chat history of the page). --relay-only makes the page's RTCPeerConnection use iceTransportPolicy "relay",
so the media can only travel through the TURN server, as for a browser on another network. Certificate errors are ignored: the demo
certificate is self-signed.
"""
import argparse
import json
import pathlib
import re
import sys

from playwright.sync_api import TimeoutError as PlaywrightTimeout, sync_playwright

INIT_JS = r"""
(() => {
  const Orig = window.RTCPeerConnection;
  window.__pcs = [];
  window.__samples = [];
  window.__ice = [];
  window.RTCPeerConnection = function (cfg, ...rest) {
    cfg = cfg || {};
    if (%RELAY%) cfg.iceTransportPolicy = "relay";
    const pc = new Orig(cfg, ...rest);
    window.__pcs.push(pc);
    const t0 = performance.now();
    const note = (o) => window.__ice.push(Object.assign({ t_ms: Math.round(performance.now() - t0) }, o));
    pc.addEventListener("icegatheringstatechange", () => note({ gathering: pc.iceGatheringState }));
    pc.addEventListener("icecandidateerror", (e) => note({ error: [e.errorCode, e.errorText, e.url, e.address, e.port].join(" ") }));
    pc.addEventListener("icecandidate", (e) => note({ candidate: e.candidate ? [e.candidate.type, e.candidate.protocol, e.candidate.relayProtocol].filter(Boolean).join("/") : "end" }));
    pc.addEventListener("connectionstatechange", () => note({ connection: pc.connectionState }));
    return pc;
  };
  window.RTCPeerConnection.prototype = Orig.prototype;
  Object.keys(Orig).forEach((k) => { try { window.RTCPeerConnection[k] = Orig[k]; } catch (e) {} });

  window.__startSampler = () => {
    const t0 = performance.now();
    setInterval(async () => {
      const pc = window.__pcs[window.__pcs.length - 1];
      const s = { t: (performance.now() - t0) / 1000 };
      (await pc.getStats()).forEach((r) => {
        if (r.type === "media-source" && r.kind === "audio") s.mic = r.audioLevel;
        if (r.type === "inbound-rtp" && r.kind === "audio") s.ain = r.audioLevel;
        if (r.type === "inbound-rtp" && r.kind === "video") { s.vdec = r.framesDecoded; s.vdrop = r.framesDropped; s.vfreeze = r.freezeCount; }
      });
      window.__samples.push(s);
    }, 50);
  };

  window.__final = async () => {
    const pc = window.__pcs[window.__pcs.length - 1];
    const rep = await pc.getStats();
    let sel = null;
    rep.forEach((r) => { if (r.type === "transport" && r.selectedCandidatePairId) sel = rep.get(r.selectedCandidatePairId); });
    const out = { connection: pc.connectionState, ice: pc.iceConnectionState };
    const cand = (c) => c && { type: c.candidateType, protocol: c.protocol, relayProtocol: c.relayProtocol, address: (c.address || c.ip) + ":" + c.port };
    if (sel) out.pair = { local: cand(rep.get(sel.localCandidateId)), remote: cand(rep.get(sel.remoteCandidateId)), rtt_ms: sel.currentRoundTripTime * 1000 };
    rep.forEach((r) => {
      const jb = 1000 * r.jitterBufferDelay / Math.max(1, r.jitterBufferEmittedCount);
      const codec = rep.get(r.codecId) && rep.get(r.codecId).mimeType;
      if (r.type === "inbound-rtp" && r.kind === "video") out.video = { width: r.frameWidth, height: r.frameHeight, decoded: r.framesDecoded, dropped: r.framesDropped, freezes: r.freezeCount, jitter_buffer_ms: jb, codec: codec };
      if (r.type === "inbound-rtp" && r.kind === "audio") out.audio = { concealed_samples: r.concealedSamples, total_samples: r.totalSamplesReceived, jitter_buffer_ms: jb, codec: codec };
      if (r.type === "outbound-rtp") (out.sent = out.sent || {})[r.kind] = r.bytesSent;
    });
    return out;
  };
})();
"""


def bursts(samples, key, threshold, gap=0.6):
    """(start, end) times where `key` stays above the threshold, merging pauses shorter than `gap` seconds."""
    found = []
    for s in samples:
        if s.get(key, 0) > threshold:
            if found and s["t"] - found[-1][1] <= gap:
                found[-1][1] = s["t"]
            else:
                found.append([s["t"], s["t"]])
    return found


def sample_at(samples, t):
    return min(samples, key=lambda s: abs(s["t"] - t))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--url", required=True)
    ap.add_argument("--mic", required=True, help="WAV played by the fake microphone (16-bit PCM, with silence before the speech)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--relay-only", action="store_true")
    ap.add_argument("--seconds", type=float, default=60.0, help="give up this long after the conversation starts")
    ap.add_argument("--connect-timeout", type=float, default=25.0, help="give up if the connection is not up this long after clicking start")
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = []

    with sync_playwright() as p:
        browser = p.chromium.launch(args=["--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream",
                                          f"--use-file-for-fake-audio-capture={args.mic}%noloop", "--autoplay-policy=no-user-gesture-required"])
        ctx = browser.new_context(ignore_https_errors=True, viewport={"width": 1100, "height": 900}, locale="en-US")
        ctx.add_init_script(INIT_JS.replace("%RELAY%", "true" if args.relay_only else "false"))
        page = ctx.new_page()
        page.on("console", lambda m: log.append(f"{m.type}: {m.text[:200]}"))
        page.on("pageerror", lambda e: log.append(f"pageerror: {str(e)[:200]}"))
        offers = []

        def on_response(r):
            # The page ignores a refused offer (concurrency_limit_reached), so this is the only place where it shows.
            if "/webrtc/offer" in r.url:
                try:
                    body = r.text()
                except Exception as e:  # noqa: BLE001
                    body = f"<{e}>"
                offers.append({"status": r.status, "body": body[:160].replace("\n", " ")})

        page.on("response", on_response)
        page.goto(args.url, wait_until="networkidle", timeout=60000)
        page.get_by_text(re.compile("允许访问|allow", re.I)).first.click()
        start = page.get_by_text(re.compile("开始对话|start", re.I)).first
        start.wait_for(timeout=15000)
        page.screenshot(path=str(out / "browser_before_start.png"))
        start.click()
        page.wait_for_function("() => window.__pcs.length > 0", timeout=15000)
        connected = True
        try:
            page.wait_for_function("() => window.__pcs[window.__pcs.length - 1].connectionState === 'connected'",
                                   timeout=args.connect_timeout * 1000)
        except PlaywrightTimeout:
            connected = False
        if connected:
            page.evaluate("() => window.__startSampler()")
        else:
            page.screenshot(path=str(out / "browser_end.png"))
            refused = any("concurrency_limit_reached" in o["body"] for o in offers)
            summary = {"url": args.url, "relay_only": args.relay_only,
                       "error": ("the gateway refused the offer: concurrency_limit_reached (another session is still active)" if refused
                                 else "not connected %.0f s after clicking start" % args.connect_timeout),
                       "offers": offers, "ice_log": page.evaluate("() => window.__ice"), "console": log[-12:]}
            json.dump(summary, open(out / "summary.json", "w"), indent=1, ensure_ascii=False)
            browser.close()
            print(json.dumps(summary, indent=1, ensure_ascii=False))
            sys.exit(1)

        # The question is played by the fake microphone; wait for the answer to start and finish, with a screenshot while it speaks.
        shot = False
        for _ in range(int(args.seconds * 2)):
            page.wait_for_timeout(500)
            samples = page.evaluate("() => window.__samples")
            speech = bursts(samples, "mic", 0.01)
            reply = [b for b in bursts(samples, "ain", 0.01) if speech and b[0] > speech[0][1]]
            if not reply:
                continue
            now = samples[-1]["t"]
            if not shot and now - reply[0][0] >= 4.0:
                page.screenshot(path=str(out / "browser_reply.png"))
                shot = True
            if now - reply[0][1] > 3.0:
                break

        final = page.evaluate("() => window.__final()")
        videos = page.evaluate("() => [...document.querySelectorAll('video')].map(v => [v.videoWidth, v.videoHeight])")
        summary = {"url": args.url, "relay_only": args.relay_only, "webrtc": final, "video_elements_wxh": videos,
                   "page_text": page.inner_text("body")[:400], "ice_log": page.evaluate("() => window.__ice")}
        if speech and reply:
            speech_end = speech[0][1]
            a, b = sample_at(samples, reply[0][0]), sample_at(samples, reply[0][1])
            summary["speech_s"] = [round(speech[0][0], 2), round(speech_end, 2)]
            summary["reply_sound_after_speech_end_s"] = round(reply[0][0] - speech_end, 2)
            summary["reply_span_s"] = round(reply[0][1] - reply[0][0], 2)
            summary["video_during_reply"] = {"decoded_fps": round((b["vdec"] - a["vdec"]) / (b["t"] - a["t"]), 1),
                                             "dropped": b["vdrop"] - a["vdrop"], "freezes": b["vfreeze"] - a["vfreeze"]}
        else:
            summary["error"] = "no reply heard (speech burst: %s)" % bool(speech)
        json.dump(summary, open(out / "summary.json", "w"), indent=1, ensure_ascii=False)
        json.dump(samples, open(out / "samples.json", "w"))
        page.screenshot(path=str(out / "browser_end.png"))
        browser.close()

    print(json.dumps(summary, indent=1, ensure_ascii=False))
    errors = [m for m in log if m.startswith(("error", "pageerror"))]
    print("console errors:", errors[:6] if errors else "none")
    sys.exit(1 if "error" in summary else 0)

if __name__ == "__main__":
    main()
