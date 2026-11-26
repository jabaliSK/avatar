"""Start OpenAvatarChat (src/demo.py, unmodified) after a few runtime adjustments: the H.264 bitrate limits that its RTC client hard-codes
are raised, aiortc's built-in public STUN server is switched off, and reconnecting browsers are handled (rtc_session.py).

The stock client pins the video encoder to 0.5-2.5 Mbps, which is low for a 720p+ photoreal face. The values are set inside the
client module at import time, so the module has to be imported first, under the same name the engine will use for it
(`client.rtc_client.client_handler_rtc`, via the `src/handlers` search path). The engine then reuses the cached module, which keeps
both the aiortc patches and the limits applied exactly once.
"""
import os
import runpy
import sys

OAC_ROOT = os.environ.get("OAC_ROOT", "/opt/oac")
SRC = os.path.join(OAC_ROOT, "src")

os.chdir(OAC_ROOT)
# `python src/demo.py` puts the project root (so `src.*` imports resolve) and then src/ on the path; the kka handlers also import
# `handlers.vad...` through src/.
sys.path[:0] = [OAC_ROOT, SRC]
sys.path.append(os.path.join(SRC, "handlers"))


def _bps(name: str, default_kbps: int) -> int:
    return int(float(os.environ.get(name, default_kbps)) * 1000)


import client.rtc_client.client_handler_rtc  # noqa: E402,F401  (applies the aiortc H.264 patches)
import rtc_session  # noqa: E402
from aiortc.codecs import h264  # noqa: E402
from aiortc.rtcicetransport import RTCIceGatherer  # noqa: E402
from fastrtc.stream import Body, Stream  # noqa: E402
from fastrtc.webrtc_connection_mixin import WebRTCConnectionMixin  # noqa: E402

# OpenAvatarChat builds the server's peer connection without ICE servers, and aiortc then falls back to Google's public STUN server
# (stun.l.google.com): a DNS lookup and a UDP request to a third party in every session. An offline deployment must not do that, and
# nothing here needs it: browsers on other machines reach the gateway through the TURN relay, and on one network host candidates suffice.
RTCIceGatherer.getDefaultIceServers = classmethod(lambda cls: [])

# One conversation at a time, and a browser that went away is only noticed ~30 s later. Without this a reload (or a closed and reopened
# tab) is refused silently for that long: a client reconnecting from the same address replaces its stale session at once.
# KKA_RECONNECT_TAKEOVER=0 restores strict first come, first served. The time-limit task of FastRTC no longer pins finished peer
# connections in memory for `connection_ttl`, and what a session left behind (reference cycles, freed heap pages) is released once the
# gateway has been idle for a few seconds.
_takeover = os.environ.get("KKA_RECONNECT_TAKEOVER", "1") != "0"
rtc_session.install_reconnect(Stream, Body, _takeover)
rtc_session.install_time_limit_release(WebRTCConnectionMixin)
rtc_session.install_idle_trim(WebRTCConnectionMixin)
print(f"launch: a reconnecting client replaces its own stale session = {_takeover}", flush=True)

# The handlers, the RTC stack and the LLM stream all share one interpreter. A thread that needs the GIL back after a native call (every
# torch op in the TTS) waits up to the switch interval for a busy thread to give it up; 5 ms by default, which adds up over the
# hundreds of small ops in one synthesis call.
sys.setswitchinterval(float(os.environ.get("KKA_GIL_SWITCH_MS", "0.5")) / 1000)

h264.MIN_BITRATE = _bps("KKA_H264_MIN_KBPS", 1000)
h264.DEFAULT_BITRATE = _bps("KKA_H264_KBPS", 4000)
h264.MAX_BITRATE = _bps("KKA_H264_MAX_KBPS", 8000)
print(f"launch: H.264 bitrate min/default/max = {h264.MIN_BITRATE // 1000}/{h264.DEFAULT_BITRATE // 1000}/{h264.MAX_BITRATE // 1000} kbps",
      flush=True)

runpy.run_path(os.path.join(SRC, "demo.py"), run_name="__main__")
