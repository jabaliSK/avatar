"""The optional shared key for the actions that change something: creating, editing and deleting avatars.

Talking to an avatar needs no key (that is what the page is for). With KKA_STUDIO_KEY set, the Studio asks for it once per browser
tab and sends it as the X-Studio-Key header. Wrong keys are counted per client address, so the key cannot be guessed quickly.
"""
import hmac
import time
from urllib.parse import urlsplit

from fastapi import HTTPException, Request

MAX_TRACKED = 1000
READ_ONLY = ("GET", "HEAD", "OPTIONS")


def reject_cross_site(request: Request):
    """A browser tab on another site must not be able to make this browser create, change or delete anything here.

    Browsers say where a request comes from (Sec-Fetch-Site, Origin); programs such as curl send neither and are not affected.
    """
    if request.method in READ_ONLY:
        return
    site = request.headers.get("sec-fetch-site")
    origin = request.headers.get("origin")
    hosts = {request.headers.get("host", ""), request.headers.get("x-forwarded-host", "")} - {""}
    if (site and site not in ("same-origin", "none")) or (origin and urlsplit(origin).netloc not in hosts):
        raise HTTPException(403, "Requests from other sites are not accepted.")


class KeyGuard:
    def __init__(self, key, attempts=10, window_s=60.0, clock=time.monotonic):
        self.key = key.encode()
        self.attempts, self.window_s, self.clock = attempts, window_s, clock
        self.failures = {}  # client address -> times of its recent wrong keys

    @property
    def locked(self):
        return bool(self.key)

    def check(self, request):
        if not self.key:
            return
        who = request.client.host if request.client else "unknown"
        now = self.clock()
        if len(self.failures) > MAX_TRACKED:
            self.failures = {w: t for w, t in self.failures.items() if t and now - t[-1] <= self.window_s}
        recent = [t for t in self.failures.get(who, ()) if now - t <= self.window_s]
        if len(recent) >= self.attempts:
            self.failures[who] = recent
            raise HTTPException(429, "Too many wrong keys. Wait a minute and try again.")
        given = request.headers.get("x-studio-key", "").encode("utf-8", "replace")
        if hmac.compare_digest(given, self.key):
            self.failures.pop(who, None)
            return
        self.failures[who] = recent + [now]
        raise HTTPException(401, "A valid Studio key is required for this action.")
