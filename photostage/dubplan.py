"""The body of a dubbed reply: which clip frame is under each 25 fps picture of the speech.

With PHOTOSTAGE_RENDERER=musetalk the server renders the whole picture of the avatar while it speaks: the gesture clips'
own frames, with MuseTalk lip-syncing the lower face inside them (scripts/dub_worker.py). The head can then never come
apart from the body, because it is the clip's own head. The reply's text and its length are known before a frame is
rendered, so the gestures are planned here, up front, by the rules the browser's ClipDirector uses (photostage/web/clips.js):
a gesture that suits the sentence being said, a rest between two gestures, none that starts when the speech is almost over,
and none that overruns the speech by more than CLIP_OVERRUN. A rest is a quiet clip (the avatar idling), not a still frame,
so the body never freezes while the mouth moves. Clips meet in a short crossfade. Gestures begin and end near the photo's
pose, but a rest is a loop that is cut wherever its length ends, and a crossfade between two different poses shows both: a
double face. With `poses` (a small picture of every frame: pose_of) each rest therefore begins where in its loop both of its
ends look most like what they meet, the gesture before and after it, or the photo's pose at either end of the reply (where
the player's own clips start). Lip-sync fades out after the last sound, leaving the clip's own mouth; when the speech ends
during a gesture it holds (on silence: a closed mouth) until the gesture is back in the photo's pose, so the clip's own
mouth never shows while its body still moves (a generated gesture's mouth can move on after its speech).
"""

import math
import random
import re
from dataclasses import dataclass, field

import numpy as np

FPS = 25
FIRST_TAIL = 0.35  # a short reply can still open with one short gesture
MIN_TAIL = 1.2  # subsequent gestures need this much speech left
OVERRUN = 1.5  # a gesture may last this much longer than the speech that is left
GAP = (0.25, 1.2)  # the body rests this long between two gestures at the normal amount of motion
RECENT = 2  # the latest gestures are not repeated
LOOKAHEAD = 0.4  # a gesture suits what is said this long after it begins
CROSSFADE = 0.2  # two clips meet over this long
DUB_FADE = 0.2  # lip-sync gives way to the clip's own mouth over this long, after the last sound or a gesture
REST_TAIL = 0.3  # a reply that ends during a rest is pictured this much longer
REST_ROLES = ("idle", "listen", "think")
POSE_SIZE = (60, 105)  # a frame's pose is the frame at this size, in grey
SIMILAR = {
    "greet": ("greet", "explain", "calm"), "ask": ("ask", "explain", "calm"),
    "emphasis": ("emphasis", "explain", "calm"), "explain": ("explain", "emphasis", "calm"),
    "calm": ("calm", "explain", "emphasis"),
}
# The browser's clipSentences and clipMood, in Python; tests/test_dubplan.py keeps them in step.
SENTENCE = re.compile(r"[^.!?\u2026\n]+[.!?\u2026]*[\"')\]]*")
WORDY = re.compile(r"[\w\u00c0-\uffff]")
GREETING = re.compile(r"^\W*(hello|hi|hey|welcome|greetings|good (morning|afternoon|evening)|nice to meet)\b", re.I)
QUESTION = re.compile(
    r"^\W*(?:(?:could|would|can|will)\s+you|(?:do|does|did|have|has)\s+(?:you|we|they)|"
    r"(?:what|where|when|why|who|which|how)\s+(?:is|are|was|were|do|does|did|can|could|would|should|will|have|has))\b",
    re.I)


def sentences(text):
    """The reply as sentences, each with the share of the speech (from, to) that its words take."""
    found, total = [], 0
    for part in SENTENCE.findall(str(text or "")):
        words = len(part.split())
        if not words or not WORDY.search(part):
            continue
        found.append({"text": part.strip(), "words": words, "from": total, "to": total + words})
        total += words
    for sentence in found:
        sentence["from"] /= total
        sentence["to"] /= total
    return found


def mood(sentence, index):
    text, words = sentence["text"], sentence["words"]
    normalized = text.lower().replace("\u2019", "'")
    if index == 0 and GREETING.search(text):
        return "greet"
    if re.search(r"\?[\"')\]]*$", text) or QUESTION.search(text):
        return "ask"
    if re.search(r"![\"')\]]*$", text):
        return "emphasis"
    if re.search(r"\b(thank you|i appreciate|i'm sorry|i am sorry|my apologies|i understand|that sounds difficult)\b",
                 normalized):
        return "calm"
    if re.search(r"\b(absolutely|congratulations|crucial|delighted|excited|great news|importantly|key point|wonderful)\b",
                 normalized):
        return "emphasis"
    if re.search(r"\b(because|for example|for instance|first|finally|in summary|next|the reason|to begin with)\b",
                 normalized):
        return "explain"
    if words >= 18:
        return "explain"
    if words <= 6:
        return "calm"
    return "emphasis" if index % 2 else "explain"


@dataclass
class Clip:
    name: str
    role: str
    mood: str
    frames: int
    fps: float
    loop: bool

    @property
    def seconds(self):
        """How long the clip plays: a clip that does not loop never shows its last frame twice."""
        return self.frames / self.fps if self.loop else (self.frames - 1) / self.fps

    def frame(self, seconds):
        index = int(math.floor(seconds * self.fps + 1e-6))
        return index % self.frames if self.loop else min(self.frames - 1, max(0, index))


@dataclass
class Plan:
    clips: list  # names; a frame refers to a clip by its place in this list
    frames: list  # per picture: [clip, frame, next clip or -1, its frame, its weight 0-1, lip-sync weight 0-1]
    gestures: list = field(default_factory=list)  # [name, start seconds, end seconds] of each gesture, for the log
    fps: int = FPS
    quiet_from: int | None = None

    @property
    def seconds(self):
        return len(self.frames) / self.fps

    def message(self):
        return {"fps": self.fps, "clips": self.clips, "frames": self.frames,
                **({"quiet_from": self.quiet_from} if self.quiet_from is not None else {})}


def continuous(clip, speech, start_frame):
    """Continue one native-speed loop, then use its matching pre-rendered silent face; never dissolve two mouths."""
    if not clip.loop or clip.role != "idle" or clip.frames < 2 or not 0 < speech <= 45:
        raise ValueError("Continuous speech requires an idle loop and 0 to 45 seconds of audio.")
    if isinstance(start_frame, bool) or not isinstance(start_frame, int) or not 0 <= start_frame < clip.frames:
        raise ValueError("Invalid continuous-loop start frame.")
    frames = [[0, clip.frame(start_frame / clip.fps + i / FPS), -1, 0, 0.0, 1.0]
              for i in range(math.ceil((speech + 0.65) * FPS))]
    return Plan([clip.name], frames, quiet_from=math.ceil((speech + 0.4) * FPS))


def clips_of(manifest):
    """The clips of a clip pack (clips.json) that a plan can use."""
    found = {}
    for name, spec in (manifest.get("clips") or {}).items():
        if spec.get("role") not in {"idle", "listen", "think", "talk"} or not spec.get("fps") or not spec.get("duration"):
            continue
        frames = round(spec["duration"] * spec["fps"])
        if frames >= 2:
            found[name] = Clip(name, spec["role"], spec.get("mood", spec["role"]), frames, float(spec["fps"]),
                               bool(spec.get("loop")))
    return found


def pose_of(image):
    """What a frame (an RGB array) looks like, small enough to compare with another frame's: see begin_rests."""
    import cv2

    grey = cv2.cvtColor(np.ascontiguousarray(image), cv2.COLOR_RGB2GRAY)
    return cv2.resize(grey, POSE_SIZE, interpolation=cv2.INTER_AREA).astype(np.float32).ravel()


def difference(a, b):
    return 0.0 if a is None or b is None else float(np.sqrt(np.mean((a - b) ** 2)))


def begin_rests(pieces, poses, anchor):
    """Sets where in its loop each rest of the timeline begins, so that its first frame looks most like the frame it is
    faded in over (the last of the gesture before it, or the photo's pose, `anchor`, at the start of the reply) and its last
    like the one it fades into (the first of the next gesture, or the photo's pose at the end). `poses`: clip name -> the
    pose_of each of its frames. The rests' lengths, and so the timeline, are kept."""
    def pose(clip, seconds):
        found = poses.get(clip.name)
        return None if found is None else found[clip.frame(seconds)]

    for i, piece in enumerate(pieces):
        clip, _, length, _ = piece
        if clip.role not in REST_ROLES or clip.name not in poses:
            continue
        came = pose(pieces[i - 1][0], pieces[i - 1][2] + pieces[i - 1][3]) if i else anchor
        goes = pose(pieces[i + 1][0], pieces[i + 1][3]) if i + 1 < len(pieces) else anchor
        # A clip that does not loop can only begin early enough to last the rest.
        starts = clip.frames if clip.loop else max(1, math.floor((clip.seconds - length) * clip.fps + 1e-6) + 1)
        cost, first = min((difference(came, pose(clip, k / clip.fps)) + difference(pose(clip, k / clip.fps + length), goes),
                           k) for k in range(starts))
        piece[3] = first / clip.fps


def plan(clips, text, speech, *, pace=1.0, rng=None, still=None, poses=None, anchor=None):
    """The pictures of `speech` seconds of `text`. `clips`: name -> Clip. `pace` scales the rests (the player's amount of
    motion); at 0 the body only rests. `still`, a one-frame clip, stands in for a pack without a quiet clip (the photo).
    `poses` and `anchor` (the photo's pose) let the rests begin where they meet the gestures best: see begin_rests."""
    if not speech > 0:
        raise ValueError("The speech must last a positive number of seconds.")
    rng = rng or random.Random()
    rests = next(([c for c in clips.values() if c.role == role] for role in REST_ROLES
                  if any(c.role == role for c in clips.values())), [])
    if not rests:
        if still is None:
            raise ValueError("A dubbed reply needs a quiet clip to rest in.")
        rests = [still]
    talks = [c for c in clips.values() if c.role == "talk"] if pace > 0 else []
    said = sentences(text)

    def mood_at(seconds):
        if not said:
            return "explain"
        share = min(max(seconds / speech, 0.0), 0.9999)
        index = next((i for i, s in enumerate(said) if share < s["to"]), len(said) - 1)
        return mood(said[index], index)

    recent = []

    def rank(wanted, left):
        usable = [c for c in talks if c.seconds <= left + OVERRUN]
        pool = usable
        for keep in range(RECENT, 0, -1):
            fresh = [c for c in usable if c.name not in recent[-keep:]]
            if fresh:
                pool = fresh
                break
        for similar in SIMILAR.get(wanted, ()):
            group = [c for c in pool if c.mood == similar]
            if group:
                return group
        return pool

    # The timeline as pieces [clip, start, length, where in the clip it begins], then sampled at FPS.
    pieces, gestures, now = [], [], 0.0
    lips = speech  # lip-sync lasts this long, then fades out
    while now < speech:
        left = speech - now
        choice = None
        if left >= (MIN_TAIL if gestures else FIRST_TAIL):
            pool = rank(mood_at(now + LOOKAHEAD), left)
            choice = pool[rng.randrange(len(pool))] if pool else None
        if choice:
            pieces.append([choice, now, choice.seconds, 0.0])
            gestures.append([choice.name, round(now, 3), round(now + choice.seconds, 3)])
            recent = (recent + [choice.name])[-RECENT:]
            now += choice.seconds
            if now >= speech + DUB_FADE + 1 / FPS:
                lips = now - DUB_FADE - 1 / FPS  # gone by the gesture's last picture
                break
        rest = rests[rng.randrange(len(rests))]
        left = speech - now
        length = (GAP[0] + rng.random() * (GAP[1] - GAP[0])) / max(pace, 0.25)
        # Where no gesture fits now (fewer fit as the speech runs out) or after this rest, the rest lasts to the end; one
        # after a gesture that outlasts the speech gives the lip-sync time to fade out, and the clips time to meet.
        if not choice or left - length < MIN_TAIL:
            length = max(left + REST_TAIL, CROSSFADE + 2 / FPS)
        pieces.append([rest, now, length, 0.0])
        now += length
    if poses:
        begin_rests(pieces, poses, anchor)
    end = max(speech, now)
    names = sorted({piece[0].name for piece in pieces})
    place = {name: i for i, name in enumerate(names)}
    frames = []
    for index in range(max(1, math.ceil(end * FPS - 1e-6))):
        at = index / FPS
        current = max((i for i, p in enumerate(pieces) if p[1] <= at + 1e-9), default=0)
        clip, start, _, phase = pieces[current]
        row = [place[clip.name], clip.frame(at - start + phase), -1, 0, 0.0]
        if current:
            # Crossfade from where the previous clip stopped (it is held there) into this one.
            fade = (at - start) / CROSSFADE
            if fade < 1:
                before, began, length, offset = pieces[current - 1]
                weight = round(fade * fade * (3 - 2 * fade), 4)
                held = [place[before.name], before.frame(min(length, at - began) + offset)]
                row = held + ([place[clip.name], clip.frame(at - start + phase), weight] if weight > 0 else [-1, 0, 0.0])
        after = (at - lips) / DUB_FADE
        row.append(round(1.0 if after <= 0 else max(0.0, 1 - after), 4))
        frames.append(row)
    return Plan(names, frames, gestures)
