"""The avatars on disk: their names, the options of their player page, thumbnails and removal.

An avatar is a folder named by its key (32 hex digits) under the data directory. Besides the renderer's own files it holds a
meta.json with the name its owner gave it, when it was made and the options its player page was saved with. Avatars made
before meta.json existed have none and get a name and a date from what they do have.
"""

import json
import math
import os
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

KEY = re.compile(r"[a-f0-9]{32}")
PUBLISHED = ("rig.json", "ref.png", "bias.png", "idle.json", "idle.bin")
NAME_LIMIT = 60
THUMB_SIDE = 640


@dataclass(frozen=True)
class Option:
    """One thing the player page can show or do. The viewer's browser never needs to know more than the name: the player
    reads the same definitions from the server."""
    name: str
    kind: str                # "switch", "choice" or "number"
    default: object
    label: str
    hint: str
    needs: str = ""          # the option that has to be on for this one to matter
    choices: tuple = ()      # (value, label) pairs of a "choice"
    low: float = 0.0
    high: float = 1.0
    step: float = 0.1

    def public(self):
        shown = {"name": self.name, "kind": self.kind, "default": self.default, "label": self.label, "hint": self.hint}
        if self.needs:
            shown["needs"] = self.needs
        if self.kind == "choice":
            shown["choices"] = [{"value": value, "label": label} for value, label in self.choices]
        if self.kind == "number":
            shown.update(min=self.low, max=self.high, step=self.step)
        return shown


OPTIONS = (
    Option("chat", "switch", True, "Chat box", "The text box and Send button below the avatar."),
    Option("mic", "switch", True, "Microphone", "A Record button that turns speech into text. Browsers only allow the "
           "microphone on HTTPS or on localhost.", needs="chat"),
    Option("modes", "switch", True, "Mode switch", "Lets the viewer choose between a conversation and speaking an exact "
           "text.", needs="chat"),
    Option("mode", "choice", "chat", "Starting mode", "Conversation answers with the language model. Speak exactly this "
           "text says what was typed.", needs="chat",
           choices=(("chat", "Conversation"), ("speak", "Speak exactly this text"))),
    Option("stop", "switch", True, "Stop button", "Lets the viewer interrupt the avatar.", needs="chat"),
    Option("reply", "switch", True, "Reply text", "Shows what the avatar says as text below it.", needs="chat"),
    Option("status", "switch", True, "Status line", "Shows progress such as Thinking and Speaking. Errors always show."),
    Option("fullscreen", "switch", True, "Fullscreen button", "Lets the viewer fill the screen with the avatar."),
    Option("motion", "number", 1.0, "Body motion", "Breathing, sway and hand gestures. 0 turns them off, 1 is the normal "
           "strength and 2 the liveliest. Hands only move as far as the photo allows.", low=0.0, high=2.0, step=0.1),
    Option("clips", "switch", True, "Recorded gestures", "Plays short recorded clips of the avatar's body: thinking while a "
           "reply is prepared, gestures while it is spoken. Only an avatar that has a clip pack uses them, and with them off "
           "(or when a browser cannot play them) the built-in body motion is used.", needs="motion"),
    Option("home", "switch", False, "Link to all avatars", "A small link back to the console. Leave it off on screens "
           "that others use."),
)
DEFINITIONS = {option.name: option for option in OPTIONS}


def schema():
    return [option.public() for option in OPTIONS]


def check_option(option, value):
    """The value, or a ValueError saying what the option accepts."""
    if option.kind == "switch" and isinstance(value, bool):
        return value
    if option.kind == "choice" and isinstance(value, str) and value in dict(option.choices):
        return value
    if (option.kind == "number" and isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and option.low <= value <= option.high):
        return float(value)
    expected = {"switch": "true or false", "choice": "one of " + ", ".join(dict(option.choices)),
                "number": f"a number from {option.low:g} to {option.high:g}"}[option.kind]
    raise ValueError(f"{option.name} must be {expected}.")


def clean_options(values):
    """Options asked for by a caller, checked. Anything unknown or out of range is refused rather than guessed at."""
    if not isinstance(values, dict):
        raise ValueError("options must be an object.")
    unknown = sorted(set(values) - set(DEFINITIONS))
    if unknown:
        raise ValueError("Unknown option: " + ", ".join(unknown) + ".")
    return {name: check_option(DEFINITIONS[name], value) for name, value in values.items()}


def resolve(saved):
    """Every option with its value: the default unless the avatar was saved with another. Whatever a damaged meta.json
    holds that is not valid is ignored."""
    resolved = {option.name: option.default for option in OPTIONS}
    for name, value in (saved if isinstance(saved, dict) else {}).items():
        if name in DEFINITIONS:
            try:
                resolved[name] = check_option(DEFINITIONS[name], value)
            except ValueError:
                pass
    return resolved


def overrides(values):
    """Only what differs from the defaults is saved, so that a default changed later reaches the avatars that never chose."""
    return {name: value for name, value in values.items() if value != DEFINITIONS[name].default}


def clean_name(text):
    """A name for a person to read: one line, no control characters, not longer than the cards can show."""
    text = re.sub(r"[\x00-\x1f\x7f]+", " ", text if isinstance(text, str) else "")
    return " ".join(text.split())[:NAME_LIMIT].rstrip()


def name_from_upload(filename):
    return clean_name(Path(filename or "").stem)


def default_name(key):
    return f"Avatar {key[:6]}"


def stamp(seconds=None):
    """A UTC time (now, unless given as seconds since the epoch) the way meta.json and the API write it."""
    moment = datetime.now(timezone.utc) if seconds is None else datetime.fromtimestamp(seconds, timezone.utc)
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def folder(data, key):
    """The folder of a finished avatar, or None: a key that is not one, or an avatar still being made, is not there."""
    path = Path(data) / key
    if not KEY.fullmatch(key) or not all((path / name).is_file() for name in PUBLISHED):
        return None
    return path


def read_json(path):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def read_meta(avatar):
    meta = read_json(avatar / "meta.json")
    created, options = meta.get("created"), meta.get("options")
    return {"name": clean_name(meta.get("name")), "created": created if isinstance(created, str) else "",
            "options": options if isinstance(options, dict) else {}}


def write_meta(avatar, meta):
    """Replaces meta.json whole or not at all: a page reading it while it is rewritten never sees half of it."""
    temporary = avatar / "meta.json.tmp"
    temporary.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, avatar / "meta.json")


def update(avatar, name=None, options=None):
    """Rename an avatar and/or change some of its options (the others keep theirs). Raises ValueError for what is invalid,
    before anything is written."""
    meta = read_meta(avatar)
    if name is not None:
        meta["name"] = clean_name(name)
        if not meta["name"]:
            raise ValueError("Give the avatar a name.")
    if options is not None:
        meta["options"] = overrides({**resolve(meta["options"]), **clean_options(options)})
    write_meta(avatar, meta)


def made_at(avatar):
    """When the avatar was made, for one saved before meta.json recorded it: the age of its photo (or, failing that, of its rig)."""
    for name in ("photo.png", "rig.json"):
        try:
            return stamp((avatar / name).stat().st_mtime)
        except OSError:
            continue
    return ""


def describe(avatar):
    meta, rig, body = read_meta(avatar), read_json(avatar / "rig.json"), read_json(avatar / "body.json")
    hands = body.get("hands")
    return {
        "key": avatar.name, "name": meta["name"] or default_name(avatar.name),
        "created": meta["created"] or made_at(avatar),
        "size": rig.get("size"), "url": f"/a/{avatar.name}", "thumb": f"/assets/{avatar.name}/thumb.jpg",
        # What the body does: how many hands move and whether the whole body breathes and sways.
        "motion": {"hands": len(hands) if isinstance(hands, list) else 0, "sway": bool(body.get("sway"))},
        "options": resolve(meta["options"]),
        "build": build_state(avatar),
    }


def build_state(avatar):
    """How far the avatar's animations are (build.json, which photostage.builder keeps): {"state": "ready"} for one
    without, or whose build only adds to animations it already has (filler phrases)."""
    state = read_json(avatar / "build.json")
    if not state or state.get("job") == "fillers":
        return {"state": "ready"}
    keys = ("state", "kind", "step", "steps", "label", "percent", "eta", "error", "started", "finished")
    return {k: state[k] for k in keys if k in state}


def listing(data):
    avatars = []
    for path in Path(data).iterdir():
        if folder(data, path.name):
            try:
                avatars.append(describe(path))
            except OSError:
                continue  # deleted while the list was being made
    return sorted(avatars, key=lambda item: (item["created"], item["key"]), reverse=True)


def thumbnail(avatar):
    """The small picture of the photo that the console shows, made the first time it is asked for."""
    target = avatar / "thumb.jpg"
    if not target.is_file():
        with Image.open(avatar / "photo.png") as photo:
            small = photo.convert("RGB")
        small.thumbnail((THUMB_SIDE, THUMB_SIDE), Image.Resampling.LANCZOS)
        temporary = avatar / "thumb.jpg.tmp"
        small.save(temporary, "JPEG", quality=85, optimize=True)
        os.replace(temporary, target)
    return target


def remove(avatar):
    shutil.rmtree(avatar)
