"""What the Studio accepts from a user: the photo, the name and the description, and the avatar id made from the name.

Nothing here trusts the browser. The photo is decoded and written again as a new PNG (so what the models and the renderer read is
never the uploaded file), and the text has its control characters removed and is checked to be English.
"""
import io
import re
import unicodedata
from typing import NamedTuple

from PIL import Image, ImageOps, UnidentifiedImageError

FORMATS = ["PNG", "JPEG", "WEBP"]  # also what Image.open may try: the other decoders (EPS, ...) are not worth the attack surface
MIN_SIDE = 256
MAX_LONG_SIDE = 3072     # larger photos are scaled down; the models work at about one megapixel
MAX_PIXELS = 60_000_000  # a small file can hold a huge image
MAX_ASPECT = 4.0
THUMB_SIDE = 640
ID_MAX = 40              # the worker's limit for an avatar id

# Chinese, Japanese and Korean text, including the full-width forms. Written as escapes so that this file holds none itself.
CJK = re.compile("[\u1100-\u11ff\u2e80-\u4dbf\u4e00-\u9fff\uac00-\ud7ff\uf900-\ufaff\uff00-\uffef\u3000-\u30ff\u31f0-\u31ff]")


class InputError(ValueError):
    """The input cannot be used; the message is written for the person who gave it."""


class Photo(NamedTuple):
    png: bytes
    thumbnail: bytes  # JPEG
    size: tuple


def clean_text(value, label, limit, required=False):
    """A title or description as one line: control and formatting characters removed, English only, at most `limit` characters."""
    kept = []
    for ch in value or "":
        kind = unicodedata.category(ch)
        if kind == "Cf":
            continue  # zero-width and bidirectional-override characters
        kept.append(" " if kind.startswith("C") or ch.isspace() else ch)
    text = " ".join("".join(kept).split())
    if CJK.search(text):
        raise InputError(f"{label} must be written in English.")
    if len(text) > limit:
        raise InputError(f"{label} is longer than {limit} characters.")
    if required and not text:
        raise InputError(f"{label} is required.")
    return text


def slugify(text, fallback="avatar"):
    """The avatar id for a name: lower-case ASCII letters, digits and single hyphens, at most ID_MAX characters."""
    ascii_text = unicodedata.normalize("NFKD", text or "").encode("ascii", "ignore").decode()
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_text.lower()).strip("-")[:ID_MAX].strip("-")
    return slug or fallback


def unique_id(base, taken):
    """`base`, or `base-2`, `base-3`, ... when that id is in use."""
    if base not in taken:
        return base
    n = 2
    while True:
        suffix = f"-{n}"
        candidate = base[: ID_MAX - len(suffix)].rstrip("-") + suffix
        if candidate not in taken:
            return candidate
        n += 1


def title_from_filename(filename):
    """A readable name for an upload that was given none: 'gulf_man-white.JPG' -> 'Gulf Man White'."""
    stem = re.sub(r"\.[A-Za-z0-9]{1,5}$", "", filename or "")
    words = re.sub(r"[^A-Za-z0-9]+", " ", stem).split()
    return " ".join(words).title()[:80] or "New Avatar"


def prepare_photo(data):
    """Check an uploaded photo and make the copy that is used from here on. Raises InputError with a message for the user."""
    try:
        image = Image.open(io.BytesIO(data), formats=FORMATS)
        width, height = image.size  # from the header: nothing has been decoded yet
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError, Image.DecompressionBombError):
        raise InputError("That file is not a photo the Studio can read. Use a PNG, JPEG or WebP image.") from None
    if width * height > MAX_PIXELS:
        raise InputError(f"The photo is larger than {MAX_PIXELS // 1_000_000} megapixels. Scale it down and try again.")
    try:
        image.load()
        image = ImageOps.exif_transpose(image)
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError):
        raise InputError("The photo is damaged or incomplete. Export it again and retry.") from None
    width, height = image.size
    if min(width, height) < MIN_SIDE:
        raise InputError(f"The photo is too small: its shorter side must be at least {MIN_SIDE} pixels.")
    if max(width, height) / min(width, height) > MAX_ASPECT:
        raise InputError("The photo is too narrow. Use a portrait or a full-body photo with a normal aspect ratio.")
    if image.mode in ("RGBA", "LA", "PA") or "transparency" in image.info:
        rgba = image.convert("RGBA")
        image = Image.alpha_composite(Image.new("RGBA", rgba.size, (255, 255, 255, 255)), rgba).convert("RGB")
    else:
        image = image.convert("RGB")
    if max(image.size) > MAX_LONG_SIDE:
        image.thumbnail((MAX_LONG_SIDE, MAX_LONG_SIDE), Image.LANCZOS)
    png = io.BytesIO()
    image.save(png, "PNG")
    thumb = image.copy()
    thumb.thumbnail((THUMB_SIDE, THUMB_SIDE), Image.LANCZOS)
    jpeg = io.BytesIO()
    thumb.save(jpeg, "JPEG", quality=85)
    return Photo(png.getvalue(), jpeg.getvalue(), image.size)
