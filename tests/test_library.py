import json
import os
import re
from pathlib import Path

import pytest
from PIL import Image, UnidentifiedImageError

from photostage import library

KEY = "0123456789abcdef0123456789abcdef"
DEFAULTS = {option.name: option.default for option in library.OPTIONS}
NOTHING = {"name": "", "created": "", "options": {}}
OLD, NEWER = 1_600_000_000, 1_700_000_000  # 2020-09-13T12:26:40Z and 2023-11-14T22:13:20Z


def make(data, key=KEY, *, photo=None, meta=None, body=None, rig=None):
    """A finished avatar folder: what the renderer publishes, and optionally a photo (its size), meta.json and body.json."""
    avatar = data / key
    avatar.mkdir()
    for name in library.PUBLISHED:
        (avatar / name).write_bytes(b"x")
    (avatar / "rig.json").write_text(json.dumps(rig if rig is not None else {"size": [900, 600]}))
    if photo:
        Image.new("RGB", photo, (90, 120, 150)).save(avatar / "photo.png")
    if meta is not None:
        (avatar / "meta.json").write_text(json.dumps(meta))
    if body is not None:
        (avatar / "body.json").write_text(json.dumps(body))
    return avatar


def touch(path, seconds):
    os.utime(path, (seconds, seconds))


def test_the_options_are_a_consistent_table():
    names = [option.name for option in library.OPTIONS]
    assert len(set(names)) == len(names)
    for option in library.OPTIONS:
        assert library.check_option(option, option.default) == option.default, option.name
        assert not option.needs or option.needs in names, option.name
    assert json.loads(json.dumps(library.schema())) == library.schema()


@pytest.mark.parametrize("name, value, expected", [
    ("chat", False, False), ("mode", "speak", "speak"), ("motion", 0, 0.0), ("motion", 0.5, 0.5), ("motion", 2, 2.0),
])
def test_valid_option_values_are_accepted(name, value, expected):
    result = library.check_option(library.DEFINITIONS[name], value)
    assert result == expected and type(result) is type(expected)


@pytest.mark.parametrize("name, value, accepts", [
    ("chat", "yes", "true or false"), ("chat", 1, "true or false"), ("chat", None, "true or false"),
    ("mode", "talk", "one of chat, speak"), ("mode", True, "one of chat, speak"),
    ("motion", "1", "a number from 0 to 2"), ("motion", True, "a number from 0 to 2"), ("motion", -0.1, "a number from 0 to 2"),
    ("motion", 2.01, "a number from 0 to 2"), ("motion", float("nan"), "a number from 0 to 2"),
    ("motion", float("inf"), "a number from 0 to 2"),
])
def test_an_invalid_option_value_is_refused_saying_what_the_option_accepts(name, value, accepts):
    with pytest.raises(ValueError, match=f"^{name} must be {accepts}\\.$"):
        library.check_option(library.DEFINITIONS[name], value)


def test_options_asked_for_are_checked_as_a_whole():
    assert library.clean_options({"chat": False, "motion": 1}) == {"chat": False, "motion": 1.0}
    assert library.clean_options({}) == {}
    wrong = {"zoom": 1, "chat": True, "colour": "red", "volume": 3, "audio": 1, "theme": "dark", "border": 2}
    with pytest.raises(ValueError, match=r"^Unknown option: audio, border, colour, theme, volume, zoom\.$"):
        library.clean_options(wrong)
    with pytest.raises(ValueError, match="options must be an object"):
        library.clean_options(["chat"])
    with pytest.raises(ValueError, match="motion must be"):
        library.clean_options({"chat": True, "motion": 9})


def test_every_option_has_a_value_and_what_a_damaged_file_holds_is_ignored():
    assert library.resolve({}) == DEFAULTS
    for damaged in (None, [], "chat", 7):
        assert library.resolve(damaged) == DEFAULTS
    saved = {"chat": False, "motion": 1.5, "mode": "nonsense", "reply": "no", "ghost": True}
    assert library.resolve(saved) == {**DEFAULTS, "chat": False, "motion": 1.5}


def test_only_what_differs_from_the_defaults_is_saved():
    chosen = {**DEFAULTS, "chat": False, "motion": 2.0}
    assert library.overrides(chosen) == {"chat": False, "motion": 2.0}
    assert library.resolve(library.overrides(chosen)) == chosen
    assert library.overrides(DEFAULTS) == {}
    assert library.overrides({"motion": 1.0}) == {}


@pytest.mark.parametrize("text, expected", [
    ("Dr. Rao", "Dr. Rao"),
    ("  Dr.\t Rao \n", "Dr. Rao"),
    ("line one\nline two\r\n\x00three\x7f", "line one line two three"),
    ("Zoë 田中", "Zoë 田中"),
    ("", ""), ("   ", ""), ("\x00\x01", ""), (None, ""), (5, ""),
    ("x" * 80, "x" * 60),
    (" ".join(["word"] * 20), " ".join(["word"] * 12)),
])
def test_a_name_is_one_readable_line_of_limited_length(text, expected):
    assert library.clean_name(text) == expected


@pytest.mark.parametrize("filename, expected", [
    ("Rashid 2.0.PNG", "Rashid 2.0"), ("../../etc/passwd.png", "passwd"), ("a/b/c.jpg", "c"), ("", ""), (None, ""),
])
def test_an_uploads_name_is_its_file_name_without_folders_or_extension(filename, expected):
    assert library.name_from_upload(filename) == expected


def test_default_names_and_stamps():
    assert library.default_name(KEY) == "Avatar 012345"
    assert library.stamp(0) == "1970-01-01T00:00:00Z"
    assert library.stamp(OLD) == "2020-09-13T12:26:40Z"
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", library.stamp())


def test_only_a_finished_avatar_has_a_folder(tmp_path):
    avatar = make(tmp_path)
    assert library.folder(tmp_path, KEY) == avatar
    for name in library.PUBLISHED:
        (avatar / name).rename(avatar / (name + ".part"))
        assert library.folder(tmp_path, KEY) is None, name
        (avatar / (name + ".part")).rename(avatar / name)
    assert library.folder(tmp_path, KEY) == avatar


@pytest.mark.parametrize("key", ["", "abc", "A" * 32, KEY.upper(), "g" * 32, "0" * 31, "0" * 33, "../" + "0" * 29, KEY + "\n"])
def test_a_key_that_is_not_one_is_never_a_folder_even_where_a_folder_has_its_name(tmp_path, key):
    make(tmp_path)
    if key and "/" not in key:
        make(tmp_path, key)
    assert library.folder(tmp_path, key) is None


def test_meta_is_read_whatever_state_it_is_in(tmp_path):
    avatar = make(tmp_path)
    assert library.read_meta(avatar) == NOTHING
    for content in ("{not json", "[1, 2]", '"text"', "", b"\xff\xfe\x00"):
        (avatar / "meta.json").write_bytes(content if isinstance(content, bytes) else content.encode())
        assert library.read_meta(avatar) == NOTHING, content
    (avatar / "meta.json").write_text(json.dumps({"name": " A\nB ", "created": 5, "options": ["x"]}))
    assert library.read_meta(avatar) == {"name": "A B", "created": "", "options": {}}


def test_meta_is_replaced_whole_and_keeps_unicode_readable(tmp_path):
    avatar = make(tmp_path)
    library.write_meta(avatar, {"name": "Zoë", "created": "then", "options": {}})
    assert "Zoë" in (avatar / "meta.json").read_text(encoding="utf-8")
    library.write_meta(avatar, {"name": "Two", "created": "then", "options": {"chat": False}})
    assert library.read_meta(avatar) == {"name": "Two", "created": "then", "options": {"chat": False}}
    assert [path.name for path in avatar.iterdir() if "meta" in path.name] == ["meta.json"]


def test_an_update_renames_and_changes_only_the_options_it_is_given(tmp_path):
    created = "2024-01-01T00:00:00Z"
    avatar = make(tmp_path, meta={"name": "Old", "created": created, "options": {"chat": False}})
    library.update(avatar, name="  New \n name ")
    assert library.read_meta(avatar) == {"name": "New name", "created": created, "options": {"chat": False}}
    library.update(avatar, options={"motion": 1.5})
    assert library.read_meta(avatar)["options"] == {"chat": False, "motion": 1.5}
    library.update(avatar, options={"chat": True})  # the default again, so no longer saved
    assert library.read_meta(avatar) == {"name": "New name", "created": created, "options": {"motion": 1.5}}
    library.update(avatar)
    assert library.read_meta(avatar) == {"name": "New name", "created": created, "options": {"motion": 1.5}}


@pytest.mark.parametrize("name, options, message", [
    ("", None, "Give the avatar a name."),
    ("   ", None, "Give the avatar a name."),
    ("\x00\n", None, "Give the avatar a name."),
    ("Fine", {"motion": 3}, "motion must be"),
    ("Fine", {"colour": "red"}, "Unknown option"),
    ("Fine", [1], "options must be an object"),
])
def test_an_invalid_update_is_refused_before_anything_is_written(tmp_path, name, options, message):
    avatar = make(tmp_path, meta={"name": "Old", "created": "2024-01-01T00:00:00Z", "options": {"chat": False}})
    before = (avatar / "meta.json").read_bytes()
    with pytest.raises(ValueError, match=message):
        library.update(avatar, name, options)
    assert (avatar / "meta.json").read_bytes() == before


def test_the_age_of_an_avatar_without_meta_is_that_of_its_photo_or_else_its_rig(tmp_path):
    avatar = make(tmp_path, photo=(10, 10))
    touch(avatar / "photo.png", OLD)
    touch(avatar / "rig.json", NEWER)
    assert library.made_at(avatar) == "2020-09-13T12:26:40Z"
    (avatar / "photo.png").unlink()
    assert library.made_at(avatar) == "2023-11-14T22:13:20Z"
    (avatar / "rig.json").unlink()
    assert library.made_at(avatar) == ""


def test_an_avatar_is_described_for_the_console(tmp_path):
    saved = {"chat": False, "motion": 9, "bogus": 1}
    avatar = make(tmp_path, meta={"name": "Dr. Rao", "created": "2024-05-01T10:00:00Z", "options": saved},
                  body={"version": 1, "sway": 17.7, "hands": [{}, {}]})
    assert library.describe(avatar) == {
        "key": KEY, "name": "Dr. Rao", "created": "2024-05-01T10:00:00Z", "size": [900, 600],
        "url": f"/a/{KEY}", "thumb": f"/assets/{KEY}/thumb.jpg", "motion": {"hands": 2, "sway": True}, "build": {"state": "ready"},
        "options": {**DEFAULTS, "chat": False},  # a saved value that is not valid is as good as not saved
    }


def test_an_avatar_made_before_meta_and_body_existed_is_still_described(tmp_path):
    avatar = make(tmp_path, photo=(10, 10))
    touch(avatar / "photo.png", OLD)
    shown = library.describe(avatar)
    assert shown["name"] == "Avatar 012345" and shown["created"] == "2020-09-13T12:26:40Z"
    assert shown["motion"] == {"hands": 0, "sway": False} and shown["options"] == DEFAULTS
    for body in ('{"hands": "none", "sway": 0}', "[1]", "broken"):
        (avatar / "body.json").write_text(body)
        assert library.describe(avatar)["motion"] == {"hands": 0, "sway": False}, body
    library.update(avatar, name="Renamed")  # meta.json appears, and the avatar keeps the age it had
    assert library.describe(avatar)["created"] == "2020-09-13T12:26:40Z"


@pytest.mark.parametrize("reverse", [False, True], ids=["found in one order", "found in the other"])
def test_the_listing_is_newest_first_and_leaves_out_what_is_not_a_finished_avatar(tmp_path, monkeypatch, reverse):
    found = Path.iterdir  # the folder's own order is arbitrary; the listing must not depend on it
    monkeypatch.setattr(Path, "iterdir", lambda self: iter(sorted(found(self), reverse=reverse)))
    first, second, third, aged = "a" * 32, "b" * 32, "c" * 32, "d" * 32
    make(tmp_path, first, meta={"created": "2024-01-01T00:00:00Z"})
    make(tmp_path, second, meta={"created": "2024-03-01T00:00:00Z"})
    make(tmp_path, third, meta={"created": "2024-03-01T00:00:00Z"})  # the same second: the key decides
    touch(make(tmp_path, aged, photo=(10, 10)) / "photo.png", NEWER)  # no meta: its photo's age
    (tmp_path / ("e" * 32)).mkdir()  # an avatar still being made
    (tmp_path / "tmpab12cd").mkdir()
    (tmp_path / "stray.txt").write_text("x")
    assert [item["key"] for item in library.listing(tmp_path)] == [third, second, first, aged]
    assert library.listing(tmp_path)[0]["name"] == "Avatar cccccc"


def test_an_avatar_deleted_while_the_list_is_made_is_skipped(tmp_path, monkeypatch):
    make(tmp_path, "a" * 32)
    make(tmp_path, "b" * 32)
    real = library.describe

    def flaky(avatar):
        if avatar.name == "a" * 32:
            raise FileNotFoundError(avatar)
        return real(avatar)

    monkeypatch.setattr(library, "describe", flaky)
    assert [item["key"] for item in library.listing(tmp_path)] == ["b" * 32]


def test_the_thumbnail_is_made_once_small_and_in_colour(tmp_path):
    avatar = make(tmp_path, photo=(1536, 2752))
    path = library.thumbnail(avatar)
    assert path == avatar / "thumb.jpg" and not (avatar / "thumb.jpg.tmp").exists()
    with Image.open(path) as thumb:
        assert thumb.format == "JPEG" and thumb.mode == "RGB"
        assert max(thumb.size) == library.THUMB_SIDE and abs(thumb.width / thumb.height - 1536 / 2752) < 0.01
    path.write_bytes(b"kept")
    assert library.thumbnail(avatar).read_bytes() == b"kept"


def test_a_small_or_transparent_photo_gets_a_thumbnail_without_being_enlarged(tmp_path):
    avatar = make(tmp_path)
    Image.new("RGBA", (120, 80), (200, 30, 30, 0)).save(avatar / "photo.png")
    with Image.open(library.thumbnail(avatar)) as thumb:
        assert thumb.size == (120, 80) and thumb.mode == "RGB"


def test_an_unreadable_photo_has_no_thumbnail_and_leaves_nothing_behind(tmp_path):
    avatar = make(tmp_path)
    with pytest.raises(FileNotFoundError):
        library.thumbnail(avatar)
    (avatar / "photo.png").write_bytes(b"not an image")
    with pytest.raises(UnidentifiedImageError):
        library.thumbnail(avatar)
    assert not (avatar / "thumb.jpg").exists() and not (avatar / "thumb.jpg.tmp").exists()


def test_removing_an_avatar_deletes_everything_in_it(tmp_path):
    avatar = make(tmp_path, photo=(10, 10))
    (avatar / "hand-0.png").write_bytes(b"x")
    library.remove(avatar)
    assert not avatar.exists() and library.folder(tmp_path, KEY) is None
