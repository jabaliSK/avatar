"""Every phoneme override in kka.yaml must be spelled with symbols Kokoro knows.

Kokoro drops a symbol that is not in its vocabulary without any warning, which turns a product name into mumbling, so a typo in an
override only shows up when someone listens. The vocabulary is read from Kokoro's config.json (KOKORO_CONFIG, default
/models/Kokoro-82M/config.json); the test is skipped when that file is not mounted.
"""
import json
import os
import re
from pathlib import Path

import pytest
import yaml

CONFIG = Path(__file__).resolve().parents[1] / "config" / "kka.yaml"
VOCAB_FILE = Path(os.environ.get("KOKORO_CONFIG", "/models/Kokoro-82M/config.json"))
LINK = re.compile(r"\[([^\]]+)\]\(/([^)]*)/\)")


def _overrides() -> dict:
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    return cfg["default"]["chat_engine"]["handler_configs"]["KokoroTTS"]["pronunciations"]


def test_overrides_are_present():
    overrides = _overrides()
    assert overrides["HPE"] == "H P E"
    assert any(LINK.search(v) for v in overrides.values())


def test_link_text_is_the_word_it_replaces():
    for word, spoken in _overrides().items():
        for text, _ in LINK.findall(spoken):
            assert text.lower() == word.lower(), f"{word!r} is rewritten as a link for {text!r}"


@pytest.mark.skipif(not VOCAB_FILE.exists(), reason="Kokoro config.json is not mounted")
def test_phonemes_use_only_symbols_in_the_kokoro_vocabulary():
    vocab = set(json.loads(VOCAB_FILE.read_text(encoding="utf-8"))["vocab"])
    for word, spoken in _overrides().items():
        for _, phonemes in LINK.findall(spoken):
            unknown = sorted({c for c in phonemes if c not in vocab})
            assert not unknown, f"{word!r}: /{phonemes}/ has symbols Kokoro does not know: {unknown}"
