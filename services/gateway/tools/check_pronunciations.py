#!/usr/bin/env python
"""Show how the avatar's voice will say a sentence, and name the words it cannot say.

Kokoro's text-to-phoneme step (misaki) only knows the words in its dictionaries. Its espeak fallback is GPL-3 and is not loaded,
so a product or place name it has never seen is dropped without a warning and the avatar leaves it out or mumbles. This runs the
same normalisation, the `pronunciations` table of kka.yaml and misaki that the live TTS handler uses, offline and without a GPU.

    docker run --rm --entrypoint python -v $PWD/services/gateway:/opt/kka-gw:ro -w /opt/oac kka-gateway:latest \\
        /opt/kka-gw/tools/check_pronunciations.py "Welcome to the HPE ProLiant lab in Lusail."
    ... check_pronunciations.py --file phrases.txt          one sentence per line ('-' reads standard input)
    ... check_pronunciations.py --config other.yaml "..."   default: <gateway>/config/kka.yaml, else /opt/oac/config/kka.yaml

A word that is dropped is listed under "not said". Fix it in kka.yaml under KokoroTTS.pronunciations, either as a respelling
(`Ezmeral: "Ez mer ul"`) or as exact phonemes in misaki's American set (`Lusail: "[Lusail](/lusˈAl/)"`), rebuild the gateway image
and run this again. Exit status is 1 when any sentence has a word that is not said, so a list of expected phrases can gate a release.
"""
import argparse
import os
import sys

GATEWAY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = ["/opt/oac/src", "/opt/oac/src/handlers", "/opt/kka-handlers", os.path.join(GATEWAY, "handlers")]

import yaml  # noqa: E402


def default_config() -> str:
    for path in (os.path.join(GATEWAY, "config", "kka.yaml"), "/opt/oac/config/kka.yaml"):
        if os.path.isfile(path):
            return path
    return "kka.yaml"


def read_sentences(args, parser) -> list:
    sentences = list(args.sentences)
    if args.file:
        source = sys.stdin if args.file == "-" else open(args.file, encoding="utf-8")
        with source:
            sentences += [line.strip() for line in source if line.strip()]
    if not sentences:
        parser.error("give a sentence, or --file")
    return sentences


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sentences", nargs="*", help="text to check")
    ap.add_argument("--file", help="text file with one sentence per line, or - for standard input")
    ap.add_argument("--config", default=default_config(), help="kka.yaml that holds the KokoroTTS voice and pronunciations")
    args = ap.parse_args(argv)
    sentences = read_sentences(args, ap)

    from kka.tts import kokoro_tts as tts

    tts._block_espeak()
    from kokoro import KPipeline

    with open(args.config, encoding="utf-8") as f:
        section = yaml.safe_load(f)["default"]["chat_engine"]["handler_configs"]["KokoroTTS"]
    voice = str(section.get("voice") or tts.KokoroTTSConfig().voice).split(",")[0].strip()
    patterns = tts.compile_pronunciations(section.get("pronunciations") or {})
    pipeline = KPipeline(lang_code=voice[0], model=False, repo_id=tts.KOKORO_REPO)
    print(f"voice {voice}, {len(patterns)} pronunciation overrides from {args.config}\n")

    missing, bad = [], 0
    for sentence in sentences:
        spoken = tts.apply_pronunciations(tts.normalise(sentence), patterns)
        phonemes, tokens = pipeline.g2p(spoken)
        dropped = [t.text for t in tokens if any(c.isalnum() for c in t.text) and (t.phonemes is None or "\u2753" in t.phonemes)]
        print(sentence)
        if spoken != sentence:
            print(f"  says      : {spoken}")
        print(f"  phonemes  : {phonemes}")
        if dropped:
            bad += 1
            print(f"  not said  : {', '.join(dropped)}")
        for word in dropped:
            if word.lower() not in (m.lower() for m in missing):
                missing.append(word)
        print()

    print(f"{len(sentences)} sentence(s) checked, {bad} with words that are not said" + (f": {', '.join(missing)}" if missing else ""))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
