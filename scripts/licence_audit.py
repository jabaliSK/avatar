#!/usr/bin/env python3
"""List the Python packages of an image whose licence is not plain MIT / BSD / Apache / ISC / PSF.

Runs inside an image (standard library only), so it audits what is actually installed:

    for s in gateway renderer bodygen; do docker run --rm -i --entrypoint python kka-$s:latest - < scripts/licence_audit.py; done

Exit status is 1 when a package is under a strong-copyleft (GPL, AGPL) or an unrecognised licence, which needs a human decision.
Proprietary NVIDIA wheels and weak copyleft (MPL, LGPL) are listed but do not fail the audit; docs and README say how they are used.
It reads package metadata only. System packages (apt) and model weights are not covered.
"""
import re
import sys
from importlib import metadata

PERMISSIVE = re.compile(r"\b(MIT|BSD|Apache|ISC|PSF|Python Software Foundation|Zlib|HPND|Unlicense|CC0|Blue ?Oak|0BSD|Public Domain|PIL)\b", re.I)


def licence_text(dist) -> str:
    md = dist.metadata
    parts = [md.get("License-Expression") or ""]
    text = md.get("License") or ""
    if text and len(text) < 120 and "\n" not in text:
        parts.append(text)
    parts += [c.split("::", 1)[1].strip() for c in md.get_all("Classifier") or [] if c.startswith("License ::")]
    return " | ".join(dict.fromkeys(p for p in parts if p)) or "UNKNOWN"


def classify(name: str, text: str) -> str:
    low = text.lower()
    if re.match(r"(nvidia-|cuda-|tensorrt)", name.lower()) or "proprietary" in low:
        return "proprietary"
    without_lgpl = re.sub(r"lgpl|lesser general public|library general public", "", low)
    if re.search(r"gpl|general public", without_lgpl):
        return "strong copyleft"
    if re.search(r"lgpl|lesser general public|library general public|mpl|mozilla", low):
        return "weak copyleft"
    return "permissive" if PERMISSIVE.search(text) else "unknown"


def main() -> int:
    groups: dict[str, list[str]] = {}
    total = 0
    for dist in sorted(metadata.distributions(), key=lambda d: (d.metadata["Name"] or "").lower()):
        name, text = dist.metadata["Name"] or "?", licence_text(dist)
        total += 1
        kind = classify(name, text)
        if kind != "permissive":
            groups.setdefault(kind, []).append(f"{name}=={dist.version}  ({text[:90]})")
    print(f"{total} Python packages, {total - sum(map(len, groups.values()))} permissive")
    for kind in ("strong copyleft", "unknown", "weak copyleft", "proprietary"):
        for line in groups.get(kind, []):
            print(f"  {kind:16} {line}")
    return 1 if groups.get("strong copyleft") or groups.get("unknown") else 0


if __name__ == "__main__":
    sys.exit(main())
