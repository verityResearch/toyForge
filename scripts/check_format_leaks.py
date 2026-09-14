"""Audit gate: catch Python-templating escape leaks in C source.

The build's `-Wformat -Werror` already catches malformed `printf`-family format
strings, so the *only* class it can't see is escape artifacts in **non-format
contexts** — e.g. a `%` over-escaped to `%%` (a Python `%`-formatting leak)
sitting in a comment or a non-`printf` string, where it's just wrong text. This
gate flags exactly that class.

Rule for `%%` (the demonstrated leak): legitimate only inside a `printf`-family
format string; flagged when it appears in a comment or on a line with no
`printf`-family call. Extensible: add patterns to LEAK_RULES.

Usage:
    python scripts/check_format_leaks.py [PATH ...]   # default: c/src c/include c/tests
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

DEFAULT_ROOTS = ("c/src", "c/include", "c/tests")
SRC_EXTS = (".c", ".h")

# printf-family calls whose first string arg is a real format string (where
# `%%` is a legitimate literal percent).
_PRINTF_RE = re.compile(r"\b(f?printf|s(?:n)?printf|d?printf|v(?:f|s|sn)?printf)\s*\(")


def _comment_mask(line: str, in_block: bool) -> tuple[list[bool], bool]:
    """Per-character mask: True where the char is inside a comment. Tracks
    multi-line /* */ via in_block; handles // and inline /* */. Ignores comment
    markers inside string literals."""
    mask = [False] * len(line)
    i = 0
    in_str = False
    str_ch = ""
    while i < len(line):
        c = line[i]
        nxt = line[i + 1] if i + 1 < len(line) else ""
        if in_block:
            mask[i] = True
            if c == "*" and nxt == "/":
                mask[i + 1] = True
                i += 2
                in_block = False
                continue
            i += 1
            continue
        if in_str:
            if c == "\\":
                i += 2
                continue
            if c == str_ch:
                in_str = False
            i += 1
            continue
        if c in ('"', "'"):
            in_str = True
            str_ch = c
            i += 1
            continue
        if c == "/" and nxt == "/":
            for j in range(i, len(line)):
                mask[j] = True
            break
        if c == "/" and nxt == "*":
            mask[i] = mask[i + 1] = True
            i += 2
            in_block = True
            continue
        i += 1
    return mask, in_block


def _check_double_percent(line: str, mask: list[bool]) -> list[str]:
    """Flag each `%%` that is in a comment, or in code with no printf-family
    call on the line (so it's a comment/fputs/bare-literal leak, not a real
    format string)."""
    hits = []
    has_printf = bool(_PRINTF_RE.search(line))
    for m in re.finditer(r"%%", line):
        pos = m.start()
        in_comment = mask[pos]
        if in_comment:
            hits.append("`%%` in a comment (should be a single `%`?)")
        elif not has_printf:
            hits.append("`%%` outside any printf-family call (non-format `%%`)")
    return hits


# Extensible: (name, per-line-checker(line, comment_mask) -> list[str]).
LEAK_RULES = [("double_percent", _check_double_percent)]


def scan_file(path: Path) -> list[tuple[int, str]]:
    findings: list[tuple[int, str]] = []
    in_block = False
    for lineno, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
        mask, in_block = _comment_mask(line, in_block)
        for _name, checker in LEAK_RULES:
            for msg in checker(line, mask):
                findings.append((lineno, msg))
    return findings


def iter_sources(paths: list[str]) -> list[Path]:
    out: list[Path] = []
    for p in paths:
        pp = Path(p)
        if pp.is_dir():
            for ext in SRC_EXTS:
                out.extend(pp.rglob(f"*{ext}"))
        elif pp.suffix in SRC_EXTS:
            out.append(pp)
    return sorted(set(out))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="*", default=list(DEFAULT_ROOTS))
    args = parser.parse_args()
    paths = args.paths or list(DEFAULT_ROOTS)

    total = 0
    files = iter_sources(paths)
    for f in files:
        for lineno, msg in scan_file(f):
            print(f"{f}:{lineno}: {msg}")
            total += 1
    if total:
        print(f"\nformat-leak check: {total} leak(s) found")
        return 1
    print(f"format-leak check: clean ({len(files)} files scanned)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
