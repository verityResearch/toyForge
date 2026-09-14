"""Run the vendored official JSON Schema Test Suite (Draft 2020-12) through the
C validator, with a documented skip list for unsupported vocabularies.

The suite files under c/tests/jsonschema-suite/draft2020-12 are limited to the
keywords tf_jsonschema_validate implements. Individual cases that exercise
features still treated as pass-through annotations (regex pattern, non-local /
dynamic refs, unknown-keyword annotation collection, etc.) are skipped by the
SKIP table below and counted separately.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

SUITE = Path("c/tests/jsonschema-suite/draft2020-12")

# Case-level skips: {filename: [exact "<group> :: <test>" labels]}. Only one
# remains: Unicode-property (\p{...}) regex escapes, which need Unicode property
# tables the bounded regex engine does not carry. The metaschema-validation cases
# (defs.json + ref.json, which both $ref the Draft 2020-12 metaschema) now PASS
# via the vendored metaschema/vocabulary registry (METASCHEMA_DIR) + full
# $dynamicRef dynamic-scope resolution.
SKIP: dict[str, list[str]] = {
    "pattern.json": [
        "pattern with Unicode property escape requires unicode mode :: Digits do not match",
    ],
}


# The vendored Draft 2020-12 metaschema + vocabulary documents; passed as a
# registry so cases that $ref the metaschema resolve. Harmless for other cases
# (the docs only matter when referenced).
METASCHEMA_DIR = SUITE.parent / "metaschema"


def _c_valid(toyforge_c: Path, schema: object, data: object, tmp: Path) -> bool:
    (tmp / "s.json").write_text(json.dumps(schema))
    (tmp / "d.json").write_text(json.dumps(data))
    proc = subprocess.run(
        [
            str(toyforge_c),
            "jsonschema-validate",
            "--schema",
            str(tmp / "s.json"),
            "--instance",
            str(tmp / "d.json"),
            "--registry-dir",
            str(METASCHEMA_DIR),
        ],
        check=False,
        text=True,
        capture_output=True,
    )
    if proc.returncode not in (0, 1):
        raise RuntimeError(f"C error: {proc.stderr or proc.stdout}")
    return proc.returncode == 0


def _skipped(filename: str, label: str) -> bool:
    return any(s in label for s in SKIP.get(filename, []))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--toyforge-c", type=Path, default=Path("c/build/toyforge-c"))
    parser.add_argument("--show-failures", action="store_true")
    args = parser.parse_args()

    passed = skipped = 0
    failures: list[str] = []
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for path in sorted(SUITE.glob("*.json")):
            groups = json.loads(path.read_text())
            for group in groups:
                for test in group["tests"]:
                    label = f"{group['description']} :: {test['description']}"
                    if _skipped(path.name, label):
                        skipped += 1
                        continue
                    got = _c_valid(args.toyforge_c, group["schema"], test["data"], tmp)
                    if got == test["valid"]:
                        passed += 1
                    else:
                        failures.append(
                            f"{path.name}: {label} (expected {test['valid']}, got {got})"
                        )

    total = passed + len(failures)
    print(f"jsonschema suite: {passed}/{total} passed, {skipped} skipped")
    if failures:
        if args.show_failures:
            print("\n".join(failures))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
