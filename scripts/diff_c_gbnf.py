"""Differential: C `build-jsonrpc-gbnf` vs the Python GBNF builder.

The JSON-RPC GBNF grammar is the decode-time *constraint* used by
`llamacpp-eval` (`--grammar-path schemas/jsonrpc.gbnf`); the verifier is the
*reward*. The grammar is generated from `schemas/jsonrpc-methods.json` by both
`toyforge.llamacpp.build_jsonrpc_gbnf` (Python) and `write_jsonrpc_gbnf` (C),
and the checked-in `schemas/jsonrpc.gbnf` is the file eval actually consumes.

This gate asserts all three are byte-identical, so the strict-C grammar
generator can never drift from the Python oracle or from the on-disk grammar
without the cycle/CI catching it.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from toyforge.llamacpp import build_jsonrpc_gbnf

SCHEMAS_DIR = Path("schemas")
CHECKED_IN = SCHEMAS_DIR / "jsonrpc.gbnf"


def _c_gbnf(toyforge_c: Path) -> str:
    proc = subprocess.run(
        [str(toyforge_c), "build-jsonrpc-gbnf", "--schemas-dir", str(SCHEMAS_DIR)],
        check=True,
        text=True,
        capture_output=True,
    )
    return proc.stdout


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--toyforge-c", type=Path, default=Path("c/build/toyforge-c"))
    args = parser.parse_args()

    c_gbnf = _c_gbnf(args.toyforge_c)
    py_gbnf = build_jsonrpc_gbnf(SCHEMAS_DIR)
    on_disk = CHECKED_IN.read_text()

    failures: list[str] = []
    if c_gbnf != py_gbnf:
        failures.append("C build-jsonrpc-gbnf output differs from Python build_jsonrpc_gbnf")
    if c_gbnf != on_disk:
        failures.append(f"C build-jsonrpc-gbnf output differs from checked-in {CHECKED_IN}")
    if py_gbnf != on_disk:
        failures.append(f"Python build_jsonrpc_gbnf output differs from checked-in {CHECKED_IN}")

    if failures:
        print("\n".join(failures))
        # Show the first divergence to aid debugging.
        for label, got in (("python", py_gbnf), ("checked-in", on_disk)):
            if c_gbnf != got:
                c_lines, g_lines = c_gbnf.splitlines(), got.splitlines()
                for i in range(max(len(c_lines), len(g_lines))):
                    cl = c_lines[i] if i < len(c_lines) else "<EOF>"
                    gl = g_lines[i] if i < len(g_lines) else "<EOF>"
                    if cl != gl:
                        print(f"  first diff vs {label} at line {i + 1}:")
                        print(f"    C : {cl!r}")
                        print(f"    {label} : {gl!r}")
                        break
        return 1

    line_count = py_gbnf.count("\n")
    print(f"build-jsonrpc-gbnf parity: C == Python == {CHECKED_IN} ({line_count} grammar lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
