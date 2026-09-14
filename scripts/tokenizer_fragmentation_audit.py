"""Audit how the chosen tokenizer fragments JSON-RPC method/param names.

Library API (testable without `train` extra):
    audit_strings(tokenizer, list_of_strings) -> dict

CLI mode (requires transformers; `--extra train`):
    uv run python scripts/tokenizer_fragmentation_audit.py --model Qwen/Qwen3-8B-Instruct
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def audit_strings(tokenizer: Any, strings: list[str]) -> dict[str, dict]:
    """For each string, return token count, ids, decoded surface pieces, and a
    `fragmented` flag (True iff token_count > 1)."""
    out: dict[str, dict] = {}
    for s in strings:
        encoded = tokenizer(s, add_special_tokens=False)
        ids = list(encoded["input_ids"])
        surface_pieces = [tokenizer.decode([i]) for i in ids]
        out[s] = {
            "token_count": len(ids),
            "token_ids": ids,
            "surface_pieces": surface_pieces,
            "fragmented": len(ids) > 1,
        }
    return out


def _strings_from_schemas(schemas_dir: Path) -> list[str]:
    """Collect every method name + every required param name + structural tokens."""
    methods_doc = json.loads((schemas_dir / "jsonrpc-methods.json").read_text())
    names: list[str] = []
    for method_name, spec in methods_doc["methods"].items():
        names.append(method_name)
        for param in (spec.get("params") or {}).get("properties", {}):
            names.append(param)
    # Structural fragments worth knowing
    names.extend(["{", "}", '":', '","', '"method"', '"params"', '"jsonrpc"', '"id"'])
    return sorted(set(names))


def render_markdown(audit: dict, model_id: str) -> str:
    out = ["# Tokenizer fragmentation audit\n", f"**Tokenizer:** `{model_id}`\n"]
    fragmented = sorted(
        ((s, info) for s, info in audit.items() if info["fragmented"]),
        key=lambda kv: -kv[1]["token_count"],
    )
    if fragmented:
        out.append(f"## Fragmented strings ({len(fragmented)})\n")
        out.append("| String | Token count | Pieces |")
        out.append("|---|---|---|")
        for s, info in fragmented:
            pieces = " ╊ ".join(repr(p) for p in info["surface_pieces"])
            out.append(f"| `{s}` | {info['token_count']} | {pieces} |")
    else:
        out.append("_All audited strings tokenize to a single token._\n")
    out.append("")
    out.append("## Single-token strings")
    for s, info in audit.items():
        if not info["fragmented"]:
            out.append(f"- `{s}` (id={info['token_ids'][0]})")
    out.append("")
    out.append("## Interpretation")
    out.append("")
    n_fragmented = sum(1 for info in audit.values() if info["fragmented"])
    n_total = len(audit)
    pct = (n_fragmented / n_total * 100) if n_total else 0.0
    out.append(
        f"**{n_fragmented} of {n_total} audited strings ({pct:.0f}%) tokenize into 2+ tokens.**"
    )
    out.append("")
    if n_fragmented == 0:
        out.append(
            "No fragmentation detected — every API surface name is a single token. "
            "This is the best-case scenario for fast, accurate generation."
        )
    elif pct < 25:
        out.append(
            "Low fragmentation. The model should learn these multi-token names "
            "reliably during SFT cold-start. No action needed."
        )
    elif pct < 50:
        out.append(
            "Moderate fragmentation. Some param names take 3+ tokens; consider "
            "whether they could be shortened in a future schema revision. If "
            "Phase 1 SFT struggles on `schema=0` or `method_known=0` failures, "
            "the most-fragmented strings (top of the table above) are likely "
            "culprits."
        )
    else:
        out.append(
            "High fragmentation. Most API surface names span multiple tokens; "
            "the model will need substantially more SFT data than the typical "
            "~750 expansions to reliably emit them. Consider: (1) renaming the "
            "most-fragmented methods, (2) using a tokenizer with extended "
            "vocabulary for this domain, or (3) constrained decoding (Outlines/"
            "XGrammar) to guarantee surface form correctness regardless of "
            "tokenizer."
        )
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--schemas-dir", default="schemas", type=Path)
    ap.add_argument("--output", default=None, type=Path)
    args = ap.parse_args()

    from transformers import AutoTokenizer  # local import — requires train extra

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    strings = _strings_from_schemas(args.schemas_dir)
    audit = audit_strings(tokenizer, strings)
    md = render_markdown(audit, args.model)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(md)
        print(f"wrote {args.output}")
    else:
        print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
