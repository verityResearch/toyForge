"""Run a sweep of training configs from a sweeps/{name}.yaml spec.

Usage:
    uv run python scripts/sweep.py sweeps/phase3-lora-rank.yaml

The spec must declare:
    base: path/to/base-config.yaml      # used as `extends:` value in generated configs
    axes:
        lora.r: [8, 16, 32, 64]
        seed: [42, 7, 123]

For each combination, materializes train_configs/.generated/{spec_stem}-{slug}.yaml
with `extends: <base>` plus the overrides, then invokes `just train config=...`.

The `base` value is written verbatim as the `extends:` path in each generated config,
so it must be a path *relative to* `train_configs/.generated/` (the directory where
generated configs live).  For a base config at `train_configs/phase1-sft.yaml`, use:

    base: ../phase1-sft.yaml

The dotted-key axis names are unflattened into nested YAML structure:
    `lora.r: 64` -> `lora: {r: 64}`
"""

from __future__ import annotations

import argparse
import itertools
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

_SLUG_SAFE = re.compile(r"[^A-Za-z0-9_-]")


def _unflatten(flat: dict[str, Any]) -> dict[str, Any]:
    """{'lora.r': 64, 'lora.alpha': 128} -> {'lora': {'r': 64, 'alpha': 128}}"""
    out: dict[str, Any] = {}
    for dotted_key, value in flat.items():
        parts = dotted_key.split(".")
        d = out
        for p in parts[:-1]:
            d = d.setdefault(p, {})
            if not isinstance(d, dict):
                raise ValueError(
                    f"axis path collision at {dotted_key!r}: existing value is not a dict"
                )
        # Guard against the reverse order too: a dotted key already filled
        # parts[-1] with a nested dict, and we're about to overwrite it.
        if parts[-1] in d and isinstance(d[parts[-1]], dict):
            raise ValueError(f"axis path collision at {dotted_key!r}: would overwrite nested dict")
        d[parts[-1]] = value
    return out


def _slug(overrides: dict[str, Any]) -> str:
    """Produce a filesystem-safe slug from override values, e.g. 'lora_r64-seed42'.

    Uses the full dotted axis name (with '.' replaced by '_') so axes that share
    a key tail (e.g. 'lora.r' and 'schedule.r') produce distinct slugs and never
    silently overwrite each other's generated config files.

    NOTE: this format differs from the pre-fix 'r64-seed42' style (leaf-only).
    Existing sweep configs generated before this fix will not be overwritten — they
    simply have a different filename.  No users have shipped configs in production.
    """
    parts = []
    for name, v in overrides.items():
        key_safe = name.replace(".", "_")
        # Replace any non-alphanumeric (preserving _ and -) with _ to ensure
        # the slug is safe on FAT/NTFS as well as POSIX filesystems.
        v_str = _SLUG_SAFE.sub("_", str(v))
        parts.append(f"{key_safe}{v_str}")
    return "-".join(parts)


def generate_sweep_configs(
    spec_path: Path,
    out_dir: Path,
) -> list[Path]:
    """Materialize all sweep config files. Returns the list of generated paths."""
    spec = yaml.safe_load(spec_path.read_text()) or {}
    base = spec.get("base")
    axes = spec.get("axes") or {}

    if not base:
        raise ValueError(f"sweep spec {spec_path} missing required 'base' key")
    if not isinstance(axes, dict) or not axes:
        raise ValueError(f"sweep spec {spec_path} requires non-empty 'axes' dict")

    axis_names = list(axes.keys())
    axis_values = [axes[name] for name in axis_names]

    out_dir.mkdir(parents=True, exist_ok=True)

    generated: list[Path] = []
    for combo in itertools.product(*axis_values):
        overrides = dict(zip(axis_names, combo, strict=True))
        slug = _slug(overrides)
        config_path = out_dir / f"{spec_path.stem}-{slug}.yaml"
        doc = {"extends": base, **_unflatten(overrides)}
        config_path.write_text(yaml.safe_dump(doc, sort_keys=False))
        generated.append(config_path)

    return generated


def run_sweep(spec_path: Path, dry_run: bool = False) -> tuple[int, bool]:
    """Run a sweep. Generates configs, then invokes `just train config=...` for each.

    Returns (completed_count, failed). `failed=True` means at least one config
    exited non-zero, regardless of how many succeeded before it.
    """
    out_dir = Path("train_configs") / ".generated"
    configs = generate_sweep_configs(spec_path, out_dir)

    print(f"Generated {len(configs)} configs in {out_dir}", flush=True)

    if dry_run:
        for c in configs:
            print(f"  would run: just train config={c}", flush=True)
        return len(configs), False

    completed = 0
    for c in configs:
        print(f"\n=== Running: {c} ===", flush=True)
        result = subprocess.run(["just", "train", f"config={c}"], check=False)
        if result.returncode != 0:
            print(f"FAILED: {c} (returncode={result.returncode})", flush=True)
            return completed, True
        completed += 1
    return completed, False


def main() -> int:
    ap = argparse.ArgumentParser(description="Run a sweep from sweeps/{name}.yaml")
    ap.add_argument("spec", type=Path, help="Path to sweep spec YAML")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate configs but don't invoke just train",
    )
    args = ap.parse_args()
    if not args.spec.exists():
        print(f"sweep spec not found: {args.spec}", file=sys.stderr)
        return 1
    completed, failed = run_sweep(args.spec, dry_run=args.dry_run)
    if failed:
        return 1
    return 0 if completed > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
