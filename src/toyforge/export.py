"""Merge a trained LoRA adapter into its base and export a single GGUF for the
strict-C llama.cpp runtime — the deploy seam between toyForge (PyTorch/PEFT) and
the strict-C llama.cpp port, which loads a *merged* GGUF only (no runtime adapter).

Pipeline:  PEFT ``merge_and_unload`` → HF ``save_pretrained`` →
``convert_hf_to_gguf.py`` → (optional) ``llama-quantize``.

The strict-C runtime has no runtime-LoRA path, so the adapter must be fused here, offline,
before export. ``base_model`` must be an HF-loadable model — a hub id
(``Qwen/Qwen3-8B-Instruct``) or a local HF directory. Bridging a custom-arch base
into HF format is a separate step; this module assumes the merged model is
HF-format llama-arch, which is what ``convert_hf_to_gguf.py`` consumes.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from toyforge._console import console

# Upstream converter lives in the sibling llama.cpp checkout; override via --convert-script.
DEFAULT_CONVERT_SCRIPT = Path.home() / "llama.cpp" / "convert_hf_to_gguf.py"


@dataclass(frozen=True)
class MergeExportResult:
    """Artifacts produced by :func:`merge_export`."""

    merged_dir: Path
    gguf_path: Path
    quantized_path: Path | None
    provenance_path: Path


def _git_sha() -> str:
    """Short toyForge HEAD sha for provenance (empty if unavailable)."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def _adapter_fingerprint(adapter_dir: Path) -> str:
    """sha256 of the adapter's config — cheap identity of the LoRA, no weight hashing."""
    cfg = Path(adapter_dir) / "adapter_config.json"
    if not cfg.exists():
        return ""
    return hashlib.sha256(cfg.read_bytes()).hexdigest()


def merge_adapter(
    base_model: str,
    adapter_dir: Path,
    out_dir: Path,
    *,
    dtype: str = "bfloat16",
    attn_impl: str = "sdpa",
) -> Path:
    """Fuse the PEFT adapter at ``adapter_dir`` into ``base_model`` and save the
    merged model + tokenizer to ``out_dir`` (returns ``out_dir``).

    Loaded on CPU (``device_map=None``) so the merge + save are unsharded and
    deterministic — this is a one-time offline export, not a hot path. The base +
    adapter load mirrors ``eval/runner.py`` so a merged-then-exported model matches
    what eval graded.
    """
    from peft import PeftModel  # noqa: PLC0415
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

    adapter_dir = Path(adapter_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Tokenizer from the adapter dir (it carries any added tokens / chat template).
    tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=dtype,
        device_map=None,  # CPU, unsharded — clean merge + save for offline export
        attn_implementation=attn_impl,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, str(adapter_dir))
    merged = model.merge_and_unload()
    merged.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    console.print(f"[green]Merged adapter into base:[/green] {out_dir}")
    return out_dir


def _convert_cmd(
    merged_dir: Path, out_path: Path, convert_script: Path, outtype: str, python_bin: str
) -> list[str]:
    """Build the ``convert_hf_to_gguf.py`` argv (pure — unit-testable)."""
    return [
        python_bin,
        str(convert_script),
        str(merged_dir),
        "--outfile",
        str(out_path),
        "--outtype",
        outtype,
    ]


def convert_to_gguf(
    merged_dir: Path,
    out_path: Path,
    *,
    convert_script: Path = DEFAULT_CONVERT_SCRIPT,
    outtype: str = "f16",
    python_bin: str = "python3",
) -> Path:
    """Shell to llama.cpp's ``convert_hf_to_gguf.py`` → a GGUF (arch=llama).

    The converter brings its own deps (``gguf``); run it from an environment that
    has them (commonly the llama.cpp checkout's own venv via ``--python-bin``).
    """
    convert_script = Path(convert_script)
    if not convert_script.exists():
        raise FileNotFoundError(
            f"convert_hf_to_gguf.py not found at {convert_script}. Pass --convert-script "
            "pointing at your llama.cpp checkout's converter."
        )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = _convert_cmd(Path(merged_dir), out_path, convert_script, outtype, python_bin)
    console.print(f"[cyan]convert →[/cyan] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    return out_path


def _quantize_cmd(in_gguf: Path, out_gguf: Path, quant: str, quantize_bin: str) -> list[str]:
    """Build the ``llama-quantize`` argv (pure — unit-testable)."""
    return [quantize_bin, str(in_gguf), str(out_gguf), quant]


def quantize_gguf(
    in_gguf: Path,
    out_gguf: Path,
    *,
    quant: str = "q8_0",
    quantize_bin: str = "llama-quantize",
) -> Path:
    """Quantize an f16 GGUF (e.g. → q8_0, validated against the strict-C runtime).

    Requires a built ``llama-quantize`` binary on PATH or at ``--quantize-bin``;
    raises if absent so the caller can fall back to the un-quantized GGUF.
    """
    found = shutil.which(quantize_bin) or (Path(quantize_bin).exists() and quantize_bin)
    if not found:
        raise FileNotFoundError(
            f"{quantize_bin!r} not found. Build llama.cpp's quantize tool, pass "
            "--quantize-bin, or use --quant none to ship the f16 GGUF "
            "(the strict-C runtime runs both)."
        )
    out_gguf = Path(out_gguf)
    out_gguf.parent.mkdir(parents=True, exist_ok=True)
    cmd = _quantize_cmd(Path(in_gguf), out_gguf, quant, quantize_bin)
    console.print(f"[cyan]quantize →[/cyan] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    return out_gguf


def merge_export(
    base_model: str,
    adapter_dir: Path,
    out_dir: Path,
    *,
    outtype: str = "f16",
    quant: str = "none",
    convert_script: Path = DEFAULT_CONVERT_SCRIPT,
    quantize_bin: str = "llama-quantize",
    dtype: str = "bfloat16",
    attn_impl: str = "sdpa",
    python_bin: str = "python3",
) -> MergeExportResult:
    """End-to-end deploy seam: merge → convert → (optional) quantize, with a
    provenance sidecar. Returns the artifact paths.

    ``quant="none"`` ships the f16 GGUF directly (the strict-C runtime runs f16 and q8_0 — both
    are oracle-gated). ``quant="q8_0"`` additionally quantizes if the tool is present.
    """
    out_dir = Path(out_dir)
    merged_dir = out_dir / "merged"
    gguf_path = out_dir / "model-f16.gguf"

    merge_adapter(base_model, adapter_dir, merged_dir, dtype=dtype, attn_impl=attn_impl)
    convert_to_gguf(
        merged_dir, gguf_path, convert_script=convert_script, outtype=outtype, python_bin=python_bin
    )

    quantized_path: Path | None = None
    if quant and quant.lower() != "none":
        quantized_path = out_dir / f"model-{quant}.gguf"
        quantize_gguf(gguf_path, quantized_path, quant=quant, quantize_bin=quantize_bin)

    final = quantized_path or gguf_path
    provenance_path = Path(f"{final}.provenance.json")
    _write_provenance(
        provenance_path,
        base_model=base_model,
        adapter_dir=Path(adapter_dir),
        gguf_path=final,
        outtype=outtype,
        quant=quant,
        convert_script=Path(convert_script),
    )
    console.print(f"[green]GGUF exported:[/green] {final}")
    console.print(f"[green]Provenance:[/green] {provenance_path}")
    return MergeExportResult(
        merged_dir=merged_dir,
        gguf_path=gguf_path,
        quantized_path=quantized_path,
        provenance_path=provenance_path,
    )


def _write_provenance(
    path: Path,
    *,
    base_model: str,
    adapter_dir: Path,
    gguf_path: Path,
    outtype: str,
    quant: str,
    convert_script: Path,
) -> None:
    """Write the export provenance sidecar (matches toyForge's provenance-everywhere
    convention: who/what/when, for reproducibility of a deployed GGUF)."""
    import json  # noqa: PLC0415

    record = {
        "kind": "toyforge-merge-export",
        "exported_at_utc": datetime.now(UTC).isoformat(),
        "toyforge_git_sha": _git_sha(),
        "base_model": base_model,
        "adapter_dir": str(adapter_dir),
        "adapter_config_sha256": _adapter_fingerprint(adapter_dir),
        "gguf": str(gguf_path),
        "outtype": outtype,
        "quant": quant,
        "convert_script": str(convert_script),
    }
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
