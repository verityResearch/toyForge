"""Unit tests for the merge+export deploy seam (toyforge.export).

These cover the CPU-only, train-extra-free surface: argv construction, the
fail-closed guards (missing converter / quantize tool), the adapter fingerprint,
and the provenance sidecar. The merge itself (PEFT + transformers) needs a model
and is exercised by an integration run, not here — but everything that decides
*how* the export shells out is locked down deterministically.
"""

from __future__ import annotations

import json

import pytest

from toyforge.export import (
    _adapter_fingerprint,
    _convert_cmd,
    _quantize_cmd,
    _write_provenance,
    convert_to_gguf,
    quantize_gguf,
)


def test_convert_cmd_argv():
    cmd = _convert_cmd(
        merged_dir="/m",
        out_path="/out/x.gguf",
        convert_script="/c/conv.py",
        outtype="f16",
        python_bin="python3",
    )
    assert cmd == [
        "python3",
        "/c/conv.py",
        "/m",
        "--outfile",
        "/out/x.gguf",
        "--outtype",
        "f16",
    ]


def test_quantize_cmd_argv():
    cmd = _quantize_cmd("/in.gguf", "/out.gguf", "q8_0", "llama-quantize")
    assert cmd == ["llama-quantize", "/in.gguf", "/out.gguf", "q8_0"]


def test_convert_missing_script_fails_closed(tmp_path):
    # A converter path that does not exist must raise before any subprocess.
    missing = tmp_path / "nope" / "convert_hf_to_gguf.py"
    with pytest.raises(FileNotFoundError):
        convert_to_gguf(tmp_path, tmp_path / "x.gguf", convert_script=missing)


def test_quantize_missing_binary_fails_closed(tmp_path):
    # An absent quantize binary must raise (caller falls back to f16), not silently pass.
    with pytest.raises(FileNotFoundError):
        quantize_gguf(
            tmp_path / "in.gguf",
            tmp_path / "out.gguf",
            quant="q8_0",
            quantize_bin="definitely-not-a-real-binary-xyz",
        )


def test_adapter_fingerprint_absent_and_present(tmp_path):
    assert _adapter_fingerprint(tmp_path) == ""  # no adapter_config.json
    (tmp_path / "adapter_config.json").write_text('{"r": 32}', encoding="utf-8")
    fp = _adapter_fingerprint(tmp_path)
    assert len(fp) == 64  # sha256 hex
    # stable
    assert fp == _adapter_fingerprint(tmp_path)


def test_provenance_sidecar_shape(tmp_path):
    out = tmp_path / "model-f16.gguf.provenance.json"
    _write_provenance(
        out,
        base_model="Qwen/Qwen3-8B-Instruct",
        adapter_dir=tmp_path,
        gguf_path=tmp_path / "model-f16.gguf",
        outtype="f16",
        quant="none",
        convert_script=tmp_path / "conv.py",
    )
    rec = json.loads(out.read_text())
    assert rec["kind"] == "toyforge-merge-export"
    assert rec["base_model"] == "Qwen/Qwen3-8B-Instruct"
    assert rec["outtype"] == "f16"
    assert rec["quant"] == "none"
    assert "exported_at_utc" in rec
    assert "toyforge_git_sha" in rec
