"""CPU-only tests for llama.cpp eval provenance."""

from __future__ import annotations

from pathlib import Path

from toyforge.eval.runner import run_eval


def test_run_eval_llamacpp_runtime_empty_set_records_provenance(tmp_path: Path, schemas_dir: Path):
    test_path = tmp_path / "test.jsonl"
    test_path.write_text("")
    out_dir = tmp_path / "reports"
    grammar = schemas_dir / "jsonrpc.gbnf"

    sc = run_eval(
        adapter_dir=None,
        test_path=test_path,
        schemas_dir=schemas_dir,
        out_dir=out_dir,
        base_model="qwen.gguf",
        run_name="llama-empty",
        runtime="llamacpp",
        constrained=True,
        llamacpp_base_url="http://localhost:8080/v1",
        llamacpp_model="qwen-local",
        grammar_path=grammar,
        llamacpp_commit="abc123",
    )

    assert sc.runtime == "llamacpp"
    assert sc.constrained_decoding is True
    assert sc.grammar_sha256
    assert sc.llamacpp_model == "qwen-local"
    assert sc.llamacpp_commit == "abc123"
    assert (out_dir / "llama-empty.json").exists()
