"""Eval runner: load adapter, sample, verify, scorecard."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from toyforge._console import console as _console
from toyforge._console import warn_overwrite as _warn_overwrite
from toyforge.eval.score import Scorecard, aggregate
from toyforge.schemas import load_schemas
from toyforge.verifier.trajectory import verify_trajectory
from toyforge.verifier.types import StepResult, Subscores, TrajectoryResult


def _trajectory_result_to_dict(tr: TrajectoryResult) -> dict[str, Any]:
    """Convert TrajectoryResult to a JSON-serializable dict."""
    return {
        "passed": tr.passed,
        "final_state": tr.final_state,
        "error_message": tr.error_message,
        "steps": [
            {
                "passed": step.passed,
                "subscores": step.subscores.as_dict(),
                "parsed_thinking": step.parsed_thinking,
                "parsed_call": step.parsed_call,
                "method": step.method,
                "new_state": step.new_state,
                "error_message": step.error_message,
            }
            for step in tr.steps
        ],
    }


def _trajectory_result_from_dict(d: dict[str, Any]) -> TrajectoryResult:
    """Inverse of _trajectory_result_to_dict."""
    return TrajectoryResult(
        passed=d["passed"],
        final_state=d["final_state"],
        error_message=d["error_message"],
        steps=[
            StepResult(
                passed=s["passed"],
                subscores=Subscores(**s["subscores"]),
                parsed_thinking=s["parsed_thinking"],
                parsed_call=s["parsed_call"],
                method=s["method"],
                new_state=s["new_state"],
                error_message=s["error_message"],
            )
            for s in d["steps"]
        ],
    )


def _render_failed_trajectories(entries: list[dict]) -> str:
    """Render a markdown section listing failed trajectories grouped by initial_state."""
    failed = [e for e in entries if not e["passed"]]
    if not failed:
        return "\n\n## Failed trajectories\nNone (all greedy passes).\n"
    out = ["", "", "## Failed trajectories"]
    # Group by initial_state
    by_state: dict[str, list[dict]] = {}
    for e in failed:
        by_state.setdefault(e["initial_state"] or "(unknown)", []).append(e)
    for state in sorted(by_state):
        out.append(f"\n### {state}\n")
        out.append("| trajectory_id | first_failing_step | first_failing_subscore | error |")
        out.append("|---|---|---|---|")
        for e in by_state[state]:
            err = (e["error_message"] or "")[:80].replace("|", "\\|")
            out.append(
                f"| {e['trajectory_id']} | {e['first_failing_step']} | "
                f"{e['first_failing_subscore']} | {err} |"
            )
    return "\n".join(out) + "\n"


def _load_test(jsonl_path: Path) -> list[dict]:
    """Load test trajectories from JSONL. Raises FileNotFoundError if absent.
    Skips malformed lines with a warning (mirrors compare.compare_runs)."""
    if not jsonl_path.exists():
        raise FileNotFoundError(
            f"test set not found: {jsonl_path}. Run `just expand` (phase0) to generate data/ first."
        )
    rows: list[dict] = []
    for i, line in enumerate(jsonl_path.read_text().splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as e:
            _console.print(f"[yellow]skip malformed line {i} in {jsonl_path.name}: {e}[/yellow]")
    return rows


def run_eval(
    adapter_dir: Path | None,
    test_path: Path,
    schemas_dir: Path,
    out_dir: Path,
    base_model: str,
    k: int = 8,
    max_new_tokens: int = 1024,
    run_name: str = "eval",
    attn_impl: str = "sdpa",
    resume: bool = False,
    temperature: float = 0.7,
    runtime: str = "transformers",
    constrained: bool = False,
    llamacpp_base_url: str = "http://localhost:8080/v1",
    llamacpp_model: str | None = None,
    grammar_path: Path = Path("schemas/jsonrpc.gbnf"),
    llamacpp_commit: str = "",
) -> Scorecard:
    """Evaluate a trained adapter (or base model) and write reports/{run_name}.{json,md}.

    When adapter_dir is None the base model is loaded directly with no PEFT adapter
    (B0 zero-shot baseline).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Provenance: computed before any heavy work so timestamp is accurate ---
    eval_timestamp_utc = datetime.now(UTC).isoformat()
    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        git_sha = ""
    test_set_sha256 = hashlib.sha256(Path(test_path).read_bytes()).hexdigest()

    schemas = load_schemas(schemas_dir)
    test = _load_test(Path(test_path))

    # --- Resume / partial file handling ---
    partial_path = out_dir / f"{run_name}_partial.jsonl"
    completed_by_id: dict[str, dict] = {}

    if resume and partial_path.exists():
        with partial_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                tid = entry.get("trajectory_id")
                if tid:
                    completed_by_id[tid] = entry
        _console.print(
            f"[yellow]Resuming:[/yellow] loaded {len(completed_by_id)} completed trajectories"
            " from partial file"
        )
    elif not resume and partial_path.exists():
        partial_path.unlink()

    if runtime not in {"transformers", "llamacpp"}:
        raise ValueError(f"unknown eval runtime: {runtime!r}")

    # Determine whether any generation work is needed
    remaining_trajectories = [t for t in test if t.get("trajectory_id", "") not in completed_by_id]
    need_generation = len(remaining_trajectories) > 0

    tokenizer = None
    model = None
    llamacpp_client = None
    if need_generation and runtime == "transformers":
        from peft import PeftModel  # noqa: PLC0415
        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

        # Tokenizer: from adapter dir if provided, else from base model
        tokenizer_source = str(adapter_dir) if adapter_dir is not None else base_model
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype="bfloat16",
            device_map="auto",
            attn_implementation=attn_impl,
            trust_remote_code=True,
        )
        if adapter_dir is not None:
            model = PeftModel.from_pretrained(base, str(adapter_dir))
        else:
            model = base
        model.eval()
    elif need_generation and runtime == "llamacpp":
        from toyforge.llamacpp import LlamaCppClient, LlamaCppConfig  # noqa: PLC0415

        llamacpp_client = LlamaCppClient(
            LlamaCppConfig(
                base_url=llamacpp_base_url,
                model=llamacpp_model or base_model,
                grammar_path=grammar_path,
                commit=llamacpp_commit,
            )
        )

    greedy: list = []
    sampled: list[list] = []
    sampled_finals: list[list[str]] = []

    partial_log = partial_path.open("a")
    try:
        with Progress(
            SpinnerColumn(),
            BarColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=cast(Console, _console),
        ) as progress:
            eval_task = progress.add_task(f"eval: 0/{len(test)}", total=len(test))
            for traj in test:
                tid = traj.get("trajectory_id", "")
                if tid in completed_by_id:
                    entry = completed_by_id[tid]
                    greedy.append(_trajectory_result_from_dict(entry["greedy"]))
                    sampled.append([_trajectory_result_from_dict(s) for s in entry["sampled"]])
                    sampled_finals.append(list(entry["sampled_finals"]))
                else:
                    if runtime == "llamacpp":
                        from toyforge.llamacpp import render_step_outputs_llamacpp  # noqa: PLC0415

                        if llamacpp_client is None:
                            raise RuntimeError("llama.cpp client was not initialized")
                        per_step = render_step_outputs_llamacpp(
                            llamacpp_client,
                            traj,
                            k=k,
                            max_new_tokens=max_new_tokens,
                            temperature=temperature,
                            constrained=constrained,
                        )
                    else:
                        from toyforge.rollout import render_step_outputs  # noqa: PLC0415

                        per_step = render_step_outputs(
                            model,
                            tokenizer,
                            traj,
                            k=k,
                            max_new_tokens=max_new_tokens,
                            temperature=temperature,
                        )
                    greedy_outputs = [step_outs[0] for step_outs in per_step]
                    sampled_trajectories = [
                        [step_outs[1 + j] for step_outs in per_step] for j in range(k)
                    ]
                    g = verify_trajectory(traj, greedy_outputs, schemas)
                    s = [verify_trajectory(traj, st, schemas) for st in sampled_trajectories]
                    sf = [st[-1] if st else "" for st in sampled_trajectories]
                    greedy.append(g)
                    sampled.append(s)
                    sampled_finals.append(sf)
                    # Stream result to partial file for crash recovery
                    entry = {
                        "trajectory_id": tid,
                        "greedy": _trajectory_result_to_dict(g),
                        "sampled": [_trajectory_result_to_dict(sr) for sr in s],
                        "sampled_finals": sf,
                    }
                    partial_log.write(json.dumps(entry) + "\n")
                    partial_log.flush()
                progress.update(
                    eval_task,
                    advance=1,
                    description=f"eval: {len(greedy)}/{len(test)}",
                )
    finally:
        partial_log.close()

    scorecard = aggregate(run_name, greedy, sampled, sampled_finals, k=k, trajectories=test)
    grammar_digest = ""
    if runtime == "llamacpp":
        from toyforge.llamacpp import grammar_sha256  # noqa: PLC0415

        grammar_digest = grammar_sha256(grammar_path)
    scorecard = replace(
        scorecard,
        eval_timestamp_utc=eval_timestamp_utc,
        git_sha=git_sha,
        adapter_dir=str(Path(adapter_dir).resolve()) if adapter_dir is not None else "(base-only)",
        base_model=base_model,
        test_set_path=str(Path(test_path).resolve()),
        test_set_sha256=test_set_sha256,
        eval_mode="teacher_forced",
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        runtime=runtime,
        constrained_decoding=constrained,
        grammar_sha256=grammar_digest,
        llamacpp_base_url=llamacpp_base_url if runtime == "llamacpp" else "",
        llamacpp_model=(llamacpp_model or base_model) if runtime == "llamacpp" else "",
        llamacpp_commit=llamacpp_commit if runtime == "llamacpp" else "",
    )

    _warn_overwrite([out_dir / f"{run_name}.{ext}" for ext in ("json", "md")])
    (out_dir / f"{run_name}.json").write_text(json.dumps(scorecard.to_dict(), indent=2))
    md = (
        f"# Eval report — {run_name}\n\n"
        f"- n_examples: {scorecard.n_examples}\n"
        f"- pass@1: {scorecard.pass_at_1:.3f}\n"
        f"- pass@{scorecard.k}: {scorecard.pass_at_k:.3f}\n"
        f"- pass@maj: {scorecard.pass_at_maj:.3f}\n\n"
        f"## Provenance\n"
        f"- timestamp_utc: {scorecard.eval_timestamp_utc}\n"
        f"- git_sha: {scorecard.git_sha or '(no git context)'}\n"
        f"- adapter_dir: {scorecard.adapter_dir}\n"
        f"- base_model: {scorecard.base_model}\n"
        f"- test_set_path: {scorecard.test_set_path}\n"
        f"- test_set_sha256: {scorecard.test_set_sha256}\n"
        f"- eval_mode: {scorecard.eval_mode}\n"
        f"- max_new_tokens: {scorecard.max_new_tokens}\n"
        f"- temperature: {scorecard.temperature}\n\n"
        f"- runtime: {scorecard.runtime}\n"
        f"- constrained_decoding: {scorecard.constrained_decoding}\n"
        f"- grammar_sha256: {scorecard.grammar_sha256}\n"
        f"- llamacpp_base_url: {scorecard.llamacpp_base_url}\n"
        f"- llamacpp_model: {scorecard.llamacpp_model}\n"
        f"- llamacpp_commit: {scorecard.llamacpp_commit}\n\n"
        f"## Failure modes (greedy)\n"
        + "\n".join(f"- {name}: {count}" for name, count in scorecard.failure_modes.items())
        + _render_failed_trajectories(scorecard.per_trajectory)
    )
    (out_dir / f"{run_name}.md").write_text(md)

    # Clean exit: delete partial file
    if partial_path.exists():
        partial_path.unlink()

    _console.print(
        f"[green]Eval done.[/green] pass@1={scorecard.pass_at_1:.3f}"
        f" pass@{k}={scorecard.pass_at_k:.3f}"
    )
    return scorecard
