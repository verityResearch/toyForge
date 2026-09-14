"""Scorecard aggregation."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import asdict, dataclass, field

from toyforge.verifier.types import TrajectoryResult

_THINK_RE = re.compile(r"^\s*<think>.*?</think>\s*", re.DOTALL)


def _canonical_call(raw: str) -> str:
    """Extract just the JSON-RPC tail from a `<think>…</think>{json}` string.
    Returns the raw input unchanged if no <think> wrapper is present."""
    return _THINK_RE.sub("", raw, count=1)


@dataclass
class Scorecard:
    run_name: str
    n_examples: int = 0
    pass_at_1: float = 0.0
    pass_at_k: float = 0.0
    pass_at_maj: float = 0.0
    k: int = 8
    failure_modes: dict[str, int] = field(default_factory=dict)
    per_trajectory: list[dict] = field(default_factory=list)
    # Provenance — populated by run_eval via dataclasses.replace; aggregate() leaves these empty.
    eval_timestamp_utc: str = ""  # ISO 8601, e.g. "2026-05-28T15:42:01.123456+00:00"
    git_sha: str = ""  # short SHA from `git rev-parse --short HEAD`; "" outside a repo
    adapter_dir: str = ""  # absolute path to the adapter directory
    base_model: str = ""  # the HF model id used as base
    test_set_path: str = ""  # absolute path to the test jsonl
    test_set_sha256: str = ""  # sha256 hex digest of the test jsonl bytes
    eval_mode: str = "teacher_forced"  # vs "auto_regressive" (future)
    max_new_tokens: int = 0
    temperature: float = 0.0
    runtime: str = "transformers"
    constrained_decoding: bool = False
    grammar_sha256: str = ""
    llamacpp_base_url: str = ""
    llamacpp_model: str = ""
    llamacpp_commit: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def aggregate(
    run_name: str,
    greedy_results: list[TrajectoryResult],
    sampled_results: list[list[TrajectoryResult]],
    sampled_final_calls: list[list[str]],
    k: int,
    trajectories: list[dict] | None = None,  # the input test rows, length-equal to greedy_results
) -> Scorecard:
    n = len(greedy_results)
    if n == 0:
        return Scorecard(run_name=run_name)

    pass1 = sum(1 for r in greedy_results if r.passed) / n
    passk = sum(1 for sl in sampled_results if any(s.passed for s in sl)) / n

    maj = 0
    for sl, calls in zip(sampled_results, sampled_final_calls, strict=True):
        if not calls:
            continue
        canonical = [_canonical_call(c) for c in calls]
        most_common, _count = Counter(canonical).most_common(1)[0]
        # The majority canonical call wins; pass@maj asks whether that *call class*
        # is verifiable. If any trajectory emitting the winning call passes the
        # verifier, the consensus answer is reachable. Using first-occurrence's
        # `passed` would silently fail when the first emitter had an unrelated
        # multi-step issue but later emitters of the same call passed.
        winner_passed = any(
            sr.passed for sr, c in zip(sl, canonical, strict=True) if c == most_common
        )
        if winner_passed:
            maj += 1
    pass_maj = maj / n

    # Failure-mode breakdown: count which subscore was the first to fail across greedy samples.
    failures: Counter[str] = Counter()
    subscore_names = [
        "parse",
        "schema",
        "method_known",
        "precondition_met",
        "transition_valid",
        "sequence_optimal",
    ]
    for r in greedy_results:
        if r.passed:
            continue
        for step in r.steps:
            sub = step.subscores.as_dict()
            failing = next((name for name in subscore_names if sub[name] < 1.0), None)
            if failing is not None:
                failures[failing] += 1
                break  # only break the step loop, not the trajectory loop

    # Per-trajectory diagnostics: which seed failed, at which step, on which subscore.
    per_trajectory: list[dict] = []
    trajs = trajectories or [{} for _ in greedy_results]
    for r, traj in zip(greedy_results, trajs, strict=True):
        entry: dict = {
            "trajectory_id": traj.get("trajectory_id", ""),
            "initial_state": traj.get("initial_state", ""),
            "passed": r.passed,
            "first_failing_step": None,
            "first_failing_subscore": None,
            "error_message": None,
        }
        if not r.passed:
            for step_idx, step in enumerate(r.steps):
                sub = step.subscores.as_dict()
                failing = next((name for name in subscore_names if sub[name] < 1.0), None)
                if failing is not None:
                    entry["first_failing_step"] = step_idx
                    entry["first_failing_subscore"] = failing
                    entry["error_message"] = step.error_message
                    break
        per_trajectory.append(entry)

    return Scorecard(
        run_name=run_name,
        n_examples=n,
        pass_at_1=pass1,
        pass_at_k=passk,
        pass_at_maj=pass_maj,
        k=k,
        failure_modes=dict(failures),
        per_trajectory=per_trajectory,
    )
