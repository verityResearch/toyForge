"""Teacher-driven scenario expansion with verifier-gated acceptance."""

from __future__ import annotations

import json
import random
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
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
from toyforge.scenario_gen.provenance import build_provenance, hand_seed_provenance
from toyforge.scenario_gen.seeds import load_seeds
from toyforge.scenario_gen.teachers.base import Teacher
from toyforge.schemas import load_schemas
from toyforge.verifier.trajectory import verify_trajectory

try:
    from toyforge import __version__ as _TOYFORGE_VERSION
except ImportError:
    _TOYFORGE_VERSION = "0.0.0"

_SYSTEM_PROMPT = (
    "You produce JSON trajectories for a reasoning-model training corpus.\n"
    "A trajectory is a sequence of steps; each step has <think>reasoning</think>"
    " followed by a JSON-RPC call against the support-ticket API.\n\n"
    "## Method -> trigger map (authoritative)\n"
    "Each method can ONLY emit the listed triggers. Never claim another trigger.\n"
    "  ticket_open            -> ticket_open.accepted\n"
    "  ticket_get             -> (query-only, no trigger, no state change)\n"
    "  ticket_status          -> (query-only, no trigger, no state change)\n"
    "  ticket_history         -> (query-only, no trigger, no state change)\n"
    "  admin_agent_add        -> (query-only, no trigger, no state change)\n"
    "  ticket_reopen          -> ticket_reopen.issued\n"
    "  resolution_confirm     -> resolution_confirm.passed | resolution_confirm.failed\n"
    "  admin_escalate         -> escalation.start | escalation.resolved\n"
    "  admin_lifecycle_apply  -> lifecycle.close | lifecycle.archive_window_elapsed\n\n"
    "## Param format constraints (enforced by verifier)\n"
    "- `requester_id`: must match ^user: (e.g. user:alice, user:acme-it).\n"
    "- `body_text`: the customer's message; must be 4+ characters.\n"
    "- `additionalProperties: false` on every method — do NOT add extra fields"
    " (e.g. no `notes`, `priority`, `reason`). Stick to the documented params only.\n"
    "- `review_id`: non-empty string. `ticket_id`: non-empty string.\n\n"
    "## Query-step structure\n"
    "For query-only methods, omit `expected_trigger` and `expected_state_after` from"
    " the step. The `final_state` must equal the `initial_state`.\n\n"
    "## <think> quality bar\n"
    "Each thinking block must: (a) name the prior_state explicitly,"
    " (b) state which trigger will be emitted (or 'query-only, no trigger'),"
    " (c) explain WHY this method/transition is correct given the situation"
    " (not just paraphrase the prompt). Reasoning must be substantive, not decorative.\n\n"
    "## Multi-step prior_calls\n"
    "For step N > 0 in a multi-step trajectory, `prompt_context.prior_calls` must"
    " list the prior steps' `tool_call` objects in order. This grounds the model's"
    " reasoning in what has already been called.\n\n"
    "## Variation guidance\n"
    "Vary requester_id names, ticket_id values, agent_id values, and scenario"
    " descriptions across expansions. Avoid repeating the seed's identifiers"
    " (e.g. if seed uses user:alice and T-7, use different values).\n\n"
    "## Worked example\n"
    "Seed (abbreviated):\n"
    '  {"trajectory_id": "seed_escalation_start", "initial_state": "ESCALATED",'
    ' "steps": [{"prompt_context": {"prior_state": "ESCALATED",'
    ' "situation": "T-7 failed customer confirmation twice"},'
    ' "thinking": "...", "tool_call": {"jsonrpc": "2.0", "method": "admin_escalate",'
    ' "params": {"agent_id": "agent-4", "ticket_id": "T-7"}, "id": 1},'
    ' "expected_trigger": "escalation.start", "expected_state_after": "ENGINEERING"}],'
    ' "final_state": "ENGINEERING"}\n\n'
    "Good expansion:\n"
    '  {"trajectory_id": "exp_escalation_start_abc123", "initial_state": "ESCALATED",'
    ' "source": "teacher_expansion", "difficulty": "easy",'
    ' "description": "Hand T-88 to engineering after the fix did not hold.",'
    ' "steps": [{"prompt_context": {"prior_state": "ESCALATED",'
    ' "situation": "T-88 was reopened and the customer rejected the resolution."},'
    ' "thinking": "The ticket is in ESCALATED because the customer rejected the'
    " resolution on review. admin_escalate from ESCALATED emits escalation.start"
    " (not escalation.resolved — that comes only from ENGINEERING). I pass agent_id=agent-12"
    ' and ticket_id=T-88 to open the engineering escalation.",'
    ' "tool_call": {"jsonrpc": "2.0", "method": "admin_escalate",'
    ' "params": {"agent_id": "agent-12", "ticket_id": "T-88"}, "id": 1},'
    ' "expected_trigger": "escalation.start", "expected_state_after": "ENGINEERING"}],'
    ' "final_state": "ENGINEERING"}\n\n'
    "## Negative example (verifier will REJECT this)\n"
    "BAD: calling admin_escalate but claiming expected_trigger: routing.assigned\n"
    '  {"method": "admin_escalate", ..., "expected_trigger": "routing.assigned"}\n'
    "Why it fails: admin_escalate can only emit escalation.start or escalation.resolved."
    " The verifier sets precondition_met=0 and transition_valid=0 for this mismatch.\n\n"
    "## Output format\n"
    "Return ONLY the JSON trajectory (no prose, no markdown fences),"
    " matching the seed's top-level structure exactly."
    ' Set source="teacher_expansion" and use a unique trajectory_id.'
)


def _build_user_prompt(seed: dict[str, Any]) -> str:
    return f"Seed trajectory (extend with a NEW variant):\n\n{json.dumps(seed, indent=2)}"


def _render_outputs(trajectory: dict[str, Any]) -> list[str]:
    return [
        f"<think>{step['thinking']}</think>{json.dumps(step['tool_call'])}"
        for step in trajectory["steps"]
    ]


# Matches an opening fence + optional language tag + optional whitespace/newline.
# Works for both multi-line (```json\n...) and single-line (```json{...}) shapes.
_FENCE_OPEN_RE = re.compile(r"^```[A-Za-z]*\s*")


def _strip_markdown_fences(raw: str) -> str:
    """Strip ```json ... ``` or ``` ... ``` wrappers if present.

    Teachers often emit fenced code blocks despite the system prompt asking for
    raw JSON. Without this, every fenced response becomes a wasted rejection.
    Idempotent on already-bare JSON. Handles both multi-line and single-line
    fence shapes.
    """
    s = raw.strip()
    if s.startswith("```"):
        s = _FENCE_OPEN_RE.sub("", s, count=1).rstrip()
        if s.endswith("```"):
            s = s[:-3].rstrip()
    return s


def expand_seeds(
    seeds_path: Path,
    schemas_dir: Path,
    teacher: Teacher,
    expansions_per_seed: int,
    out_dir: Path,
    shuffle_seed: int = 42,
    max_retries: int = 3,
    train_frac: float = 0.8,
    dev_frac: float = 0.1,
    resume: bool = False,
    num_workers: int = 10,
) -> dict[str, int]:
    """Expand seed trajectories with the teacher. Verifier-gates every expansion.

    Splits:
      - train.jsonl: train_frac of teacher expansions
      - dev.jsonl:   dev_frac of teacher expansions
      - test.jsonl:  ALL hand-seeds (never teacher-expanded — clean test set)

    Rejected outputs go to out_dir/rejected/rejections.jsonl.
    Each accepted trajectory is streamed to out_dir/accepted_partial.jsonl as it
    is produced; on clean exit the partial file is deleted (data/{train,dev,test}.jsonl
    are the canonical artifacts). Pass resume=True to recover from a crash: the
    partial file is read to pre-populate accepted trajectories and skip seeds that
    already have enough expansions.

    `num_workers` controls the ThreadPoolExecutor size. Workers call teacher.complete
    concurrently; all shared-state mutations happen in the main thread under a lock
    after each future resolves. The Anthropic and OpenAI SDKs are thread-safe.

    Returns a stats dict with accepted/rejected counts.

    Note: `shuffle_seed` controls ONLY the train/dev split shuffle. With parallel
    execution, `accepted` ordering is non-deterministic (futures complete in any
    order), so two runs with the same `shuffle_seed` may produce different splits.
    Use temperature=0.0 on the TeacherConfig for closer-to-deterministic teacher
    output (still subject to provider-side non-determinism).
    """
    if train_frac + dev_frac > 1.0:
        raise ValueError(
            f"train_frac ({train_frac}) + dev_frac ({dev_frac}) > 1.0; "
            f"would silently truncate splits"
        )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rejected_dir = out_dir / "rejected"
    rejected_dir.mkdir(exist_ok=True)

    schemas = load_schemas(schemas_dir)
    seeds = load_seeds(seeds_path)
    rng = random.Random(shuffle_seed)

    accepted: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    rejected_count = 0

    # --- Partial-file / resume setup ---
    partial_path = out_dir / "accepted_partial.jsonl"
    existing_by_seed: dict[str, int] = {}
    if resume and partial_path.exists():
        with partial_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    traj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                seed_id = traj.get("provenance", {}).get("seed_trajectory_id", "")
                if not seed_id:
                    # Legacy entries without provenance — skip gracefully.
                    continue
                existing_by_seed[seed_id] = existing_by_seed.get(seed_id, 0) + 1
                accepted.append(traj)
                seen_ids.add(traj.get("trajectory_id", ""))
        _console.print(
            f"[yellow]Resuming:[/yellow] loaded {len(accepted)} accepted from partial file"
        )
    elif not resume and partial_path.exists():
        # Stale partial from a previous crashed run — truncate to avoid mixing.
        partial_path.unlink()

    # Append mode — previous runs' rejection history is preserved. Each run starts
    # with a sentinel header that downstream readers (analyze_rejections.py) detect
    # for run boundaries and skip from per-rejection counts.
    #
    # Open rej_log FIRST, in an outer try/finally, so it is always closed even
    # if partial_log.open() raises (e.g., permission error on out_dir).  The
    # sentinel is written only after BOTH files are open so a failed
    # partial_log.open() leaves no orphaned sentinel in rejections.jsonl.
    rej_log = (rejected_dir / "rejections.jsonl").open("a")
    try:
        # Open partial file in append mode for streaming accepted trajectories.
        # If this raises, the outer finally closes rej_log with no sentinel written.
        partial_log = partial_path.open("a")
        try:
            # Both files are open — safe to write the sentinel now.
            rej_log.write(
                json.dumps(
                    {
                        "_run_start": True,
                        "timestamp": datetime.now(UTC).isoformat(),
                        "shuffle_seed": shuffle_seed,
                    }
                )
                + "\n"
            )
            total = len(seeds) * expansions_per_seed

            # Protects all shared state: accepted, seen_ids, rejected_count,
            # partial_log, rej_log, and progress bar updates.
            state_lock = threading.Lock()

            def _process_one_expansion(
                seed: dict[str, Any],
            ) -> tuple[str, dict[str, Any] | None, list[dict[str, Any]]]:
                """Try up to max_retries to produce one accepted expansion for this seed.

                Runs in a worker thread. Touches only its own locals + read-only closure
                vars (teacher, schemas, _SYSTEM_PROMPT, shuffle_seed, max_retries,
                _TOYFORGE_VERSION, build_provenance, _strip_markdown_fences,
                _render_outputs, verify_trajectory, _build_user_prompt).
                Returns (seed_id, produced_or_none, list_of_rejection_log_entries).
                """
                rejections: list[dict[str, Any]] = []
                produced: dict[str, Any] | None = None
                seed_id = seed["trajectory_id"]
                for _attempt in range(max_retries):
                    try:
                        raw = teacher.complete(_SYSTEM_PROMPT, _build_user_prompt(seed))
                    except Exception as e:
                        rejections.append(
                            {
                                "seed_id": seed_id,
                                "attempt": _attempt,
                                "reason": "teacher_error",
                                "error": f"{type(e).__name__}: {e}",
                                "provenance": build_provenance(
                                    teacher_config=teacher.config,
                                    system_prompt=_SYSTEM_PROMPT,
                                    expansion_seed=shuffle_seed,
                                    seed_trajectory_id=seed_id,
                                    toyforge_version=_TOYFORGE_VERSION,
                                ),
                            }
                        )
                        continue
                    try:
                        traj = json.loads(_strip_markdown_fences(raw))
                    except json.JSONDecodeError as e:
                        rejections.append(
                            {
                                "seed_id": seed_id,
                                "attempt": _attempt,
                                "reason": "json_decode",
                                "error": str(e),
                                "raw": raw[:500],
                                "provenance": build_provenance(
                                    teacher_config=teacher.config,
                                    system_prompt=_SYSTEM_PROMPT,
                                    expansion_seed=shuffle_seed,
                                    seed_trajectory_id=seed_id,
                                    toyforge_version=_TOYFORGE_VERSION,
                                ),
                            }
                        )
                        continue
                    try:
                        outputs = _render_outputs(traj)
                        r = verify_trajectory(traj, outputs, schemas)
                    except (KeyError, TypeError, IndexError) as e:
                        rejections.append(
                            {
                                "seed_id": seed_id,
                                "attempt": _attempt,
                                "reason": "malformed_structure",
                                "error": f"{type(e).__name__}: {e}",
                                "raw": raw[:500],
                                "provenance": build_provenance(
                                    teacher_config=teacher.config,
                                    system_prompt=_SYSTEM_PROMPT,
                                    expansion_seed=shuffle_seed,
                                    seed_trajectory_id=seed_id,
                                    toyforge_version=_TOYFORGE_VERSION,
                                ),
                            }
                        )
                        continue
                    if r.passed:
                        produced = traj
                        break
                    rejections.append(
                        {
                            "seed_id": seed_id,
                            "attempt": _attempt,
                            "reason": "verifier_failed",
                            "error": r.error_message,
                            "step_errors": [st.error_message for st in r.steps],
                            "provenance": build_provenance(
                                teacher_config=teacher.config,
                                system_prompt=_SYSTEM_PROMPT,
                                expansion_seed=shuffle_seed,
                                seed_trajectory_id=seed_id,
                                toyforge_version=_TOYFORGE_VERSION,
                            ),
                        }
                    )
                return seed_id, produced, rejections

            with Progress(
                SpinnerColumn(),
                BarColumn(),
                TextColumn("[progress.description]{task.description}"),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                console=cast(Console, _console),
            ) as progress:
                task = progress.add_task(f"expand: 0/{total} accepted=0 rejected=0", total=total)

                with ThreadPoolExecutor(max_workers=num_workers) as executor:
                    futures = []
                    for s in seeds:
                        skip = existing_by_seed.get(s["trajectory_id"], 0)
                        remaining = max(0, expansions_per_seed - skip)
                        if skip > 0:
                            # Advance progress bar for pre-loaded expansions.
                            progress.advance(task, skip)
                        for _i in range(remaining):
                            futures.append(executor.submit(_process_one_expansion, s))

                    for fut in as_completed(futures):
                        seed_id, produced, rejections = fut.result()
                        with state_lock:
                            # Write all rejection log entries from this attempt sequence.
                            for rej in rejections:
                                rej_log.write(json.dumps(rej) + "\n")
                            if produced is None:
                                rejected_count += 1
                            else:
                                produced["source"] = "teacher_expansion"
                                produced["provenance"] = build_provenance(
                                    teacher_config=teacher.config,
                                    system_prompt=_SYSTEM_PROMPT,
                                    expansion_seed=shuffle_seed,
                                    seed_trajectory_id=seed_id,
                                    toyforge_version=_TOYFORGE_VERSION,
                                )
                                # Dedupe trajectory_id — teachers occasionally echo the seed's ID.
                                original_id = produced.get("trajectory_id", "unknown")
                                if original_id in seen_ids:
                                    # Find a unique suffix — handles the adversarial case where the
                                    # naive fallback id was itself already taken.
                                    suffix = len(accepted)
                                    candidate = f"{original_id}_{suffix}"
                                    while candidate in seen_ids:
                                        suffix += 1
                                        candidate = f"{original_id}_{suffix}"
                                    produced["trajectory_id"] = candidate
                                seen_ids.add(produced["trajectory_id"])
                                accepted.append(produced)
                                # Stream to partial file immediately for crash recovery.
                                partial_log.write(json.dumps(produced) + "\n")
                                partial_log.flush()
                            progress.update(
                                task,
                                advance=1,
                                description=(
                                    f"expand: {len(accepted) + rejected_count}/{total} "
                                    f"accepted={len(accepted)} rejected={rejected_count}"
                                ),
                            )
        finally:
            partial_log.close()
    finally:
        rej_log.close()

    rng.shuffle(accepted)
    n = len(accepted)
    n_train = int(n * train_frac)
    n_dev = int(n * dev_frac)
    train = accepted[:n_train]
    dev = accepted[n_train : n_train + n_dev]

    def _write(path: Path, items: list[dict[str, Any]]) -> None:
        with path.open("w") as f:
            for item in items:
                f.write(json.dumps(item) + "\n")

    _warn_overwrite([out_dir / f"{name}.jsonl" for name in ("train", "dev", "test")])
    _write(out_dir / "train.jsonl", train)
    _write(out_dir / "dev.jsonl", dev)
    for s in seeds:
        s["source"] = "hand_seed"
        s["provenance"] = hand_seed_provenance(
            seed_trajectory_id=s["trajectory_id"],
            toyforge_version=_TOYFORGE_VERSION,
        )
    _write(out_dir / "test.jsonl", seeds)

    # Successful run — partial file is no longer needed. Delete to avoid confusing
    # the next non-resume run (data/{train,dev,test}.jsonl are the canonical artifacts).
    if partial_path.exists():
        partial_path.unlink()

    return {
        "accepted": len(accepted),
        "rejected": rejected_count,
        "train": len(train),
        "dev": len(dev),
        "test": len(seeds),
    }
