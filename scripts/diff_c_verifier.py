"""Compare the first C verifier slice against the Python verifier oracle."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from toyforge.schemas import load_schemas
from toyforge.verifier.core import verify_step


def _call(method: str, params: dict[str, Any], *, ensure_ascii: bool = True) -> str:
    body = {"jsonrpc": "2.0", "method": method, "params": params, "id": 1}
    return f"<think>x</think>{json.dumps(body, separators=(',', ':'), ensure_ascii=ensure_ascii)}"


DEFAULT_OPEN = {
    "requester_id": "user:1",
    "body_text": "YWJj",
    "lifecycle_profile": "std",
}


CASES: list[dict[str, Any]] = [
    {
        "name": "ticket_open_valid",
        "prior_state": "NEW",
        "expected_trigger": "ticket_open.accepted",
        "expected_state_after": "TRIAGED",
        "output": _call(
            "ticket_open",
            {
                "requester_id": "user:1",
                "body_text": "YWJj",
                "lifecycle_profile": "std",
            },
        ),
    },
    {
        "name": "wrong_prior_state",
        "prior_state": "RESOLVED",
        "expected_trigger": "ticket_open.accepted",
        "expected_state_after": "TRIAGED",
        "output": _call(
            "ticket_open",
            {
                "requester_id": "user:1",
                "body_text": "YWJj",
                "lifecycle_profile": "std",
            },
        ),
    },
    {
        "name": "duplicate_method_uses_last_key",
        "prior_state": "NEW",
        "expected_trigger": "ticket_open.accepted",
        "expected_state_after": "TRIAGED",
        "output": (
            '<think>x</think>{"jsonrpc":"2.0","method":"ticket_status",'
            '"method":"ticket_open","params":{"requester_id":"user:1",'
            '"body_text":"YWJj","lifecycle_profile":"std"},"id":1}'
        ),
    },
    {
        "name": "malformed_number_fails_parse",
        "prior_state": "NEW",
        "expected_trigger": "ticket_open.accepted",
        "expected_state_after": "TRIAGED",
        "output": (
            '<think>x</think>{"jsonrpc":"2.0","method":"ticket_open",'
            '"params":{"requester_id":"user:1","body_text":"YWJj",'
            '"lifecycle_profile":"std"},"id":1e}'
        ),
    },
    {
        "name": "query_status",
        "prior_state": "RESOLVED",
        "output": _call("ticket_status", {"ticket_id": "T-123"}),
    },
    {
        "name": "query_status_escaped_unicode",
        "prior_state": "RESOLVED",
        "output": _call("ticket_status", {"ticket_id": "obj-\u00e9-\U0001f680"}),
    },
    {
        "name": "query_status_raw_utf8",
        "prior_state": "RESOLVED",
        "output": _call(
            "ticket_status",
            {"ticket_id": "obj-\u00e9-\U0001f680"},
            ensure_ascii=False,
        ),
    },
    {
        "name": "extra_property",
        "prior_state": "RESOLVED",
        "output": _call("ticket_status", {"ticket_id": "T-123", "extra": "bad"}),
    },
    {
        "name": "audit_history_valid_window",
        "prior_state": "RESOLVED",
        "output": _call(
            "ticket_history",
            {
                "ticket_id": "T-123",
                "window": {"from_utc": "2026-05-29T12:00:00Z"},
            },
        ),
    },
    {
        "name": "audit_history_bad_datetime",
        "prior_state": "RESOLVED",
        "output": _call(
            "ticket_history",
            {
                "ticket_id": "T-123",
                "window": {"from_utc": "not-a-date"},
            },
        ),
    },
    {
        "name": "infer_escalation_start",
        "prior_state": "ESCALATED",
        "infer_trigger": True,
        "output": _call("admin_escalate", {"agent_id": "agent-1", "ticket_id": "T-1"}),
    },
    {
        # Non-object params is a schema failure (schema=0), not a hard parse fail:
        # parse/method_known stay 1 and the state subscores are still computed.
        "name": "nonobject_params_array",
        "prior_state": "NEW",
        "expected_trigger": "ticket_open.accepted",
        "expected_state_after": "TRIAGED",
        "output": ('<think>x</think>{"jsonrpc":"2.0","method":"ticket_open","params":[],"id":1}'),
    },
    {
        "name": "nonobject_params_string_query",
        "prior_state": "RESOLVED",
        "output": (
            '<think>x</think>{"jsonrpc":"2.0","method":"ticket_status","params":"nope","id":1}'
        ),
    },
    # adversarial edge cases locked from the differential-fuzz audit
    {
        "name": "empty_thinking_block",
        "prior_state": "NEW",
        "expected_trigger": "ticket_open.accepted",
        "expected_state_after": "TRIAGED",
        "output": "<think></think>" + _call("ticket_open", DEFAULT_OPEN).split("</think>", 1)[1],
    },
    {
        "name": "missing_close_think_tag",
        "prior_state": "NEW",
        "expected_trigger": "ticket_open.accepted",
        "expected_state_after": "TRIAGED",
        "output": "<think>x" + _call("ticket_open", DEFAULT_OPEN).split("</think>", 1)[1],
    },
    {
        "name": "extra_param_additionalprops_false",
        "prior_state": "NEW",
        "expected_trigger": "ticket_open.accepted",
        "expected_state_after": "TRIAGED",
        "output": _call("ticket_open", {**DEFAULT_OPEN, "notes": "x"}),
    },
    {
        "name": "query_method_with_unexpected_trigger",
        "prior_state": "RESOLVED",
        "expected_trigger": "ticket_open.accepted",
        "expected_state_after": "TRIAGED",
        "output": _call("ticket_get", {"ticket_id": "o1"}),
    },
    {
        "name": "audit_verify_failed_branch",
        "prior_state": "REOPENED",
        "expected_trigger": "resolution_confirm.failed",
        "expected_state_after": "ESCALATED",
        "output": _call("resolution_confirm", {"review_id": "c1", "ticket_id": "o1"}),
    },
    {
        "name": "trailing_garbage_after_json",
        "prior_state": "NEW",
        "expected_trigger": "ticket_open.accepted",
        "expected_state_after": "TRIAGED",
        "output": _call("ticket_open", DEFAULT_OPEN) + "  junk",
    },
    {
        "name": "unicode_in_did_and_thinking",
        "prior_state": "NEW",
        "expected_trigger": "ticket_open.accepted",
        "expected_state_after": "TRIAGED",
        "output": "<think>π</think>"
        + _call(
            "ticket_open", {**DEFAULT_OPEN, "requester_id": "user:π"}, ensure_ascii=False
        ).split("</think>", 1)[1],
    },
]


# Per-method state context + a known-valid params template. The context is a
# (prior_state, expected_trigger, expected_state_after) that yields a valid arc
# for state-changing methods, so schema mutations can be compared in isolation
# from the state subscores. Query-only methods carry no trigger.
_GEN_CONTEXTS: dict[str, dict[str, Any]] = {
    "ticket_open": {
        "prior": "NEW",
        "trigger": "ticket_open.accepted",
        "after": "TRIAGED",
        "valid": {
            "requester_id": "user:1",
            "body_text": "YWJj",
            "lifecycle_profile": "std",
        },
    },
    "ticket_get": {"prior": "RESOLVED", "valid": {"ticket_id": "T-1"}},
    "ticket_status": {"prior": "RESOLVED", "valid": {"ticket_id": "T-1"}},
    "ticket_reopen": {
        "prior": "RESOLVED",
        "trigger": "ticket_reopen.issued",
        "after": "REOPENED",
        "valid": {"ticket_id": "T-1", "reopen_reason": "std"},
    },
    "resolution_confirm": {
        "prior": "REOPENED",
        "trigger": "resolution_confirm.passed",
        "after": "RESOLVED",
        "valid": {"review_id": "c-1", "resolution_notes": {}},
    },
    "ticket_history": {
        "prior": "RESOLVED",
        "valid": {"ticket_id": "T-1", "window": {"from_utc": "2026-05-29T12:00:00Z"}},
    },
    "admin_agent_add": {"prior": "RESOLVED", "valid": {"agent_profile": {}}},
    "admin_escalate": {
        "prior": "ESCALATED",
        "trigger": "escalation.start",
        "after": "ENGINEERING",
        "valid": {"agent_id": "agent-1", "ticket_id": "T-1"},
    },
    "admin_lifecycle_apply": {
        "prior": "RESOLVED",
        "trigger": "lifecycle.close",
        "after": "CLOSED",
        "valid": {"lifecycle_profile": "std"},
    },
}


def _gen_case(
    name: str, ctx: dict[str, Any], params: Any, *, infer: bool = False
) -> dict[str, Any]:
    case: dict[str, Any] = {
        "name": name,
        "prior_state": ctx["prior"],
        "output": _call(ctx["method"], params),
    }
    if infer:
        case["infer_trigger"] = True
        if ctx.get("after"):
            case["expected_state_after"] = ctx["after"]
    else:
        if ctx.get("trigger"):
            case["expected_trigger"] = ctx["trigger"]
        if ctx.get("after"):
            case["expected_state_after"] = ctx["after"]
    return case


def generated_cases() -> list[dict[str, Any]]:
    """Schema + state mutation corpus, compared against the Python oracle.

    For each method, mutate the valid params across the bounded validator's
    axes (drop required, wrong type, minLength, pattern, extra property, nested
    object shape, non-object params) so divergences surface as parity failures.
    """
    cases: list[dict[str, Any]] = []
    for method, base in _GEN_CONTEXTS.items():
        ctx = {**base, "method": method}
        valid = base["valid"]

        # Baseline valid (expected-trigger mode), plus infer-trigger variant for
        # state-changing methods.
        cases.append(_gen_case(f"gen:{method}:valid", ctx, dict(valid)))
        if ctx.get("trigger"):
            cases.append(_gen_case(f"gen:{method}:valid_infer", ctx, dict(valid), infer=True))

        # Drop each required field, one at a time.
        for key in valid:
            mutant = {k: v for k, v in valid.items() if k != key}
            cases.append(_gen_case(f"gen:{method}:drop_{key}", ctx, mutant))

        # Wrong type per property: str->int, object->str.
        for key, value in valid.items():
            wrong = "not_an_object" if isinstance(value, dict) else 12345
            cases.append(_gen_case(f"gen:{method}:wrongtype_{key}", ctx, {**valid, key: wrong}))

        # Empty string where a non-empty/min-length string is expected.
        for key, value in valid.items():
            if isinstance(value, str):
                cases.append(_gen_case(f"gen:{method}:empty_{key}", ctx, {**valid, key: ""}))

        # Extra (additionalProperties:false) property.
        cases.append(_gen_case(f"gen:{method}:extra_prop", ctx, {**valid, "zzz_extra": "x"}))

        # Non-object params of every JSON kind.
        for label, params in [
            ("array", []),
            ("string", "nope"),
            ("number", 7),
            ("null", None),
            ("bool", True),
        ]:
            cases.append(_gen_case(f"gen:{method}:nonobject_{label}", ctx, params))

    # Targeted constraint mutations.
    ci = {**_GEN_CONTEXTS["ticket_open"], "method": "ticket_open"}
    cases.append(
        _gen_case("gen:ticket_open:bad_did_pattern", ci, {**ci["valid"], "requester_id": "ex:1"})
    )
    cases.append(_gen_case("gen:ticket_open:short_base64", ci, {**ci["valid"], "body_text": "ab"}))

    ah = {**_GEN_CONTEXTS["ticket_history"], "method": "ticket_history"}
    cases.append(
        _gen_case("gen:ticket_history:window_nonobject", ah, {"ticket_id": "T-1", "window": "nope"})
    )
    cases.append(
        _gen_case(
            "gen:ticket_history:window_extra_prop",
            ah,
            {"ticket_id": "T-1", "window": {"from_utc": "2026-05-29T12:00:00Z", "zzz": 1}},
        )
    )
    cases.append(
        _gen_case(
            "gen:ticket_history:from_utc_wrongtype",
            ah,
            {"ticket_id": "T-1", "window": {"from_utc": 5}},
        )
    )
    # format: date-time is annotation-only in this environment, so a bad date
    # string must still validate (schema=1) in BOTH Python and C.
    cases.append(
        _gen_case(
            "gen:ticket_history:from_utc_bad_date_annotation_only",
            ah,
            {"ticket_id": "T-1", "window": {"from_utc": "not-a-date"}},
        )
    )
    return cases


def _parse_c_output(stdout: str) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for line in stdout.splitlines():
        if line.startswith("passed="):
            parsed["passed"] = line.split("=", 1)[1] == "true"
        elif line.startswith("subscores "):
            parts = line.removeprefix("subscores ").split()
            parsed["subscores"] = {k: float(v) for k, v in (p.split("=", 1) for p in parts)}
        elif line.startswith("new_state="):
            parsed["new_state"] = line.split("=", 1)[1]
    return parsed


def _run_c(toyforge_c: Path, case: dict[str, Any]) -> dict[str, Any]:
    cmd = [
        str(toyforge_c),
        "verify-step",
        "--schemas-dir",
        "schemas",
        "--prior-state",
        case["prior_state"],
        "--output",
        case["output"],
    ]
    if case.get("expected_trigger") is not None:
        cmd.extend(["--expected-trigger", case["expected_trigger"]])
    if case.get("expected_state_after") is not None:
        cmd.extend(["--expected-state-after", case["expected_state_after"]])
    if case.get("infer_trigger"):
        cmd.append("--infer-trigger")
    proc = subprocess.run(cmd, check=False, text=True, capture_output=True)
    if proc.stderr:
        raise RuntimeError(proc.stderr)
    return _parse_c_output(proc.stdout)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--toyforge-c", type=Path, default=Path("c/build/toyforge-c"))
    args = parser.parse_args()

    schemas = load_schemas(Path("schemas"))
    all_cases = CASES + generated_cases()
    failures: list[str] = []
    for case in all_cases:
        ctx = {"prior_state": case["prior_state"]}
        if case.get("expected_trigger") is not None:
            ctx["expected_trigger"] = case["expected_trigger"]
        if case.get("expected_state_after") is not None:
            ctx["expected_state_after"] = case["expected_state_after"]

        py = verify_step(
            ctx,
            case["output"],
            schemas,
            infer_trigger=bool(case.get("infer_trigger")),
        )
        c = _run_c(args.toyforge_c, case)
        py_sub = py.subscores.as_dict()
        c_sub = c["subscores"]
        if py.passed != c["passed"]:
            failures.append(f"{case['name']}: passed mismatch Python={py.passed} C={c['passed']}")
        for key, py_value in py_sub.items():
            if c_sub.get(key) != py_value:
                failures.append(
                    f"{case['name']}: subscore {key} mismatch Python={py_value} C={c_sub.get(key)}"
                )
        if (py.new_state or "") != c.get("new_state", ""):
            failures.append(
                f"{case['name']}: new_state mismatch Python={py.new_state!r} "
                f"C={c.get('new_state', '')!r}"
            )

    if failures:
        print("\n".join(failures))
        return 1
    print(f"{len(all_cases)} C verifier parity cases passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
