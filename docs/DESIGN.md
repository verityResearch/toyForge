# toyForge design notes

## Goal

Train a small reasoning model to emit tool calls that are correct in three independent ways: well-formed, valid against a parameter schema, and legal given the current state of a system. Measure all three with a verifier that is cheap, deterministic, and identical whether it is grading an evaluation or rewarding a policy.

## Why an invented domain

A real API brings noise the experiment doesn't need: undocumented behavior, rate limits, drifting versions. The support-ticket API here is small enough to specify completely in [`schemas/`](../schemas/) and rich enough to be non-trivial:

- **Query-only methods** that must not claim a state change.
- **Two-outcome methods** (`resolution_confirm` → `passed` | `failed`), where the correct trigger depends on the situation, not the method name.
- **A recovery loop** (REOPENED → ESCALATED → ENGINEERING → RESOLVED) that needs multi-step planning.
- **Background transitions** (routing, agent acknowledgement) that no client call can cause. A good model recognizes when there is nothing it can do.

Swapping in another domain means replacing the files in `schemas/` and `scenarios/`. The Python verifier is fully schema-driven, and the C port's built-in defaults in `c/src/schemas.c` must be kept in step.

## The verifier

`src/toyforge/verifier/` scores one step at a time:

| Subscore | Question |
| --- | --- |
| `parse` | Is the output `<think>…</think>` followed by one JSON object? |
| `schema` | Do the params validate against the method's JSON Schema (with `additionalProperties: false`)? |
| `method_known` | Is the method one of the declared methods? |
| `precondition_met` | Does the state machine allow this call from the prior state? |
| `transition_valid` | Does the claimed or inferred trigger produce a valid transition? |
| `sequence_optimal` | Is this on a shortest path to the goal? |

Subscores combine through named presets in `schemas/reward-rubric.yaml` (`shaped`, `binary`, `schema_only`), so changing the reward shape is a config change, not a code change. The verifier has no I/O, no model calls, and no hidden state. Hypothesis property tests check purity, cascade behavior (a failed parse zeros what depends on it), and state-machine soundness.

## Data integrity

- Teacher expansions are verified before admission; rejects are logged and analyzable (`analyze-rejections`).
- Every row carries provenance: source, teacher provider and model, a hash of the teacher system prompt, and a timestamp.
- `audit-independence` checks train/dev/test for exact and near-duplicate overlap before training.

## C port

The C port exists to run grading and data preparation next to a C inference runtime, without Python. Python remains the reference: each C command has a differential gate in `scripts/diff_c_*.py`, and the core modules are additionally fuzzed against Python. Build variants cover gcc, clang, ASan/UBSan, Release `-O2`, and a no-libcurl build in which network commands fail closed.
