"""Hypothesis strategies for verifier property tests.

Generates `(prior_state, model_output_string)` pairs derived from the canonical
schemas/ contents so the strategies stay locked to the real schema surface.
"""

from __future__ import annotations

import json
from pathlib import Path

from hypothesis import strategies as st

from toyforge.schemas import Schemas, load_schemas

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCHEMAS = load_schemas(_REPO_ROOT / "schemas")


def schemas() -> Schemas:
    """Return the loaded schemas (frozen dataclass, safe to share)."""
    return _SCHEMAS


def state_names() -> st.SearchStrategy[str]:
    return st.sampled_from(sorted(_SCHEMAS.states))


def method_names() -> st.SearchStrategy[str]:
    return st.sampled_from(sorted(_SCHEMAS.method_names))


def known_or_unknown_method() -> st.SearchStrategy[str]:
    """Returns a known method ~half the time and a clearly-bogus name otherwise."""
    return st.one_of(
        method_names(),
        # "ticket_openz" is a deliberate near-miss of "ticket_open".
        st.sampled_from(["definitely_not_a_method", "xxx", "ticket_openz"]),
    )


def did_strings() -> st.SearchStrategy[str]:
    """Strings that may or may not satisfy ^user:."""
    return st.one_of(
        st.from_regex(r"^user:[a-z]{2,4}:[a-z0-9._\-]{1,16}$", fullmatch=True),
        st.text(min_size=0, max_size=24),
    )


def short_text(min_size: int = 0, max_size: int = 32) -> st.SearchStrategy[str]:
    return st.text(
        alphabet=st.characters(min_codepoint=33, max_codepoint=126),
        min_size=min_size,
        max_size=max_size,
    )


def arbitrary_params() -> st.SearchStrategy[dict]:
    """Arbitrary param dicts — most will fail schema validation, by design."""
    return st.dictionaries(
        keys=short_text(min_size=1, max_size=12),
        values=st.one_of(
            st.text(max_size=24),
            st.integers(),
            st.booleans(),
        ),
        max_size=4,
    )


def model_output(
    thinking: str = "reasoning",
    method: str | None = None,
    params: dict | None = None,
    call_id: int = 1,
) -> str:
    """Render a complete <think>…</think>{json-rpc} output string."""
    call = {
        "jsonrpc": "2.0",
        "method": method if method is not None else "ticket_get",
        "params": params if params is not None else {},
        "id": call_id,
    }
    return f"<think>{thinking}</think>{json.dumps(call)}"


def model_outputs() -> st.SearchStrategy[str]:
    """Generates a mostly-well-formed output string with arbitrary internals."""
    return st.builds(
        model_output,
        thinking=short_text(max_size=64),
        method=known_or_unknown_method(),
        params=arbitrary_params(),
        call_id=st.integers(min_value=0, max_value=10_000),
    )


def malformed_outputs() -> st.SearchStrategy[str]:
    """Generates outputs that should fail at the parse stage."""
    return st.one_of(
        short_text(max_size=64),  # no <think> at all
        st.builds(lambda s: f"<think>{s}", short_text()),  # unclosed
        st.builds(lambda s: f"<think>{s}</think>not json", short_text()),
        st.builds(lambda s: f"<think>{s}</think>[1,2,3]", short_text()),  # array not object
    )


def prompt_contexts() -> st.SearchStrategy[dict]:
    """Plausible prompt_context dicts: always carries prior_state."""
    return st.builds(
        lambda state, expected_trigger, expected_state_after: {
            "prior_state": state,
            **({"expected_trigger": expected_trigger} if expected_trigger else {}),
            **({"expected_state_after": expected_state_after} if expected_state_after else {}),
        },
        state=state_names(),
        expected_trigger=st.one_of(st.none(), short_text(min_size=1, max_size=24)),
        expected_state_after=st.one_of(st.none(), state_names()),
    )
