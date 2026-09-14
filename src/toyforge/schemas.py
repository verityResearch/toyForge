"""Load and expose the schemas/ directory as a single in-memory object.

This is the single source of truth for the state machine, JSON-RPC method
schemas, trajectory format, and reward rubric. Three readers consume this:
scenario_gen, verifier, and eval.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Schemas:
    """In-memory snapshot of the schemas/ directory."""

    states: frozenset[str]
    initial_state: str
    terminal_states: frozenset[str]
    transitions: dict[tuple[str, str], str]  # (from, trigger) -> to
    method_triggers: dict[str, list[str]]  # method -> [trigger strings]
    method_names: frozenset[str]
    method_schemas: dict[str, dict[str, Any]]  # method -> JSON Schema for params
    method_validators: dict[str, Any]  # method -> pre-built Draft202012Validator
    rubric_default: dict[str, float]
    rubric_presets: dict[str, dict[str, float]]
    rubric_aggregation: str

    def method_params_schema(self, method: str) -> dict[str, Any]:
        """Return the JSON Schema for a method's params block. Raises KeyError."""
        return self.method_schemas[method]


def load_schemas(schemas_dir: Path) -> Schemas:
    """Load every file in schemas/ and return a Schemas snapshot."""
    schemas_dir = Path(schemas_dir)

    state_machine = yaml.safe_load((schemas_dir / "state-machine.yaml").read_text())
    methods_doc = json.loads((schemas_dir / "jsonrpc-methods.json").read_text())
    rubric_doc = yaml.safe_load((schemas_dir / "reward-rubric.yaml").read_text())

    transitions: dict[tuple[str, str], str] = {}
    for entry in state_machine["transitions"]:
        transitions[(entry["from"], entry["trigger"])] = entry["to"]

    method_schemas = {name: spec["params"] for name, spec in methods_doc["methods"].items()}

    from jsonschema import Draft202012Validator, FormatChecker  # noqa: PLC0415

    # FormatChecker enables runtime validation of `format` keywords (e.g.,
    # `format: date-time` in ticket_history.window). Without it, format is
    # annotation-only and any string passes silently.
    _fc = FormatChecker()
    method_validators = {
        name: Draft202012Validator(schema, format_checker=_fc)
        for name, schema in method_schemas.items()
    }

    return Schemas(
        states=frozenset(state_machine["states"]),
        initial_state=state_machine["initial_state"],
        terminal_states=frozenset(state_machine["terminal_states"]),
        transitions=transitions,
        method_triggers=dict(state_machine["method_triggers"]),
        method_names=frozenset(method_schemas.keys()),
        method_schemas=method_schemas,
        method_validators=method_validators,
        rubric_default=dict(rubric_doc["default_weights"]),
        rubric_presets={k: dict(v) for k, v in rubric_doc["presets"].items()},
        rubric_aggregation=rubric_doc["aggregation"],
    )
