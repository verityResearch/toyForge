"""Differential: C `jsonschema-validate` vs Python's Draft202012Validator.

Covers the keyword subset the C validator implements (type/enum/const,
combinators, if/then/else, object/array/string/number constraints, local
$ref/$defs). Cases using keywords the C validator does not yet implement
(pattern/patternProperties, unevaluated*, format-assertion, non-local $ref) are
intentionally excluded because it treats those as pass-through annotations.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

# (schema, instance) pairs spanning the supported keywords.
CASES: list[tuple[dict | bool, Any]] = [
    (True, 5),
    (False, 5),
    ({"type": "integer"}, 5),
    ({"type": "integer"}, 5.0),
    ({"type": "integer"}, 5.5),
    ({"type": "string"}, 5),
    ({"type": ["string", "number"]}, "a"),
    ({"type": ["string", "number"]}, True),
    ({"type": "null"}, None),
    ({"type": "boolean"}, False),
    ({"enum": [1, 2, 3]}, 2),
    ({"enum": [1, 2, 3]}, 4),
    ({"enum": ["a", {"k": 1}]}, {"k": 1}),
    ({"enum": ["a", {"k": 1}]}, {"k": 2}),
    ({"const": "2.0"}, "2.0"),
    ({"const": "2.0"}, "2.1"),
    ({"const": {"a": [1, 2]}}, {"a": [1, 2]}),
    ({"minimum": 3, "maximum": 7}, 5),
    ({"minimum": 3}, 2),
    ({"maximum": 7}, 8),
    ({"exclusiveMinimum": 3}, 3),
    ({"exclusiveMaximum": 7}, 7),
    ({"multipleOf": 2}, 8),
    ({"multipleOf": 2}, 7),
    ({"multipleOf": 0.5}, 1.5),
    ({"minLength": 2, "maxLength": 4}, "abc"),
    ({"minLength": 2}, "a"),
    ({"maxLength": 2}, "abc"),
    ({"minLength": 2}, "éé"),  # 2 code points, not bytes
    ({"minItems": 1, "maxItems": 2}, [1]),
    ({"minItems": 2}, [1]),
    ({"maxItems": 1}, [1, 2]),
    ({"uniqueItems": True}, [1, 2, 1]),
    ({"uniqueItems": True}, [1, 2, 3]),
    ({"uniqueItems": True}, [{"a": 1}, {"a": 1}]),
    ({"items": {"type": "integer"}}, [1, 2, 3]),
    ({"items": {"type": "integer"}}, [1, "a"]),
    ({"prefixItems": [{"type": "string"}, {"type": "integer"}]}, ["a", 1]),
    ({"prefixItems": [{"type": "string"}]}, [1]),
    ({"prefixItems": [{"type": "string"}], "items": {"type": "integer"}}, ["a", 1, 2]),
    ({"prefixItems": [{"type": "string"}], "items": {"type": "integer"}}, ["a", "b"]),
    ({"required": ["a"]}, {"a": 1}),
    ({"required": ["a"]}, {"b": 1}),
    ({"properties": {"a": {"type": "integer"}}}, {"a": 1}),
    ({"properties": {"a": {"type": "integer"}}}, {"a": "x"}),
    ({"additionalProperties": False}, {"a": 1}),
    ({"properties": {"a": {}}, "additionalProperties": False}, {"a": 1}),
    ({"properties": {"a": {}}, "additionalProperties": False}, {"a": 1, "b": 2}),
    ({"additionalProperties": {"type": "integer"}}, {"a": "x"}),
    ({"additionalProperties": {"type": "integer"}}, {"a": 1}),
    ({"minProperties": 2}, {"a": 1}),
    ({"maxProperties": 1}, {"a": 1, "b": 2}),
    ({"anyOf": [{"type": "string"}, {"type": "integer"}]}, 5),
    ({"anyOf": [{"type": "string"}, {"type": "boolean"}]}, 5),
    ({"oneOf": [{"type": "integer"}, {"minimum": 3}]}, 5),
    ({"oneOf": [{"type": "string"}, {"type": "integer"}]}, 5),
    ({"allOf": [{"type": "integer"}, {"minimum": 3}]}, 5),
    ({"allOf": [{"type": "integer"}, {"minimum": 7}]}, 5),
    ({"not": {"type": "string"}}, 5),
    ({"not": {"type": "integer"}}, 5),
    # if / then / else
    ({"if": {"type": "integer"}, "then": {"minimum": 3}}, 5),
    ({"if": {"type": "integer"}, "then": {"minimum": 7}}, 5),
    ({"if": {"type": "string"}, "then": {"minLength": 9}, "else": {"minimum": 3}}, 5),
    ({"if": {"type": "string"}, "then": {"minLength": 9}, "else": {"minimum": 7}}, 5),
    # contains / minContains / maxContains
    ({"contains": {"type": "integer"}}, ["a", 1, "b"]),
    ({"contains": {"type": "integer"}}, ["a", "b"]),
    ({"contains": {"const": 2}, "minContains": 2}, [2, 2, 3]),
    ({"contains": {"const": 2}, "minContains": 2}, [2, 3]),
    ({"contains": {"const": 2}, "maxContains": 1}, [2, 2]),
    # propertyNames
    ({"propertyNames": {"minLength": 2}}, {"ab": 1, "cd": 2}),
    ({"propertyNames": {"minLength": 2}}, {"a": 1}),
    # dependentRequired
    ({"dependentRequired": {"a": ["b"]}}, {"a": 1, "b": 2}),
    ({"dependentRequired": {"a": ["b"]}}, {"a": 1}),
    ({"dependentRequired": {"a": ["b"]}}, {"c": 1}),
    # dependentSchemas
    ({"dependentSchemas": {"a": {"required": ["b"]}}}, {"a": 1, "b": 2}),
    ({"dependentSchemas": {"a": {"required": ["b"]}}}, {"a": 1}),
    # pattern (regex) — supported subset, fuzz-confirmed against Python re.search
    ({"pattern": "^a"}, "abc"),
    ({"pattern": "^a"}, "xabc"),
    ({"pattern": "[0-9]{2,}"}, "x99y"),
    ({"pattern": "[0-9]{2,}"}, "x9y"),
    ({"pattern": "^.*bar$"}, "foobar"),
    ({"pattern": "^[a-z_]+$"}, "snake_case"),
    ({"pattern": "^[a-z_]+$"}, "NotLower"),
    ({"pattern": "a{2,3}"}, "aaaa"),  # bounded quantifier via search
    ({"pattern": "a{2,3}"}, "a"),
    ({"pattern": "colou?r"}, "color"),  # optional
    ({"pattern": "colou?r"}, "colour"),
    ({"pattern": r"\w+"}, "x__1"),  # shorthand word class
    ({"pattern": r"^\d{1,3}$"}, "999"),
    ({"pattern": r"^\d{1,3}$"}, "9999"),
    ({"pattern": "[^0-9]+"}, "abc"),  # negated class
    ({"pattern": "[^0-9]+"}, "123"),
    ({"pattern": "^$"}, ""),  # empty anchors
    ({"pattern": "^$"}, "x"),
    # patternProperties (regex)
    ({"type": "object", "patternProperties": {"^x": {"type": "integer"}}}, {"x1": 5}),
    ({"type": "object", "patternProperties": {"^x": {"type": "integer"}}}, {"x1": "no"}),
    (
        {
            "patternProperties": {"^f": {"type": "integer"}},
            "additionalProperties": False,
        },
        {"foo": 1},
    ),
    (
        {
            "patternProperties": {"^f": {"type": "integer"}},
            "additionalProperties": False,
        },
        {"bar": 1},
    ),
    # $ref / $defs (local JSON Pointer)
    ({"$defs": {"pos": {"type": "integer", "minimum": 0}}, "$ref": "#/$defs/pos"}, 5),
    ({"$defs": {"pos": {"type": "integer", "minimum": 0}}, "$ref": "#/$defs/pos"}, -1),
    ({"$defs": {"pos": {"type": "integer", "minimum": 0}}, "$ref": "#/$defs/pos"}, "x"),
    (
        {"$defs": {"s": {"type": "string"}}, "properties": {"a": {"$ref": "#/$defs/s"}}},
        {"a": "x"},
    ),
    (
        {"$defs": {"s": {"type": "string"}}, "properties": {"a": {"$ref": "#/$defs/s"}}},
        {"a": 1},
    ),
    # $ref with a sibling keyword (Draft 2020-12 applies both)
    ({"$defs": {"i": {"type": "integer"}}, "$ref": "#/$defs/i", "minimum": 3}, 5),
    ({"$defs": {"i": {"type": "integer"}}, "$ref": "#/$defs/i", "minimum": 3}, 2),
    # $id base URI + $ref to another $id'd subschema, and $anchor
    (
        {
            "$id": "http://ex.com/s",
            "$defs": {"a": {"$id": "http://ex.com/a", "type": "integer"}},
            "$ref": "http://ex.com/a",
        },
        5,
    ),
    (
        {
            "$id": "http://ex.com/s",
            "$defs": {"a": {"$id": "http://ex.com/a", "type": "integer"}},
            "$ref": "http://ex.com/a",
        },
        "x",
    ),
    ({"$defs": {"a": {"$anchor": "foo", "type": "integer"}}, "$ref": "#foo"}, 5),
    ({"$defs": {"a": {"$anchor": "foo", "type": "integer"}}, "$ref": "#foo"}, "x"),
    # $dynamicRef / $dynamicAnchor (no dynamic-scope override: resolves like $ref
    # to the lone $dynamicAnchor — the common case; full outermost-override is not
    # modeled and is documented-skipped in the official suite runner)
    ({"$defs": {"f": {"$dynamicAnchor": "x", "type": "integer"}}, "$dynamicRef": "#x"}, 5),
    ({"$defs": {"f": {"$dynamicAnchor": "x", "type": "integer"}}, "$dynamicRef": "#x"}, "a"),
    ({"$defs": {"f": {"$dynamicAnchor": "x", "minimum": 3}}, "$dynamicRef": "#x"}, 2),
    (
        {
            "$defs": {"s": {"$dynamicAnchor": "y", "type": "string"}},
            "properties": {"a": {"$dynamicRef": "#y"}},
        },
        {"a": "ok"},
    ),
    (
        {
            "$defs": {"s": {"$dynamicAnchor": "y", "type": "string"}},
            "properties": {"a": {"$dynamicRef": "#y"}},
        },
        {"a": 1},
    ),
    # unevaluatedProperties (annotation collection across in-place applicators)
    ({"properties": {"a": {}}, "unevaluatedProperties": False}, {"a": 1}),
    ({"properties": {"a": {}}, "unevaluatedProperties": False}, {"a": 1, "b": 2}),
    ({"allOf": [{"properties": {"a": {}}}], "unevaluatedProperties": False}, {"a": 1}),
    ({"allOf": [{"properties": {"a": {}}}], "unevaluatedProperties": False}, {"a": 1, "b": 2}),
    ({"unevaluatedProperties": {"type": "integer"}}, {"a": 1}),
    ({"unevaluatedProperties": {"type": "integer"}}, {"a": "x"}),
    # unevaluatedItems
    ({"prefixItems": [{}], "unevaluatedItems": False}, [1]),
    ({"prefixItems": [{}], "unevaluatedItems": False}, [1, 2]),
    ({"prefixItems": [{}], "unevaluatedItems": {"type": "integer"}}, [1, 2]),
    ({"prefixItems": [{}], "unevaluatedItems": {"type": "integer"}}, [1, "x"]),
    ({"contains": {"const": 2}, "unevaluatedItems": False}, [2]),
    # recursive $ref, terminates on a finite instance
    (
        {
            "$defs": {"node": {"type": "object", "properties": {"next": {"$ref": "#/$defs/node"}}}},
            "$ref": "#/$defs/node",
        },
        {"next": {"next": {}}},
    ),
    (
        {
            "type": "object",
            "required": ["did", "n"],
            "properties": {"did": {"type": "string", "minLength": 4}, "n": {"type": "integer"}},
            "additionalProperties": False,
        },
        {"did": "did:x", "n": 3},
    ),
    # objects with more than 8 properties (exercise the raised key cap)
    ({"type": "object"}, {f"k{i}": i for i in range(20)}),
    (
        {"properties": {"k0": {"type": "integer"}}, "additionalProperties": {"type": "integer"}},
        {f"k{i}": i for i in range(12)},
    ),
    ({"additionalProperties": False}, {f"k{i}": i for i in range(12)}),
    ({"maxProperties": 5}, {f"k{i}": i for i in range(12)}),
    ({"unevaluatedProperties": False}, {f"k{i}": i for i in range(12)}),
    # array with >128 items fully covered by prefixItems (exercises the raised
    # evaluated-item cap; positions past the old 128 bound were falsely flagged)
    ({"prefixItems": [{} for _ in range(130)], "unevaluatedItems": False}, list(range(130))),
    # huge integer literals beyond double/long-long range are still integers
    # (text-based integrality check, not a numeric cast)
    ({"type": "integer"}, 10**20),
    ({"type": "integer"}, -(10**20)),
    ({"type": ["integer", "null"]}, 12345678901234567890),
    ({"type": "integer"}, 1.5),
    # annotation propagation of unevaluated* through applicators (the intricate
    # path) — locked from the differential-fuzz audit
    (
        {
            "anyOf": [{"properties": {"a": {}}}, {"properties": {"b": {}}}],
            "unevaluatedProperties": False,
        },
        {"a": 1, "b": 2},
    ),
    (
        {
            "if": {"properties": {"t": {"const": 1}}, "required": ["t"]},
            "then": {"properties": {"x": {}}},
            "unevaluatedProperties": False,
        },
        {"t": 1, "x": 2},
    ),
    (
        {
            "$defs": {"a": {"properties": {"a": {}}}},
            "$ref": "#/$defs/a",
            "unevaluatedProperties": False,
        },
        {"a": 1, "b": 2},
    ),
    (
        {"dependentSchemas": {"a": {"properties": {"b": {}}}}, "unevaluatedProperties": False},
        {"a": 1, "b": 2},
    ),
    # 'not' must NOT contribute annotations
    ({"not": {"properties": {"a": {}}}, "unevaluatedProperties": False}, {"b": 1}),
    ({"anyOf": [{"prefixItems": [{}, {}]}], "unevaluatedItems": False}, [1, 2]),
    ({"contains": {"const": 2}, "unevaluatedItems": False}, [1, 2]),
    # uniqueItems numeric/type equality (1 == 1.0, but 1 != true)
    ({"uniqueItems": True}, [1, 1.0]),
    ({"uniqueItems": True}, [1, True]),
    ({"uniqueItems": True}, [{"a": 1}, {"a": 1.0}]),
    # const/enum deep equality with number normalization
    ({"const": {"a": 1}}, {"a": 1.0}),
    ({"enum": [1, 2]}, 1.0),
    # propertyNames with a pattern
    ({"propertyNames": {"pattern": "^[a-z]+$"}}, {"abc": 1, "A1": 2}),
]


def _c_valid(toyforge_c: Path, schema: dict | bool, instance: Any, tmp: Path) -> bool:
    sp = tmp / "schema.json"
    ip = tmp / "instance.json"
    sp.write_text(json.dumps(schema))
    ip.write_text(json.dumps(instance))
    proc = subprocess.run(
        [str(toyforge_c), "jsonschema-validate", "--schema", str(sp), "--instance", str(ip)],
        check=False,
        text=True,
        capture_output=True,
    )
    if proc.returncode not in (0, 1):
        raise RuntimeError(f"C validate error: {proc.stderr or proc.stdout}")
    return proc.returncode == 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--toyforge-c", type=Path, default=Path("c/build/toyforge-c"))
    args = parser.parse_args()

    failures: list[str] = []
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for idx, (schema, instance) in enumerate(CASES):
            py_valid = Draft202012Validator(schema).is_valid(instance)
            c_valid = _c_valid(args.toyforge_c, schema, instance, tmp)
            if py_valid != c_valid:
                failures.append(
                    f"case {idx}: schema={json.dumps(schema)} instance={json.dumps(instance)} "
                    f"Python={py_valid} C={c_valid}"
                )

    if failures:
        print("\n".join(failures))
        return 1
    print(f"{len(CASES)} jsonschema cases match Draft202012Validator")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
