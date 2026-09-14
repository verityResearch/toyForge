"""Tests for scripts.tokenizer_fragmentation_audit."""

from __future__ import annotations

from scripts.tokenizer_fragmentation_audit import audit_strings


class _MockTokenizer:
    """Mimics the minimal HF tokenizer surface: __call__ → {'input_ids': [...]}.

    Splits on each character so 'foo' → ['f', 'o', 'o']; this exaggerates
    fragmentation so the test can assert ≥ 2 tokens easily.
    """

    def __call__(self, text: str, add_special_tokens: bool = False):
        ids = [ord(c) for c in text]
        return {"input_ids": ids}

    def decode(self, token_ids):
        return "".join(chr(i) for i in token_ids)


def test_audit_counts_tokens():
    tok = _MockTokenizer()
    out = audit_strings(tok, ["abc", "x"])
    assert out["abc"]["token_count"] == 3
    assert out["x"]["token_count"] == 1


def test_audit_flags_multi_token_strings():
    tok = _MockTokenizer()
    out = audit_strings(tok, ["abc", "x"])
    assert out["abc"]["fragmented"] is True
    assert out["x"]["fragmented"] is False


def test_audit_decodes_tokens():
    tok = _MockTokenizer()
    out = audit_strings(tok, ["abc"])
    assert "".join(out["abc"]["surface_pieces"]) == "abc"


def test_render_markdown_includes_interpretation_section():
    """The audit output ends with an Interpretation section."""
    from scripts.tokenizer_fragmentation_audit import render_markdown

    audit = {
        "ticket_open": {
            "token_count": 1,
            "token_ids": [42],
            "surface_pieces": ["ticket_open"],
            "fragmented": False,
        },
        "requester_id": {
            "token_count": 2,
            "token_ids": [10, 20],
            "surface_pieces": ["requester", "_id"],
            "fragmented": True,
        },
    }
    md = render_markdown(audit, "test-model")
    assert "## Interpretation" in md
    # 1 of 2 = 50% fragmented → "High fragmentation" message
    assert "fragmentation" in md.lower()


def test_render_markdown_zero_fragmentation():
    from scripts.tokenizer_fragmentation_audit import render_markdown

    audit = {
        "ticket_open": {
            "token_count": 1,
            "token_ids": [42],
            "surface_pieces": ["ticket_open"],
            "fragmented": False,
        },
    }
    md = render_markdown(audit, "test-model")
    assert "No fragmentation detected" in md


def test_render_markdown_empty_audit():
    """Edge case: no strings audited."""
    from scripts.tokenizer_fragmentation_audit import render_markdown

    md = render_markdown({}, "test-model")
    assert "0 of 0" in md  # the interpretation reports counts
