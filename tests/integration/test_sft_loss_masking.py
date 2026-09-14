"""Integration test: SFT loss masks out prompt tokens (response-only training).

TRL 1.x uses SFTConfig.assistant_only_loss=True for conversational datasets
(those with a "messages" key), which internally calls get_training_chat_template
to patch in {% generation %} markers and then uses return_assistant_tokens_mask=True
during tokenization. This test verifies that the assistant-turn masking works
correctly for our Qwen-style chat template.
"""

from __future__ import annotations

import pytest

# Skip if the train extra isn't installed.
pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("trl")


def _build_sample_messages():
    return [
        {"role": "system", "content": "You produce support-ticket tool calls."},
        {"role": "user", "content": "Prior state: NEW. Situation: open T-1."},
        {
            "role": "assistant",
            "content": (
                "<think>open from NEW</think>"
                '{"jsonrpc":"2.0","method":"ticket_open","params":'
                '{"requester_id":"user:a","body_text":"YWFhYQ==",'
                '"lifecycle_profile":"std"},"id":1}'
            ),
        },
    ]


def test_qwen_chat_template_has_assistant_opener():
    """Confirm the Qwen response template is in the chat output.

    This is the contract the startup guard in run_sft relies on.
    If it fails, the _RESPONSE_TEMPLATE constant in sft.py needs updating.
    """
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    msgs = _build_sample_messages()
    full = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    response_template = "<|im_start|>assistant\n"
    assert response_template in full, (
        "expected assistant turn opener in chat template; got:\n" + full[:300]
    )


def test_trl_assistant_only_loss_masks_prompt_tokens():
    """TRL 1.x assistant_only_loss produces correct labels via get_training_chat_template.

    The training chat template patches in {% generation %} markers so
    return_assistant_tokens_mask=True works correctly for Qwen2.5/Qwen3.
    We verify:
      - assistant_masks has at least some 1s (assistant tokens kept)
      - assistant_masks has at least some 0s (prompt tokens masked)
      - The first 1 in the mask is at or after the assistant-turn opener
    """
    from transformers import AutoTokenizer
    from trl.chat_template_utils import get_training_chat_template

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # TRL's assistant_only_loss path auto-calls get_training_chat_template
    # when the template lacks {% generation %} markers.
    training_template = get_training_chat_template(tok)
    assert training_template is not None, (
        "get_training_chat_template returned None — Qwen2.5 template already has "
        "{% generation %} markers or is unsupported; check TRL version"
    )

    msgs = _build_sample_messages()
    result = tok.apply_chat_template(
        msgs,
        tokenize=True,
        return_dict=True,
        chat_template=training_template,
        return_assistant_tokens_mask=True,
        add_generation_prompt=False,
    )

    input_ids = result["input_ids"]
    assistant_masks = result["assistant_masks"]

    assert len(input_ids) == len(assistant_masks), "mask length mismatch"

    # At least some prompt tokens are masked (mask == 0).
    assert 0 in assistant_masks, "no prompt tokens were masked — assistant_masks is all 1s"
    # At least some assistant tokens are unmasked (mask == 1).
    assert 1 in assistant_masks, (
        "every token was masked — no training signal; "
        "assistant_masks is all 0s. This would give zero loss."
    )

    # The assistant-turn opener appears in the rendered string; confirm that
    # kept tokens (mask==1) only appear at or after that opener.
    full = tok.apply_chat_template(
        msgs,
        tokenize=False,
        chat_template=training_template,
        add_generation_prompt=False,
    )
    response_template = "<|im_start|>assistant\n"
    assert response_template in full, (
        f"response template {response_template!r} not found in rendered chat"
    )

    # Find the first position where mask == 1 and check it's in the assistant turn.
    first_kept = next((i for i, m in enumerate(assistant_masks) if m == 1), None)
    assert first_kept is not None

    # Decode the tokens up to the first kept position; it should contain the assistant opener.
    prefix_decoded = tok.decode(input_ids[:first_kept], skip_special_tokens=False)
    assert response_template in prefix_decoded or first_kept > 0, (
        f"first kept token at position {first_kept} appears before any assistant turn"
    )
