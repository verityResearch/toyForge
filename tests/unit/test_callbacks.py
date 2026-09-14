"""Tests for SubscoreLoggerCallback."""

from __future__ import annotations

import pytest

# Callback depends on transformers.TrainerCallback; skip on CPU-only envs.
pytest.importorskip("transformers")

from toyforge.train.callbacks import SubscoreLoggerCallback
from toyforge.verifier.types import Subscores


def _sub(transition_valid: float = 1.0, parse: float = 1.0) -> Subscores:
    return Subscores(
        parse=parse,
        schema=1.0,
        method_known=1.0,
        precondition_met=1.0,
        transition_valid=transition_valid,
        sequence_optimal=transition_valid,
    )


def test_callback_buffers_subscores_and_emits_window_mean():
    """record() appends; on_log() injects per-subscore mean into logs dict."""
    cb = SubscoreLoggerCallback(window_size=10)
    # 3 passes + 1 fail on transition_valid
    cb.record([_sub(1.0), _sub(1.0), _sub(1.0), _sub(0.0)])
    logs: dict = {}
    cb.on_log(args=None, state=None, control=None, logs=logs)
    # Window mean: 3 of 4 transition_valid = 0.75
    assert logs["subscore_mean/transition_valid"] == pytest.approx(0.75)
    # parse stayed at 1.0 across all 4
    assert logs["subscore_mean/parse"] == 1.0
    # All 6 subscore names should be present
    for name in [
        "parse",
        "schema",
        "method_known",
        "precondition_met",
        "transition_valid",
        "sequence_optimal",
    ]:
        assert f"subscore_mean/{name}" in logs


def test_callback_silent_when_buffer_empty():
    """A fresh callback with no record() calls produces no log entries."""
    cb = SubscoreLoggerCallback(window_size=10)
    logs: dict = {}
    cb.on_log(args=None, state=None, control=None, logs=logs)
    assert logs == {}  # nothing injected when no data


def test_callback_handles_none_logs():
    """If TRL passes logs=None, the callback must not crash."""
    cb = SubscoreLoggerCallback(window_size=10)
    cb.record([_sub(1.0)])
    # Should not raise.
    cb.on_log(args=None, state=None, control=None, logs=None)


def test_callback_respects_window_size():
    """A small window means older subscores are evicted."""
    cb = SubscoreLoggerCallback(window_size=3)
    # 5 entries; only the last 3 are retained
    cb.record([_sub(0.0), _sub(0.0), _sub(1.0), _sub(1.0), _sub(1.0)])
    logs: dict = {}
    cb.on_log(args=None, state=None, control=None, logs=logs)
    # Last 3 transition_valid were all 1.0
    assert logs["subscore_mean/transition_valid"] == 1.0


def test_callback_batch_with_one_item():
    """record() accepts a 1-item batch."""
    cb = SubscoreLoggerCallback(window_size=10)
    cb.record([_sub(0.5)])
    logs: dict = {}
    cb.on_log(args=None, state=None, control=None, logs=logs)
    assert logs["subscore_mean/transition_valid"] == 0.5
