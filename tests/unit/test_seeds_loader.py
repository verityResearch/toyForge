"""Tests for the seeds loader."""

from __future__ import annotations

import pytest

from toyforge.scenario_gen.seeds import load_seeds


def test_loads_seed_sample(fixtures_dir):
    seeds = load_seeds(fixtures_dir / "seeds" / "seed_sample.yaml")
    assert len(seeds) == 1
    seed = seeds[0]
    assert seed["trajectory_id"] == "seed_open_basic"
    assert seed["initial_state"] == "NEW"
    assert len(seed["steps"]) == 1
    assert seed["steps"][0]["tool_call"]["method"] == "ticket_open"


def test_load_missing_file_raises(fixtures_dir):
    with pytest.raises(FileNotFoundError):
        load_seeds(fixtures_dir / "seeds" / "does_not_exist.yaml")


def test_seeds_default_source_is_hand_seed(fixtures_dir):
    seeds = load_seeds(fixtures_dir / "seeds" / "seed_sample.yaml")
    assert seeds[0]["source"] == "hand_seed"
