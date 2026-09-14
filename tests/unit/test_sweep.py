"""Tests for scripts/sweep.py — config generation and Cartesian product."""

from __future__ import annotations

import pytest

from scripts.sweep import _slug, _unflatten, generate_sweep_configs, run_sweep


def test_unflatten_dotted_keys_to_nested():
    out = _unflatten({"lora.r": 64, "lora.alpha": 128, "seed": 42})
    assert out == {"lora": {"r": 64, "alpha": 128}, "seed": 42}


def test_unflatten_raises_on_path_collision():
    """If a dotted axis name conflicts with a scalar from another axis,
    raise rather than silently overwrite."""
    with pytest.raises(ValueError, match=r"collision"):
        # `lora` is already a scalar, but `lora.r` tries to make it a dict
        _unflatten({"lora": 64, "lora.r": 32})


def test_unflatten_raises_on_reverse_order_collision():
    """The reverse ordering — dotted key first, then scalar — must also raise,
    not silently overwrite the nested dict with the scalar."""
    with pytest.raises(ValueError, match=r"collision"):
        _unflatten({"lora.r": 32, "lora": 64})


def test_slug_uses_full_dotted_key():
    """Slug uses the full dotted axis name (with '.' -> '_') to avoid collisions
    between axes that share a key tail."""
    slug = _slug({"lora.r": 64, "seed": 42})
    assert slug == "lora_r64-seed42"  # was "r64-seed42" before the fix


def test_slug_disambiguates_shared_key_tails():
    """Two axes sharing a key tail produce distinct slugs."""
    slug = _slug({"lora.r": 8, "schedule.r": 16})
    # Without the fix, both contributed 'r{val}' and would collide.
    assert "lora_r8" in slug
    assert "schedule_r16" in slug


def test_generate_sweep_configs_cartesian_product(tmp_path):
    """N axes with [a,b] and [x,y,z] values produce 6 configs."""
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(
        "base: ../phase1-sft.yaml\naxes:\n  lora.r: [8, 16]\n  seed: [42, 7, 123]\n"
    )
    out_dir = tmp_path / "generated"
    configs = generate_sweep_configs(spec_path, out_dir)
    assert len(configs) == 6
    # Each config has extends + the override
    import yaml as y

    for cfg in configs:
        d = y.safe_load(cfg.read_text())
        assert d["extends"] == "../phase1-sft.yaml"
        assert "lora" in d
        assert "seed" in d


def test_generate_sweep_configs_missing_base_raises(tmp_path):
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text("axes:\n  lora.r: [8]\n")  # no base
    with pytest.raises(ValueError, match=r"base"):
        generate_sweep_configs(spec_path, tmp_path / "gen")


def test_generate_sweep_configs_empty_axes_raises(tmp_path):
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text("base: ../phase1-sft.yaml\n")  # no axes
    with pytest.raises(ValueError, match=r"axes"):
        generate_sweep_configs(spec_path, tmp_path / "gen")


def test_slug_replaces_unsafe_characters():
    """Slugs strip non-alphanumeric characters from values to stay filesystem-safe."""
    slug = _slug({"base_model": "Qwen/Qwen3-8B-Instruct:latest"})
    # Replaces /, :, and . with underscores
    assert "/" not in slug
    assert ":" not in slug
    assert "Qwen_Qwen3-8B-Instruct_latest" in slug


def test_run_sweep_returns_failed_true_on_subprocess_failure(tmp_path, monkeypatch):
    """When `just train` exits non-zero, run_sweep returns (completed, failed=True)."""
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text("base: ../phase1-sft.yaml\naxes:\n  lora.r: [8, 16, 32]\n")

    # Stub: first call succeeds (returncode 0), second call fails (returncode 1)
    call_count = {"n": 0}

    class FakeResult:
        def __init__(self, rc):
            self.returncode = rc

    def fake_run(*args, **kwargs):
        call_count["n"] += 1
        return FakeResult(0 if call_count["n"] == 1 else 1)

    monkeypatch.setattr("subprocess.run", fake_run)
    monkeypatch.chdir(tmp_path)

    completed, failed = run_sweep(spec_path, dry_run=False)
    assert failed is True
    assert completed == 1  # only the first succeeded


def test_run_sweep_returns_failed_false_on_all_success(tmp_path, monkeypatch):
    """When all configs succeed, run_sweep returns (n_configs, failed=False)."""
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text("base: ../phase1-sft.yaml\naxes:\n  lora.r: [8, 16]\n")

    class FakeResult:
        returncode = 0

    monkeypatch.setattr("subprocess.run", lambda *a, **k: FakeResult())
    monkeypatch.chdir(tmp_path)

    completed, failed = run_sweep(spec_path, dry_run=False)
    assert failed is False
    assert completed == 2


def test_run_sweep_dry_run_does_not_invoke_subprocess(tmp_path, monkeypatch):
    """In dry-run mode, run_sweep generates configs but doesn't call `just train`."""
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text("base: ../phase1-sft.yaml\naxes:\n  lora.r: [8, 16, 32]\n")

    subprocess_calls = []

    def fake_run(*args, **kwargs):
        subprocess_calls.append(args)

        class _R:
            returncode = 0

        return _R()

    monkeypatch.setattr("subprocess.run", fake_run)
    monkeypatch.chdir(tmp_path)

    completed, failed = run_sweep(spec_path, dry_run=True)
    # In dry-run, no subprocess call should be made
    assert subprocess_calls == []
    assert failed is False
    # completed equals the count of configs generated
    assert completed == 3


def test_generate_sweep_loads_back_through_inheritance(tmp_path):
    """A generated sweep config must load successfully via load_train_config
    (i.e., the extends path resolution actually works)."""
    # Build a tmp env: tmp_path/train_configs/phase1-sft.yaml as base
    tc_dir = tmp_path / "train_configs"
    tc_dir.mkdir()
    base = tc_dir / "phase1-sft.yaml"
    base.write_text(
        "base_model: Qwen/Qwen2.5-0.5B-Instruct\n"
        "method: vanilla\n"
        "lora:\n  r: 32\n  alpha: 64\n  target_modules: [q_proj]\n  dropout: 0.05\n"
    )

    spec_path = tmp_path / "sweep.yaml"
    spec_path.write_text("base: ../phase1-sft.yaml\naxes:\n  lora.r: [64]\n")
    gen_dir = tc_dir / ".generated"
    configs = generate_sweep_configs(spec_path, gen_dir)
    assert len(configs) == 1

    from toyforge.train.config import load_train_config

    cfg = load_train_config(configs[0])
    assert cfg.base_model == "Qwen/Qwen2.5-0.5B-Instruct"
    assert cfg.lora.r == 64  # override
    assert cfg.lora.alpha == 64  # inherited
