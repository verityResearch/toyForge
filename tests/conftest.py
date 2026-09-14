"""Shared pytest fixtures for toyForge tests."""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMAS_DIR = REPO_ROOT / "schemas"
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures"


@pytest.fixture(scope="session")
def schemas_dir() -> Path:
    """Path to the canonical schemas/ directory."""
    return SCHEMAS_DIR


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    """Path to tests/fixtures/."""
    return FIXTURES_DIR
