"""Property-test scaffolding.

Configures Hypothesis settings profiles so CI runs stay thorough
and local dev stays snappy. Tests import strategies from
tests/property/strategies.py (created in Task 1).
"""

from __future__ import annotations

from hypothesis import HealthCheck, settings

# Two profiles: 'dev' for local iteration (~25 examples), 'ci' for thorough runs.
settings.register_profile(
    "dev",
    max_examples=25,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
settings.register_profile(
    "ci",
    max_examples=200,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
settings.load_profile("dev")
