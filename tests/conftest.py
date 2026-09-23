"""Shared fixtures for the release's test suite."""

from pathlib import Path

import numpy as np
import pytest

RELEASE_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def release_root() -> Path:
    return RELEASE_ROOT


@pytest.fixture(scope="session")
def synthetic():
    """One synthetic demonstration with known ground-truth stiffness.

    Session-scoped and returned read-only by convention: generating it is cheap,
    but running the extraction over it is not, so tests that need the EXTRACTED
    result should use the `extracted` fixture rather than re-running extraction.
    """
    from compliance_vla.synthetic import synthetic_demo

    demo, true_k = synthetic_demo(seed=0)
    return demo, true_k


@pytest.fixture(scope="session")
def tight_sigma_f() -> np.ndarray:
    """Noise floor matched to the synthetic generator's own noise amplitude.

    The library default is the real rig's MEASURED floor, which is far above the
    synthetic data's noise and would mask out most of the synthetic contact
    phase. Using it here would make these tests measure the wrong thing.
    """
    return np.array([0.1, 0.1, 0.1, 0.02, 0.02, 0.02])


@pytest.fixture(scope="session")
def extracted(synthetic, tight_sigma_f):
    """The extraction pipeline's output on the synthetic demonstration."""
    from compliance_vla.extraction import ExtractionConfig, extract_demo

    demo, true_k = synthetic
    result = extract_demo(demo, ExtractionConfig(), tight_sigma_f)
    assert result["status"] == "ok", result.get("reason")
    return result, true_k
