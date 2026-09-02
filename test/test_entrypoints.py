"""
Tests for the three CITS entry points.

  1. base CITS (cits.methods.cits_full)      -- CPU, always runs
  2. GPU CITS (cits.cits_gpu)                 -- needs cuPC; skipped otherwise
  3. Version-B CITS (cits.cits_versionb)      -- needs cuPC; skipped otherwise

The GPU / Version-B tests skip (rather than fail) when cuPC is unavailable,
so the suite passes on a CPU-only machine. A GPU is NOT required for the base
tests to pass.
"""

import numpy as np
import pytest


def _toy_series(p=4, T=800, seed=0):
    """Small linear lag-1 chain X0 -> X2 <- X1, X2 -> X3, shape (p, T)."""
    rng = np.random.default_rng(seed)
    X = np.zeros((p, T))
    X[:, 0] = rng.standard_normal(p)
    for t in range(1, T):
        X[0, t] = rng.standard_normal() + 1.0
        X[1, t] = rng.standard_normal() - 1.0
        X[2, t] = 2.0 * X[0, t - 1] + 1.0 * X[1, t - 1] + rng.standard_normal()
        X[3, t] = 2.0 * X[2, t - 1] + rng.standard_normal()
        for extra in range(4, p):
            X[extra, t] = rng.standard_normal()
    return X


def _cupc_available():
    """True iff cuPC Skeleton.so can be loaded."""
    try:
        from cits._cupc_wrapper import _get_lib
        _get_lib()
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
#  Import / export tests (no GPU needed)
# ---------------------------------------------------------------------------

def test_import_cits():
    import cits
    exported = [n for n in dir(cits) if not n.startswith("__")]
    for name in ("methods", "simulate_timeseries", "cits_gpu", "cits_versionb"):
        assert name in exported, f"{name} not exported from cits ({exported})"


def test_base_entrypoint_importable():
    from cits import methods
    assert hasattr(methods, "cits_full")
    assert hasattr(methods, "cits_full_weighted")


def test_gpu_entrypoint_importable():
    import cits
    assert callable(cits.cits_gpu)


def test_versionb_entrypoint_importable():
    import cits
    assert callable(cits.cits_versionb)


# ---------------------------------------------------------------------------
#  Base CITS runs on CPU
# ---------------------------------------------------------------------------

def test_base_cits_runs_cpu():
    from cits import methods
    X = _toy_series()
    adj = methods.cits_full(X, 1, 0.05)
    adj = np.asarray(adj)
    assert adj.shape == (X.shape[0], X.shape[0])


# ---------------------------------------------------------------------------
#  GPU / Version-B: skip when cuPC unavailable
# ---------------------------------------------------------------------------

def test_gpu_cits_runs_if_cupc():
    if not _cupc_available():
        pytest.skip("cuPC (Skeleton.so) unavailable; set CUPC_DIR and build it")
    import cits
    X = _toy_series()
    B = cits.cits_gpu(X, alpha=0.05, tau=1)
    B = np.asarray(B)
    assert B.shape == (X.shape[0], X.shape[0])


def test_versionb_cits_runs_if_cupc():
    if not _cupc_available():
        pytest.skip("cuPC (Skeleton.so) unavailable; set CUPC_DIR and build it")
    import cits
    X = _toy_series()
    B = cits.cits_versionb(X, alpha=0.05, tau=1)
    B = np.asarray(B)
    assert B.shape == (X.shape[0], X.shape[0])


def test_gpu_call_raises_clearly_without_cupc():
    """When cuPC is unavailable, calling cits_gpu must raise a clear error
    mentioning cuPC / CUPC_DIR (not a bare ImportError from `import cits`)."""
    if _cupc_available():
        pytest.skip("cuPC available; clear-error path not exercised")
    import cits
    X = _toy_series()
    with pytest.raises(Exception) as excinfo:
        cits.cits_gpu(X, alpha=0.05, tau=1)
    msg = str(excinfo.value)
    assert "cuPC" in msg or "CUPC_DIR" in msg or "Skeleton.so" in msg
