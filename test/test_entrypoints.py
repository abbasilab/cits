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


def test_set_cupc_dir_exported():
    import cits
    assert callable(cits.set_cupc_dir)
    assert "set_cupc_dir" in cits.__all__


def test_set_cupc_dir_rejects_dir_without_skeleton(tmp_path):
    """set_cupc_dir must validate that Skeleton.so is present (no GPU needed)."""
    import cits
    with pytest.raises(FileNotFoundError) as excinfo:
        cits.set_cupc_dir(str(tmp_path))
    assert "Skeleton.so" in str(excinfo.value)


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


# ---------------------------------------------------------------------------
#  CPU backend: always runs, no GPU required
# ---------------------------------------------------------------------------

def test_cpu_skeleton_runs():
    """Pure-numpy PC-stable skeleton runs with no GPU and returns a
    symmetric (p, p) 0/1 adjacency."""
    from cits._pc_skeleton_cpu import pc_skeleton_cpu
    rng = np.random.default_rng(1)
    U = rng.standard_normal((300, 6))
    G, sep, inactive, level = pc_skeleton_cpu(U, alpha=0.05)
    assert G.shape == (6, 6)
    assert np.array_equal(G, G.T)
    assert np.all(np.diag(G) == 0)


def test_versionb_cpu_runs_without_gpu():
    """cits_versionb(backend='cpu') returns a (p, p) result with no GPU."""
    import cits
    X = _toy_series()
    B = cits.cits_versionb(X, alpha=0.05, tau=1, backend='cpu')
    B = np.asarray(B)
    assert B.shape == (X.shape[0], X.shape[0])


def test_cpu_matches_cupc_skeleton():
    """The CPU PC-stable skeleton must equal the cuPC skeleton on the same
    data (skipped when cuPC is unavailable)."""
    if not _cupc_available():
        pytest.skip("cuPC (Skeleton.so) unavailable; equivalence not checked")
    from cits._pc_skeleton_cpu import pc_skeleton_cpu
    from cits._cupc_wrapper import pc_skeleton_cupc
    from cits.gpu import _build_chi_nonoverlap
    from cits import simulate_timeseries
    X, _, _ = simulate_timeseries.simulate('lingauss1', noise=1.0, T=2000)
    U = _build_chi_nonoverlap(X, tau=1)
    G_cpu, _, _, _ = pc_skeleton_cpu(U, alpha=0.05)
    G_gpu, _, _, _ = pc_skeleton_cupc(U, alpha=0.05)
    assert np.array_equal(G_cpu, G_gpu)


def test_versionb_bad_backend_raises():
    import cits
    X = _toy_series()
    with pytest.raises(ValueError):
        cits.cits_versionb(X, alpha=0.05, tau=1, backend='nonsense')
