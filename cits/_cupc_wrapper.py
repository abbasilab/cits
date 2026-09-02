"""
cits._cupc_wrapper (internal)

Python ctypes wrapper for cuPC (Zarebavani et al. 2020, TPDS).

Shared library location
------------------------
The compiled cuPC skeleton library ``Skeleton.so`` is NOT bundled with this
package (it is a GPU/CUDA build and is licensed separately under cuPC's own
GPL-3.0 license). Point this wrapper at a directory containing ``Skeleton.so``
via the ``CUPC_DIR`` environment variable::

    export CUPC_DIR=/path/to/cupc

If ``CUPC_DIR`` is unset, the wrapper falls back to ``~/repos/cupc``. If
``Skeleton.so`` is not found at the resolved location, a clear, actionable
error is raised explaining how to build it.

Build cuPC::

    cd $CUPC_DIR
    nvcc -O3 --shared -Xcompiler -fPIC -o Skeleton.so cuPC-S.cu

C entry point (from cuPC-S.h):
  extern "C" void Skeleton(double* C, int *P, int *G, double *Th,
                           int *l, int *maxlevel, double *pMax, int* SepSet);

  C       : (p, p) double, correlation matrix, row-major flat
  P       : pointer to int p (number of variables)
  G       : (p, p) int, initial adjacency (1=edge, 0=no edge), diag 0;
            updated in place to skeleton output
  Th      : double array, threshold per level. Th[l] = |z_{alpha/2}| / sqrt(N - l - 3)
            Length >= 14 (cuPC's max level cap ML=14).
  l       : pointer to int, final level reached (updated in place)
  maxlevel: pointer to int, cap on level
  pMax    : (p, p) double, max |partial r| per edge (NOT a p-value -- the
            R wrapper sets pMax[which(pMax == -100000)] <- -Inf so that's
            the "no test" sentinel)
  SepSet  : (p*p, 14) int, separating set per edge (-1 padded);
            stored as flat (p*p*14,) but layout is row-major (p*p) outer
            and 14 inner, with sepsetmat[(i*p+j), :] = sep_set values
            for the edge (i, j).

This wrapper:
  pc_skeleton_cupc(X, alpha=0.05, max_level=14)
      -> (G, sep_sets, inactive_neurons, final_level)

  X : (T, p) data matrix (samples x variables).
"""

from __future__ import annotations
import os
import ctypes
import numpy as np
from scipy.stats import norm

# cuPC shared-library location is configurable via the CUPC_DIR environment
# variable. Falls back to ~/repos/cupc when unset.
_DEFAULT_CUPC_DIR = os.path.expanduser("~/repos/cupc")


def _resolve_cupc_lib_path():
    cupc_dir = os.environ.get("CUPC_DIR", _DEFAULT_CUPC_DIR)
    return cupc_dir, os.path.join(cupc_dir, "Skeleton.so")


_ML = 14  # cuPC compile-time max level


def _load_lib():
    """Load Skeleton.so and configure the Skeleton symbol's argtypes."""
    cupc_dir, lib_path = _resolve_cupc_lib_path()
    if not os.path.exists(lib_path):
        raise FileNotFoundError(
            f"cuPC shared library not found at {lib_path!r}.\n"
            f"cuPC ('Skeleton.so') is a separate GPU/CUDA build and is not "
            f"bundled with the cits package.\n"
            f"Set the CUPC_DIR environment variable to the directory that "
            f"contains (or will contain) Skeleton.so, e.g.:\n"
            f"    export CUPC_DIR=/path/to/cupc\n"
            f"Then build it once with nvcc:\n"
            f"    cd $CUPC_DIR && nvcc -O3 --shared -Xcompiler -fPIC "
            f"-o Skeleton.so cuPC-S.cu\n"
            f"(current CUPC_DIR resolves to {cupc_dir!r}; get the cuPC source "
            f"from https://github.com/LIS-Laboratory/cupc)")
    lib = ctypes.CDLL(lib_path)
    lib.Skeleton.restype = None
    lib.Skeleton.argtypes = [
        ctypes.POINTER(ctypes.c_double),  # C
        ctypes.POINTER(ctypes.c_int),     # P
        ctypes.POINTER(ctypes.c_int),     # G
        ctypes.POINTER(ctypes.c_double),  # Th
        ctypes.POINTER(ctypes.c_int),     # l
        ctypes.POINTER(ctypes.c_int),     # maxlevel
        ctypes.POINTER(ctypes.c_double),  # pMax
        ctypes.POINTER(ctypes.c_int),     # SepSet
    ]
    return lib


_LIB = None


def _get_lib():
    global _LIB
    if _LIB is None:
        _LIB = _load_lib()
    return _LIB


def _fisher_thresholds(N, alpha, n_levels=_ML):
    """Compute the per-level Fisher-z threshold used by cuPC.

    Th[l] = |z_{alpha/2}| / sqrt(N - l - 3)
    Matches the R wrapper's threshold definition (cuPC.R line 91).
    """
    z = abs(norm.ppf(alpha / 2.0))
    th = np.zeros(n_levels, dtype=np.float64)
    for l in range(n_levels):
        df = N - l - 3
        if df > 0:
            th[l] = z / np.sqrt(df)
        else:
            th[l] = 0.0
    return th


def pc_skeleton_cupc(X, alpha: float = 0.05, max_level: int = _ML,
                     zero_var_tol: float = 1e-12, verbose: bool = False):
    """Run cuPC skeleton on data matrix X.

    Parameters
    ----------
    X : array_like, shape (T, p)
        Data matrix. Treated as N=T independent samples.
    alpha : float
        Significance level for the Fisher-z partial-correlation test.
    max_level : int
        Max conditioning-set size. cuPC's compiled limit is ML=14.
    zero_var_tol : float
        Columns with variance below this are flagged inactive; their
        edges are deterministically removed.
    verbose : bool

    Returns
    -------
    G : np.ndarray, shape (p, p), int
        Symmetric undirected skeleton. G[i, j] = 1 if edge retained.
    sep_sets : dict[(i, j) -> tuple[int]]
        Separating set used to remove each removed edge.
    inactive_neurons : np.ndarray
        Indices of inactive (zero-variance) neurons.
    final_level : int
        Final conditioning-set size reached.
    """
    lib = _get_lib()

    X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
    if X.ndim != 2:
        raise ValueError(f"X must be 2D (T, p); got {X.shape}")
    T, p = X.shape
    if T < 4:
        raise ValueError(f"T={T} too small (need T >= 4 for Fisher-z)")
    if p < 2:
        raise ValueError(f"p={p} too small")

    # Zero-variance guard
    col_var = X.var(axis=0)
    inactive_mask = col_var < zero_var_tol
    inactive_neurons = np.flatnonzero(inactive_mask)
    if inactive_neurons.size > 0:
        if verbose:
            print(f"[pc_skeleton_cupc] {inactive_neurons.size} inactive "
                  f"neurons; replacing with tiny noise to keep correlation "
                  f"matrix well-defined", flush=True)
        rng = np.random.default_rng(0)
        X = X.copy()
        for j in inactive_neurons:
            X[:, j] = rng.standard_normal(T) * 1e-10

    # Correlation matrix (row-major flat)
    C = np.corrcoef(X.T)  # (p, p), symmetric, diag 1
    C = np.ascontiguousarray(C, dtype=np.float64)

    # Initial G: full, no self-loops
    G = np.ones((p, p), dtype=np.int32)
    np.fill_diagonal(G, 0)
    G = np.ascontiguousarray(G)

    # Thresholds
    Th = _fisher_thresholds(T, alpha, n_levels=max(max_level, _ML))
    Th = np.ascontiguousarray(Th, dtype=np.float64)

    # Outputs
    p_int = np.array([p], dtype=np.int32)
    l_int = np.array([0], dtype=np.int32)
    maxlevel_int = np.array([min(max_level, _ML)], dtype=np.int32)
    pMax = np.zeros((p, p), dtype=np.float64)
    pMax = np.ascontiguousarray(pMax)
    SepSet = np.full((p * p, _ML), -1, dtype=np.int32)
    SepSet = np.ascontiguousarray(SepSet)

    # Call into cuPC
    lib.Skeleton(
        C.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        p_int.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        G.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        Th.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        l_int.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        maxlevel_int.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        pMax.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        SepSet.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
    )

    final_level = int(l_int[0])

    # Force-remove edges incident on inactive neurons (defensive: should
    # already be zero from the noise, but be explicit).
    if inactive_neurons.size > 0:
        for j in inactive_neurons:
            G[j, :] = 0
            G[:, j] = 0

    # Build sep_sets dict from the SepSet matrix.
    # SepSet layout: for edge (i, j), row index = i * p + j; columns = -1 padded sep set.
    # The R wrapper only fills sepset for REMOVED edges (per its loop).
    # We replicate that: an edge (i, j) has a non-empty SepSet row iff at
    # least one column != -1 AND G[i, j] == 0.
    sep_sets = {}
    SepSet_2d = SepSet.reshape(p * p, _ML)
    for i in range(p):
        for j in range(p):
            if i == j:
                continue
            if G[i, j] != 0:
                continue  # edge retained, no sep_set
            row_idx = i * p + j
            row = SepSet_2d[row_idx]
            sep_tuple = tuple(int(v) for v in row if v != -1)
            sep_sets[(i, j)] = sep_tuple

    return G, sep_sets, inactive_neurons, final_level
