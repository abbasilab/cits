"""
cits._cupc_wrapper (internal)

Python ctypes wrapper for cuPC (Zarebavani et al. 2020, TPDS).

Shared library location
------------------------
The compiled cuPC skeleton library ``Skeleton.so`` is a REQUIRED external
dependency for the GPU and Version-B modes (base CITS does not need it). It
is a GPU/CUDA build, not on PyPI, and is licensed separately under cuPC's own
GPL-3.0 license, so it is NOT bundled with this package.

``_locate_cupc_dir()`` resolves the directory containing ``Skeleton.so``,
cached once per process, using the first location that actually contains it:

  1. ``$CUPC_DIR`` environment variable.
  2. Persisted config file
     (``${XDG_CONFIG_HOME:-~/.config}/cits/cupc_dir``).
  3. Candidate defaults: ``~/repos/cupc``, ``<package_dir>/external/cupc``,
     ``./cupc``, ``./repos/cupc``.
  4. If still not found AND running interactively (``sys.stdin.isatty()``):
     prompt for the path (up to 3 attempts), validate, and persist it to the
     config file so future runs auto-find it.
  5. Otherwise raise a clear, actionable ``FileNotFoundError`` (candidates
     tried, ``export CUPC_DIR=...``, the nvcc build command, the source URL,
     and a pointer to ``cits.set_cupc_dir(path)``).

Notebook / non-interactive users can set the path programmatically with
``cits.set_cupc_dir('/path/to/cupc')`` (validates and persists it).

Discovery only runs when a GPU/Version-B function is actually called, never
at ``import cits`` time.

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

import sys

_LIB_NAME = "Skeleton.so"
_CUPC_SOURCE_URL = "https://github.com/LIS-Laboratory/cupc"

# Resolved cuPC directory, cached once per process (see _locate_cupc_dir).
_CUPC_DIR_CACHE = None


def _config_file_path():
    """Path to the persisted cuPC-directory config file.

    ${XDG_CONFIG_HOME:-~/.config}/cits/cupc_dir
    """
    base = os.environ.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config")
    return os.path.join(base, "cits", "cupc_dir")


def _has_skeleton(cupc_dir):
    """True iff `cupc_dir` is a directory containing Skeleton.so."""
    if not cupc_dir:
        return False
    return os.path.isfile(os.path.join(cupc_dir, _LIB_NAME))


def _normalize_dir(path):
    """Expand ~ and $VARS in a user-supplied path."""
    return os.path.abspath(os.path.expanduser(os.path.expandvars(path.strip())))


def _read_persisted_dir():
    """Read the cuPC directory from the config file, or None."""
    cfg = _config_file_path()
    try:
        with open(cfg, "r") as fh:
            path = fh.read().strip()
        return _normalize_dir(path) if path else None
    except (OSError, IOError):
        return None


def _persist_dir(cupc_dir):
    """Write `cupc_dir` to the config file, creating parent dirs as needed."""
    cfg = _config_file_path()
    try:
        os.makedirs(os.path.dirname(cfg), exist_ok=True)
        with open(cfg, "w") as fh:
            fh.write(cupc_dir + "\n")
    except (OSError, IOError):
        # Persistence is best-effort; failure to write the config file must
        # not break an otherwise-valid resolution.
        pass


def _candidate_dirs():
    """Ordered candidate default directories to probe for Skeleton.so."""
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    return [
        os.path.expanduser("~/repos/cupc"),
        os.path.join(pkg_dir, "external", "cupc"),
        os.path.abspath("./cupc"),
        os.path.abspath("./repos/cupc"),
    ]


def _not_found_error():
    """Build the actionable FileNotFoundError for non-interactive failure."""
    tried = []
    env_dir = os.environ.get("CUPC_DIR")
    if env_dir:
        tried.append(f"    $CUPC_DIR       -> {_normalize_dir(env_dir)}")
    persisted = _read_persisted_dir()
    if persisted:
        tried.append(f"    config file     -> {persisted}")
    for c in _candidate_dirs():
        tried.append(f"    candidate       -> {c}")
    tried_str = "\n".join(tried) if tried else "    (none)"
    return FileNotFoundError(
        f"cuPC shared library ('{_LIB_NAME}') not found.\n"
        f"cuPC is a REQUIRED external dependency for the GPU and Version-B "
        f"modes (base CITS does not need it). It is a compiled GPU/CUDA "
        f"artifact, not on PyPI, so it is not bundled with the cits package.\n"
        f"Directories tried:\n{tried_str}\n"
        f"Fix it in any of these ways:\n"
        f"  - export CUPC_DIR=/path/to/cupc\n"
        f"  - call cits.set_cupc_dir('/path/to/cupc') (persists for future runs)\n"
        f"Build the library once with nvcc:\n"
        f"    cd /path/to/cupc && nvcc -O3 --shared -Xcompiler -fPIC "
        f"-o {_LIB_NAME} cuPC-S.cu\n"
        f"Get the cuPC source from {_CUPC_SOURCE_URL}")


def _prompt_for_dir(max_attempts=3):
    """Interactively prompt for the cuPC directory, validate, and persist.

    Returns the resolved directory on success, or None if all attempts fail.
    """
    for _ in range(max_attempts):
        try:
            raw = input(
                f"Enter path to your cuPC directory (must contain "
                f"{_LIB_NAME}): ")
        except (EOFError, KeyboardInterrupt):
            return None
        if not raw.strip():
            continue
        cupc_dir = _normalize_dir(raw)
        if _has_skeleton(cupc_dir):
            _persist_dir(cupc_dir)
            return cupc_dir
        print(f"  No {_LIB_NAME} found in {cupc_dir!r}. Try again.",
              file=sys.stderr)
    return None


def _locate_cupc_dir(force=False):
    """Resolve the directory containing Skeleton.so, cached per process.

    Discovery order (first that actually contains Skeleton.so wins):
      1. $CUPC_DIR environment variable.
      2. Persisted config file (${XDG_CONFIG_HOME:-~/.config}/cits/cupc_dir).
      3. Candidate defaults: ~/repos/cupc, <package_dir>/external/cupc,
         ./cupc, ./repos/cupc.
      4. If still not found AND interactive (sys.stdin.isatty()): prompt the
         user (up to 3 attempts); persist the answer on success.
      5. Otherwise raise an actionable FileNotFoundError.
    """
    global _CUPC_DIR_CACHE
    if _CUPC_DIR_CACHE is not None and not force:
        return _CUPC_DIR_CACHE

    # 1. Environment variable
    env_dir = os.environ.get("CUPC_DIR")
    if env_dir:
        cand = _normalize_dir(env_dir)
        if _has_skeleton(cand):
            _CUPC_DIR_CACHE = cand
            return cand

    # 2. Persisted config file
    persisted = _read_persisted_dir()
    if persisted and _has_skeleton(persisted):
        _CUPC_DIR_CACHE = persisted
        return persisted

    # 3. Candidate defaults
    for cand in _candidate_dirs():
        if _has_skeleton(cand):
            _CUPC_DIR_CACHE = cand
            return cand

    # 4. Interactive prompt
    try:
        interactive = bool(sys.stdin) and sys.stdin.isatty()
    except (AttributeError, ValueError):
        interactive = False
    if interactive:
        prompted = _prompt_for_dir()
        if prompted is not None:
            _CUPC_DIR_CACHE = prompted
            return prompted

    # 5. Give up with an actionable error
    raise _not_found_error()


def set_cupc_dir(path):
    """Set and persist the cuPC directory (containing Skeleton.so).

    For notebook / non-interactive use: validates that `path` contains
    Skeleton.so, persists it to the config file
    (${XDG_CONFIG_HOME:-~/.config}/cits/cupc_dir) so future runs auto-find
    it, and updates the in-process cache.

    Parameters
    ----------
    path : str
        Directory containing the compiled cuPC Skeleton.so.

    Returns
    -------
    str
        The normalized, validated cuPC directory.

    Raises
    ------
    FileNotFoundError
        If `path` does not contain Skeleton.so.
    """
    global _CUPC_DIR_CACHE
    cupc_dir = _normalize_dir(str(path))
    if not _has_skeleton(cupc_dir):
        raise FileNotFoundError(
            f"No {_LIB_NAME} found in {cupc_dir!r}. Build it with:\n"
            f"    cd {cupc_dir} && nvcc -O3 --shared -Xcompiler -fPIC "
            f"-o {_LIB_NAME} cuPC-S.cu\n"
            f"Get the cuPC source from {_CUPC_SOURCE_URL}")
    _persist_dir(cupc_dir)
    _CUPC_DIR_CACHE = cupc_dir
    return cupc_dir


_ML = 14  # cuPC compile-time max level


def _load_lib():
    """Load Skeleton.so and configure the Skeleton symbol's argtypes."""
    cupc_dir = _locate_cupc_dir()
    lib_path = os.path.join(cupc_dir, _LIB_NAME)
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
