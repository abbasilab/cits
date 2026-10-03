"""
CITS algorithm for Causal Inference in Time Series.

Three ways to run CITS:
  1. Base (CPU)         -- cits.methods.cits_full / cits_full_weighted
  2. GPU (cuPC)         -- cits.cits_gpu
  3. Contemporaneous /  -- cits.cits_versionb
     Version B

Or dispatch by name with ``cits.run(X, method)`` where method is one of
'base', 'gpu', 'versionb'.

The GPU and Version-B entry points require a compiled cuPC ``Skeleton.so``
and a CUDA-capable GPU (see the README "GPU setup (cuPC)" section). Their
imports degrade gracefully: ``import cits`` always succeeds on a CPU-only
machine, and the GPU/Version-B functions raise a clear error when the cuPC
dependency is unavailable at call time.

Extras:
  - ``cits.plot_graph`` / ``cits.plot_matrix`` -- publication-style figures
    (needs matplotlib: ``pip install cits[viz]``).
  - ``cits.run`` -- dispatch by method name.
  - ``cits.cite`` -- print how to cite the paper.

Logging: messages go through ``logging.getLogger('cits')``. Set its level to
control verbosity; ``CITS_QUIET=1`` silences informational notices.
"""

__version__ = "1.9.0"

from . import methods, simulate_timeseries

# cuPC directory helper. Pure-Python (no GPU needed to import); lets
# notebook / non-interactive users set and persist the cuPC location
# programmatically, bypassing the interactive prompt.
try:
    from ._cupc_wrapper import set_cupc_dir
except Exception as _set_cupc_err:  # pragma: no cover - env dependent
    _set_cupc_dir_err = _set_cupc_err

    def set_cupc_dir(path):
        raise ImportError(
            "cits.set_cupc_dir is unavailable because cits._cupc_wrapper "
            f"failed to import. Original import error: {_set_cupc_dir_err!r}"
        ) from _set_cupc_dir_err

# GPU and Version-B entry points. Import failures (e.g. missing optional
# dependencies) must not break `import cits` or the base CPU algorithm, so
# each is wrapped: if the import fails, the public name is replaced by a
# shim that raises a clear ImportError only when the function is called.
try:
    from .gpu import cits_gpu
except Exception as _gpu_import_error:  # pragma: no cover - env dependent
    _cits_gpu_err = _gpu_import_error

    def cits_gpu(*args, **kwargs):
        raise ImportError(
            "cits.cits_gpu is unavailable because its dependencies failed to "
            "import. It requires a compiled cuPC 'Skeleton.so' and a "
            "CUDA-capable GPU; set the CUPC_DIR environment variable to the "
            "directory containing Skeleton.so (see the README 'GPU setup "
            f"(cuPC)' section). Original import error: {_cits_gpu_err!r}"
        ) from _cits_gpu_err

try:
    from .contemporaneous import cits_versionb
except Exception as _vb_import_error:  # pragma: no cover - env dependent
    _cits_vb_err = _vb_import_error

    def cits_versionb(*args, **kwargs):
        raise ImportError(
            "cits.cits_versionb is unavailable because its dependencies "
            "failed to import. The contemporaneous / Version-B pipeline "
            "requires a compiled cuPC 'Skeleton.so' and a CUDA-capable GPU; "
            "set the CUPC_DIR environment variable to the directory "
            "containing Skeleton.so (see the README 'GPU setup (cuPC)' "
            f"section). Original import error: {_cits_vb_err!r}"
        ) from _cits_vb_err

# RCIT (kernel CI test) variant. Pure Python at import; torch is imported only
# when cits_rcit is called with null='gamma' (pip install cits[rcit]).
from .rcit import cits_rcit

# Plotting helpers. matplotlib is an optional dependency (extras 'viz');
# plot.py imports it lazily, so this import itself never needs matplotlib.
# Wrapped for safety so `import cits` never fails on a plotting-dep issue.
try:
    from .plot import plot_graph, plot_matrix
except Exception as _plot_import_error:  # pragma: no cover - env dependent
    _cits_plot_err = _plot_import_error

    def plot_graph(*args, **kwargs):
        raise ImportError(
            "cits.plot_graph is unavailable; plotting requires matplotlib "
            f"(pip install cits[viz]). Original import error: {_cits_plot_err!r}"
        ) from _cits_plot_err

    def plot_matrix(*args, **kwargs):
        raise ImportError(
            "cits.plot_matrix is unavailable; plotting requires matplotlib "
            f"(pip install cits[viz]). Original import error: {_cits_plot_err!r}"
        ) from _cits_plot_err


_METHODS = ("base", "gpu", "versionb", "rcit")


def run(X, method, **kwargs):
    """Unified entry point dispatching to the three CITS variants.

    Parameters
    ----------
    X : array_like, shape (p, T)
        Time series, p variables by T timepoints.
    method : str
        One of 'base' (-> cits.methods.cits_full), 'gpu' (-> cits.cits_gpu),
        'versionb' (-> cits.cits_versionb), or 'rcit' (-> cits.cits_rcit).
    **kwargs
        Passed through to the dispatched function (e.g. tau, alpha, backend).

    Returns
    -------
    The return value of the dispatched function.
    """
    if method == "base":
        return methods.cits_full(X, **kwargs)
    if method == "gpu":
        return cits_gpu(X, **kwargs)
    if method == "versionb":
        return cits_versionb(X, **kwargs)
    if method == "rcit":
        return cits_rcit(X, **kwargs)
    raise ValueError(
        f"unknown method {method!r}; valid methods are {list(_METHODS)} "
        f"('base' -> cits_full, 'gpu' -> cits_gpu, "
        f"'versionb' -> cits_versionb, 'rcit' -> cits_rcit).")


_CITE = (
    "Please cite the CITS paper:\n"
    "  Biswas, R., Sripada, S., Mukherjee, S. & Abbasi-Asl, R. CITS: "
    "Nonparametric Statistical Causal Modeling for High-Resolution Neural "
    "Time Series. arXiv:2508.01920. https://arxiv.org/abs/2508.01920\n"
    "The citation will be finalized upon publication; see the README "
    "'Citation' section for the current reference.")


_BIBTEX = """@article{biswas2025cits,
  title   = {CITS: Nonparametric Statistical Causal Modeling for High-Resolution Neural Time Series},
  author  = {Biswas, Rahul and Sripada, SuryaNarayana and Mukherjee, Somabha and Abbasi-Asl, Reza},
  journal = {arXiv preprint arXiv:2508.01920},
  year    = {2025},
  url     = {https://arxiv.org/abs/2508.01920}
}"""


def cite(bibtex=False):
    """Print how to cite CITS (arXiv pointer; finalized on publication).

    With ``bibtex=True``, print a BibTeX entry instead.
    """
    print(_BIBTEX if bibtex else _CITE)


__all__ = [
    "methods",
    "simulate_timeseries",
    "cits_gpu",
    "cits_versionb",
    "cits_rcit",
    "run",
    "set_cupc_dir",
    "plot_graph",
    "plot_matrix",
    "cite",
]
