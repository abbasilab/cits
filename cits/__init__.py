"""
CITS algorithm for Causal Inference in Time Series.

Three ways to run CITS:
  1. Base (CPU)         -- cits.methods.cits_full / cits_full_weighted
  2. GPU (cuPC)         -- cits.cits_gpu
  3. Contemporaneous /  -- cits.cits_versionb
     Version B

The GPU and Version-B entry points require a compiled cuPC ``Skeleton.so``
and a CUDA-capable GPU (see the README "GPU setup (cuPC)" section). Their
imports degrade gracefully: ``import cits`` always succeeds on a CPU-only
machine, and the GPU/Version-B functions raise a clear error when the cuPC
dependency is unavailable at call time.
"""

__version__ = "1.7.0"

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

__all__ = [
    "methods",
    "simulate_timeseries",
    "cits_gpu",
    "cits_versionb",
    "set_cupc_dir",
]
