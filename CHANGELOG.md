# Changelog

## v1.7

A correctness fix for exact paper reproduction, plus a user-experience pass so
users always know which backend is running and what to do when something fails.

### Changed

- **`cits_versionb` now reproduces the paper's montage pipeline exactly by
  default.** New parameter `weight_lagged_only: bool = False`. When False
  (default), the pipeline stops after the union LSCM refit, leaving
  lagged-only edges (present in the lagged graph but not the contemporaneous
  skeleton) unweighted -- matching `_neuropixels_versionB_pooled.versionB_directed`
  and the Fig 5A magnitudes. When True, it also OLS-weights the lagged-only
  edges (the previous, fuller behavior). Note: this changes the default
  numeric output of `cits_versionb` relative to v1.5/v1.6 (e.g. on gabors,
  41 nonzeros instead of 47); the union skeleton and edge-type outputs are
  unchanged.

### Added (user experience)

- **One-time backend notice** (stderr, once per process) when a Version-B run
  starts: `cits: using cuPC GPU backend.` (auto found cuPC),
  `cits: cuPC not found; using the CPU backend ...` (auto fell back), or the
  existing >100-variable recommendation for explicit `backend='cpu'`.
  Silence with the env var `CITS_QUIET=1`.
- **Runtime GPU-failure handling.** If a cuPC/CUDA call fails at run time:
  under `backend='auto'` a warning is emitted and the run falls back to the
  CPU backend; under explicit `backend='cupc'` it re-raises with actionable
  guidance (fix cuPC/CUDA or use `backend='cpu'`) rather than silently
  falling back.
- **Actionable `tau>1` error** for `cits_versionb` pointing to `cits_gpu` for
  higher-lag lagged-only inference. The invalid-`backend` `ValueError` is
  unchanged.
- **Progress markers** to stderr when `verbose=True`
  (`[cits versionB] 1/4 lagged skeleton (p=NN)`, `2/4 contemporaneous PC`,
  `3/4 union`, `4/4 LSCM refit`). Silent by default.
- **Equivalence test** (`test/test_paper_repro.py`, runs only when cuPC is
  available) asserting `cits_versionb(X, backend='cupc', weight_lagged_only=False)`
  reproduces the original montage pipeline bit-for-bit on the gabors input.

### Docs

- README: `weight_lagged_only` documented (default reproduces the paper);
  new "What you'll see / troubleshooting" subsection covering the backend
  notice, CPU/GPU guidance, and cuPC-failure guidance.

## v1.6

Makes the package usable on any machine with no author involvement: the GPU is
now recommended (not required) for large graphs, and every mode has a
pure-Python path.

### Added

- **CPU PC-stable skeleton (`cits._pc_skeleton_cpu.pc_skeleton_cpu`).** A
  pure-numpy, neighbor-restricted, increasing-conditioning-size PC-stable
  skeleton with a Fisher-z partial-correlation CI test (reuses the v1.4-fixed
  `cits.methods.partial_corr`). It implements the SAME algorithm as cuPC
  (NOT the exponential powerset conditioning of `cits_full`), so it recovers
  the same skeleton as cuPC. Validated: on a toy graph the CPU and cuPC
  skeletons are identical. Used for both the lagged unrolled-window skeleton
  and the contemporaneous slice.
- **`backend` argument on `cits_versionb`.** `'auto'` (new default) uses cuPC
  if a working `Skeleton.so` is found, else the CPU skeleton; `'cpu'` forces
  CPU; `'cupc'` forces GPU (clear error if unavailable). On the CPU path with
  more than ~100 variables, a one-time recommendation to use the cuPC backend
  is printed. `alpha=0.05`, `tau=1` defaults unchanged; `tau=1` remains the
  only supported union order.
- **`_pc_raw.pc_skeleton_raw` `backend='python'`/`'cpu'`** now runs the new
  CPU skeleton (previously a `NotImplementedError` stub).

### Changed

- **cuPC is optional, recommended for large graphs (> ~100 variables), not
  required.** `import cits`, base CITS, and `cits_versionb(backend='cpu')`
  need nothing native. `cits_gpu` stays cuPC-only (it is the GPU accelerator).
- **R / `kpcalg` is now an optional extra.** `rpy2` moved out of core
  `install_requires` into `extras_require['hsic']` (`pip install cits[hsic]`);
  R is imported lazily inside the HSIC test only, so the Gaussian
  partial-correlation path works with just Python + numpy. (methods.py already
  imported rpy2 lazily; no code change was needed there.)

### Docs

- README: runnable Quickstart via `cits.simulate_timeseries`; "Which method
  should I use?" and "Choosing parameters" decision guides; cuPC reframed as
  recommended-for-large-graphs; optional-R note; Zenodo DOI placeholder for
  the tagged release.

## v1.5

Two additional ways to run CITS, alongside the existing base (CPU) algorithm.
The repository now offers three entry points.

### Added

- **GPU-accelerated CITS (`cits.cits_gpu`).** The faithful scalable CITS-lag
  skeleton. It matches the exact base CITS algorithm
  (`cits.methods.cits_full` with `cond_dep='cond_dep_pcorr'`) on every axis
  except the conditioning-set search, which is replaced by the
  neighbor-restricted, increasing-size PC search (cuPC on GPU). Returns the
  rolled lagged binary adjacency. Requires a compiled cuPC `Skeleton.so` and
  a CUDA-capable GPU.

- **Contemporaneous / Version-B CITS (`cits.cits_versionb`).** Augments the
  lagged CITS skeleton with a contemporaneous PC step, unions the lagged and
  contemporaneous parents, and refits one coherent set of signed structural
  (LSCM) coefficients per child. Pipeline: lagged skeleton (`cits_gpu`) +
  contemporaneous PC skeleton (chi-subsampled lag-0 slice) + v-structure
  orientation (Meek optional) + union + signed LSCM refit with local IDA for
  undirected edges. Returns the signed weighted adjacency (beta / 0 / NaN
  convention). Requires cuPC as above.

- Internal helper modules `cits._cupc_wrapper`, `cits._pc_raw`,
  `cits._pc_orientation`, `cits._union_cpdag`, `cits._lscm_refit`.

### Notes

- The cuPC shared library `Skeleton.so` is not bundled (separate GPL-3.0
  GPU/CUDA build). Its location is configurable via the `CUPC_DIR`
  environment variable, and a clear, actionable error is raised if it is not
  found. See the README "GPU setup (cuPC)" section and cite
  `zarebavani2020cupc`.
- The GPU/Version-B imports degrade gracefully: `import cits` and the base
  CPU algorithm work on a CPU-only machine; the GPU/Version-B functions raise
  a clear error only when called without cuPC available.

## v1.4

Two correctness fixes to the partial-correlation conditional independence test.

### Fixed

- **`partial_corr` now fits an intercept.** Previously the regression of A and B
  on the conditioning set was forced through the origin (`linalg.lstsq` on
  `C[idx, :].T` without an intercept column). When the data was not centered,
  this biased the residuals and the resulting partial correlation. The fix
  prepends a column of ones to the conditioning matrix before the least-squares
  solve, making the partial correlation shift-invariant.

- **`cits_unrolled` conditioning set spans the full unrolled graph and excludes
  the two variables being tested.** Previously the conditioning powerset
  iterated over `range(2*(tau+1))`, which only covers `2*(tau+1)` nodes — far
  fewer than the `2*p*(tau+1)` nodes of the unrolled graph. The two variables
  being tested (`i = t*p + v` and `j = t1*p + v1`) could also appear in the
  conditioning set, which is incorrect by the definition of conditional
  independence. The fix changes the powerset iterable to
  `(i for i in range(2*p*(tau+1)) if i != t*p+v and i != t1*p+v1)`.

### Tests

- Added a shift-invariance regression test in `test/test_cits.py` that compares
  graph recovery before and after applying per-neuron offsets. With the fix the
  adjacency matrix and weighted effects are identical to numerical precision.

## v1.3

Earlier release. See git history for details.
