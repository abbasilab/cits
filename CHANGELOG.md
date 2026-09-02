# Changelog

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
