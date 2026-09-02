# Python Package for CITS algorithm: Causal Inference from Time Series data

CITS algorithm infers causal relationships in time series data based on structural causal model and Markovian condition of arbitrary but finite order. See the [paper](https://arxiv.org/abs/2508.01920) for details.

## Installation

You can get the latest version of CITS package as follows

`pip install cits`

This installs everything needed for the base algorithm, the CPU skeleton,
and the GPU/Version-B wiring on Python + numpy/scipy/pandas/networkx. No R
and no GPU are required to import `cits` or to run the default (Gaussian)
conditional-independence test.

## Requirements

- Python >= 3.6 (core: numpy, scipy, pandas, networkx).
- **Optional, for the non-Gaussian HSIC CI test only:** R >= 4.0 with the
  `kpcalg` package, plus the Python `rpy2` bridge. Install the Python side
  with `pip install cits[hsic]`, and the R side in R/RStudio:

  ```
  > install.packages("BiocManager")
  > BiocManager::install("graph")
  > BiocManager::install("RBGL")
  > install.packages("pcalg")
  > install.packages("kpcalg")
  ```

  The Gaussian partial-correlation test (the default) needs none of this.
- **Optional, recommended for large graphs (more than ~100 variables):** the
  cuPC GPU backend. See "GPU setup (cuPC)" below.


## Three ways to run CITS

The package exposes three entry points. All take the time series `X` with
shape `(p, T)` (p variables/neurons by T time points).

### 1. Base CITS (CPU)

The original nonparametric algorithm. Always available, no GPU required.

```python
import numpy as np
from cits import methods

X = np.random.randn(4, 1000)          # (p, T)
adj = methods.cits_full(X, tau=1, alpha=0.05)             # binary adjacency
adj, eff = methods.cits_full_weighted(X, tau=1, alpha=0.05)  # + weighted effects
```

### 2. GPU-accelerated CITS (cuPC)

The scalable CITS-lag skeleton. It matches base CITS except that the
exponential powerset conditioning search is replaced by the
neighbor-restricted PC-stable skeleton (cuPC on GPU), which is sound under
the faithfulness assumption CITS already requires. It recovers lagged
(directed-in-time, one-lag) edges only and scales to on the order of 1000
variables. Returns the rolled lagged binary adjacency. Requires cuPC (see
"GPU setup" below).

```python
import cits

B_lag = cits.cits_gpu(X, alpha=0.05, tau=1)   # (p, p) int lagged adjacency
```

### 3. Contemporaneous / Version B CITS

Base CITS infers only lagged (directed-in-time) edges. Version B adds a
contemporaneous PC step, unions the lagged and contemporaneous parents, and
refits one coherent set of signed structural (LSCM) edge weights per child.
This is the pipeline used for the paper's neural analyses. It runs the lagged
skeleton, a contemporaneous PC skeleton, v-structure orientation (Meek
optional), the union, and a signed LSCM refit with local IDA for undirected
edges.

Version B runs on **either backend**. The `backend` argument defaults to
`'auto'`: it uses cuPC when a working `Skeleton.so` is found, otherwise it
falls back to the pure-numpy CPU skeleton (both give the same skeleton). Use
`backend='cpu'` to force CPU and `backend='cupc'` to require the GPU. cuPC is
recommended above ~100 variables for speed.

```python
import cits

B = cits.cits_versionb(X, alpha=0.05, tau=1)                 # backend='auto'
B = cits.cits_versionb(X, alpha=0.05, tau=1, backend='cpu')  # force CPU, no GPU
# B[parent, child] = signed beta; 0 = non-edge; NaN = skeleton-only (sign-ambiguous)

out = cits.cits_versionb(X, alpha=0.05, tau=1, full_output=True)
# dict: 'weighted', 'skeleton', 'edge_type', 'sign_ambiguous', 'lagged', 'cpdag'
```

`import cits`, base CITS, and `cits_versionb(..., backend='cpu')` all work with
no GPU and nothing native installed. `cits_gpu` (and `backend='cupc'`) raise a
clear error only when called on a machine where cuPC is unavailable.

**Reproducing the paper (`weight_lagged_only`).** By default
(`weight_lagged_only=False`) `cits_versionb` reproduces the paper's montage
pipeline exactly: it stops after the union LSCM refit, so lagged-only edges
(present in the lagged graph but not the contemporaneous skeleton) stay in the
skeleton but carry no weight. The Fig 5A magnitudes correspond to this
default. Set `weight_lagged_only=True` to additionally OLS-weight those
lagged-only edges (a fuller weighting with extra small-magnitude nonzeros).
Only `tau=1` is supported (the union step); for lagged-only inference at
higher `tau` use `cits_gpu(X, tau=...)`.

## Quickstart

A runnable end-to-end example using the bundled simulator. No GPU needed.

```python
import numpy as np
import cits
from cits import methods, simulate_timeseries

# Simulate a small linear-Gaussian series: X is (p=4, T), plus ground truth.
X, gt_adj, gt_weighted = simulate_timeseries.simulate('lingauss1', noise=1.0, T=2000)
print(X.shape)          # (4, 2000)  -> p=4 variables, T=2000 time points

# 1) Base CITS (CPU): binary and weighted lagged adjacency.
adj = methods.cits_full(X, tau=1, alpha=0.05)             # (p, p) 0/1
adj_w, eff = methods.cits_full_weighted(X, tau=1, alpha=0.05)

# 2) Version B (CPU here): lagged + contemporaneous, signed weights.
B = cits.cits_versionb(X, alpha=0.05, tau=1, backend='cpu')

# 3) At scale, prefer the GPU lagged skeleton (needs cuPC):
# B_lag = cits.cits_gpu(X, alpha=0.05, tau=1)

# Read the result: entry [i, j] is the i -> j causal influence.
# For cits_full: 1 = edge, 0 = no edge.
# For cits_versionb: signed beta = edge weight, 0 = no edge,
#   NaN = edge present but sign-ambiguous (no identifiable weight).
parents_of_2 = np.where(adj[:, 2] != 0)[0]
print("inferred causes of variable 2:", parents_of_2)   # expect {0, 1} for lingauss1
```

### One entry point: `cits.run`

`cits.run(X, method, **kwargs)` dispatches by name, with `method` one of
`'base'`, `'gpu'`, `'versionb'`. Keyword arguments pass straight through to the
underlying function.

```python
adj = cits.run(X, 'base', tau=1, alpha=0.05)          # -> methods.cits_full
B_lag = cits.run(X, 'gpu', tau=1)                      # -> cits_gpu
B = cits.run(X, 'versionb', backend='cpu')            # -> cits_versionb
```

### Plotting (optional)

With the `viz` extra installed (`pip install cits[viz]`), draw the graph or
its adjacency in the paper's style:

```python
import matplotlib.pyplot as plt
groups = {0: 'src', 1: 'src', 2: 'mid', 3: 'sink'}
cits.plot_graph(B, labels=['a','b','c','d'], groups=groups, title='causal graph')
cits.plot_matrix(B, labels=['a','b','c','d'], groups=groups, title='adjacency')
plt.show()
```

`plot_graph` draws directed edges as arrows, reciprocal pairs as a single
double-headed edge, edge width proportional to |weight|, and colors nodes by
`groups` (colorblind-safe palette). `plot_matrix` is a diverging heatmap for
signed weights with optional group separators. Both raise a clear error
telling you to `pip install cits[viz]` if matplotlib is missing; `import cits`
never needs matplotlib.

## Which method should I use?

- **`cits.methods.cits_full` (base CITS)** — the reference algorithm with the
  full consistency guarantees. Lagged edges only, no contemporaneous edges.
  Supports the non-Gaussian HSIC CI test. Uses exhaustive powerset
  conditioning, so it is only practical for small graphs (a handful of
  variables). Use it for small problems, or when you want the exact canonical
  result or the non-Gaussian CI test.
- **`cits.cits_gpu`** — lagged-only skeleton, cuPC-accelerated; scales to
  ~1000 variables. Use it when you have many variables, need only lagged
  (directed, one-lag) edges, and have a GPU.
- **`cits.cits_versionb`** — lagged + contemporaneous edges with signed LSCM
  edge weights (the pipeline used for the paper's neural analyses). Use it
  when (i) the sampling rate is slow relative to the interaction timescale so
  within-frame/contemporaneous effects matter (e.g. calcium imaging, or
  coarse time bins), or (ii) you want signed weights and a fuller causal
  graph. Backend is `auto` (CPU for small graphs; cuPC recommended above
  ~100 variables).

The ~100-variable figure is a practical guideline, not a hard rule; benchmark
your own setup.

## Choosing parameters

- **`tau` (Markov order / maximum lag)** — the number of time steps over which
  one variable can influence another. Use `tau=1` when the interaction delay
  is about one time bin (the paper's setting for ~10 ms neural bins). Larger
  `tau` captures longer-lag dependencies but costs statistical power. Pick
  `tau` from the known interaction timescale / autocorrelation of your data.
  Note `cits_versionb` currently supports `tau=1` only.
- **`alpha` (CI-test level)** — controls sparsity. Default `0.05`. Lower
  (`0.01`) gives a sparser, more conservative graph; higher (`0.1`) is denser
  and more sensitive. Check stability across `{0.01, 0.05, 0.1}`.
- **Time-bin / sampling** — finer bins resolve directionality better but give
  fewer counts per bin (less power) and can break Gaussianity (then use the
  HSIC test). Bins coarser than the interaction delay push effects into the
  contemporaneous slice (then use `cits_versionb`). ~10 ms bins were used for
  spike data in the paper.
- **CI test** — Gaussian partial correlation (default, fast) for approximately
  linear/Gaussian data; HSIC (needs R + `kpcalg`) for nonlinear/non-Gaussian
  data.
- **`backend` (`cits_versionb`)** — `'cpu'` for small-to-moderate graphs;
  `'cupc'` recommended above ~100 variables; `'auto'` picks cuPC when
  available.
- **Reproducible graphs** — for a robust graph, run over multiple independent
  windows and keep edges that appear in a high fraction of them. The paper
  retained edges present in at least 90% of non-overlapping 90 s windows.
  This is the recommended way to threshold down to a stable graph (the 90%
  figure is a practical guideline, not a hard rule).

## GPU setup (cuPC)

cuPC is the **recommended backend for large graphs** (more than ~100
variables, as a practical guideline; benchmark your own setup). It is
**optional**: `cits_gpu` requires it, but base CITS and `cits_versionb`
(with the CPU backend, which `backend='auto'` selects automatically when no
GPU is present) run with no GPU at all.

The GPU path calls [cuPC](https://github.com/LIS-Laboratory/cupc) (Zarebavani
et al. 2020) for the Fisher-z skeleton search. Its compiled shared library
`Skeleton.so` is a GPU/CUDA build, is not on PyPI, and is licensed separately
under cuPC's own GPL-3.0 license, so it is **not bundled** with this package
and is **not** in `install_requires`.

You need a CUDA-capable GPU and the CUDA toolkit (`nvcc`). Build `Skeleton.so`
once from the cuPC source:

```
git clone https://github.com/LIS-Laboratory/cupc
cd cupc
nvcc -O3 --shared -Xcompiler -fPIC -o Skeleton.so cuPC-S.cu
```

### How the package finds `Skeleton.so`

The first time you call `cits_gpu` or `cits_versionb`, the package resolves
the cuPC directory (once per process) using the first location that actually
contains `Skeleton.so`:

1. The `CUPC_DIR` environment variable.
2. A persisted config file, `${XDG_CONFIG_HOME:-~/.config}/cits/cupc_dir`.
3. Candidate defaults: `~/repos/cupc`, `<package_dir>/external/cupc`,
   `./cupc`, `./repos/cupc`.
4. If still not found and you are on an interactive terminal, the package
   **asks once**: `Enter path to your cuPC directory (must contain
   Skeleton.so):`. A valid answer is remembered in the config file (step 2),
   so you are not asked again on future runs.
5. If still not found and the session is non-interactive (e.g. a notebook or
   a batch job), a clear error explains what was tried and how to fix it.

Set the location explicitly in either of these ways:

```
export CUPC_DIR=/path/to/cupc          # environment variable
```

```python
import cits
cits.set_cupc_dir("/path/to/cupc")     # validates and persists to the config file
```

`set_cupc_dir` is the recommended path for notebooks and other
non-interactive environments, where the prompt is not shown. Discovery never
runs at `import cits` time, so `import cits` and base CITS always work with no
cuPC present.

**Citation for cuPC** (`zarebavani2020cupc`): Zarebavani, B., Jafarinejad, F.,
Hashemi, M. & Salehkaleybar, S. (2020). cuPC: CUDA-based Parallel PC Algorithm
for Causal Structure Learning on GPU. *IEEE Transactions on Parallel and
Distributed Systems*, 31(3), 530-542.

## What you'll see / troubleshooting

- **Backend notice.** The first Version-B run in a process emits one line
  (via the `cits` logger; stderr by default) saying which backend it chose
  and why:
  - `cits: using cuPC GPU backend.` — `backend='auto'` found a working cuPC.
  - `cits: cuPC not found; using the CPU backend (fine up to ~100 variables;
    see README 'GPU setup' to enable GPU acceleration).` — `auto` fell back
    to CPU (small graph).
  - For a CPU run above ~100 variables (explicit `backend='cpu'`, or `auto`
    with no cuPC): `cits: Version B on CPU with p=NN variables may be slow;
    the CPU skeleton search scales steeply with p. In our scaling benchmark
    the cuPC GPU backend inferred p=1000-variable graphs in ~33 s. The GPU
    backend is recommended above ~100 variables (see README).`

  It prints at most once per process. Silence such info notices with
  `export CITS_QUIET=1` (warnings still show).

- **Controlling verbosity (logging).** All messages route through the standard
  logger `logging.getLogger('cits')`. Raise its level to quiet things down or
  add your own handler to capture them:

  ```python
  import logging
  logging.getLogger('cits').setLevel(logging.WARNING)  # hide info notices
  logging.getLogger('cits').setLevel(logging.ERROR)    # hide warnings too
  ```

  By default (no logging configured) notices and warnings appear on stderr.

- **CPU vs GPU guidance.** The CPU backend works everywhere and is fine for
  small-to-moderate graphs. Above ~100 variables (a practical guideline, not
  a hard rule; the CPU skeleton search scales steeply with p) the cuPC GPU
  backend is much faster — the paper's scaling benchmark inferred
  p=1000-variable graphs in ~33 s on cuPC. `backend='auto'` picks cuPC when
  available.

- **Input validation.** `cits_full`, `cits_gpu`, and `cits_versionb` check X
  up front: X must be 2D `(p variables, T timepoints)`; NaN/inf raises a clear
  error; a transposed-looking array (p > T) warns; constant/all-zero variable
  rows warn with the offending indices (partial correlation is undefined on
  constant series).

- **cuPC call failed at runtime.** If cuPC loads but the GPU call errors
  (driver / CUDA mismatch, no visible GPU, out of memory):
  - Under `backend='auto'` you'll see a warning
    `cits: cuPC GPU call failed (<reason>); falling back to the CPU backend.
    See README 'GPU setup'.` and the run continues on CPU.
  - Under explicit `backend='cupc'` the call re-raises with
    `cits: cuPC GPU call failed (<reason>). Fix your cuPC/CUDA setup or rerun
    with backend='cpu'.` (no silent fallback).

- **Progress for long runs.** Pass `verbose=True` to print concise stage
  markers to stderr: `[cits versionB] 1/4 lagged skeleton (p=NN)`,
  `2/4 contemporaneous PC`, `3/4 union`, `4/4 LSCM refit`.

- **`tau>1` with `cits_versionb`.** Raises a clear error; the union step
  supports `tau=1` only. Use `cits_gpu(X, tau=...)` for higher-lag
  lagged-only inference.

## Documentation

[Documentation is available at readthedocs.org](https://cits.readthedocs.io/en/latest/)

## Tutorial

Visit this [Google Colab](https://colab.research.google.com/drive/1TS_uVnbiW9Pb1ywBVjHdL-lnrdFkJ3wp?usp=sharing) for getting started with this package.

Alternatively, see the [Getting Started](https://cits.readthedocs.io/en/latest/gettingstarted.html) in the documentation. 

## Contributing

Your help is absolutely welcome! Please do reach out or create a future branch!

## Citation

If you use CITS, please cite the paper:

Biswas, R., Sripada, S., Mukherjee, S. & Abbasi-Asl, R. CITS: Nonparametric Statistical Causal Modeling for High-Resolution Neural Time Series. arXiv:2508.01920. [https://arxiv.org/abs/2508.01920](https://arxiv.org/abs/2508.01920)

The citation will be finalized upon publication; please check this section (or
the arXiv page) for the up-to-date reference before citing. `cits.cite()`
prints this pointer. A citeable Zenodo DOI for the software will accompany the
tagged release: `DOI: <to be assigned on release>`.

The GPU backend uses cuPC (Zarebavani et al. 2020); see "GPU setup (cuPC)" for
its separate citation and license.
