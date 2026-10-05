# Python Package for CITS algorithm: Causal Inference from Time Series data

CITS algorithm infers causal relationships in time series data based on structural causal model and Markovian condition of arbitrary but finite order. See the [paper](https://arxiv.org/abs/2508.01920) for details.

## Installation

You can get the latest version of CITS package as follows

`pip install cits`

To install from source instead:

```bash
git clone https://github.com/abbasilab/cits
cd cits
pip install .            # add the plotting helpers with: pip install ".[viz]"
```

**Typical install time:** about a minute or less on a desktop computer. A clean
install of the package and its dependencies (numpy, scipy, pandas, networkx)
took 12–16 s on a Linux server with a fast network connection; most of the
time is spent downloading numpy and scipy. The optional GPU library (cuPC)
compiles in about 30 s; see "GPU setup (cuPC)".

This installs everything needed for the base algorithm, the CPU skeleton,
and the GPU/contemporaneous wiring on Python + numpy/scipy/pandas/networkx. No R
and no GPU are required to import `cits` or to run the default (Gaussian)
conditional-independence test.

## System requirements

**Operating system.** Tested on Linux (Ubuntu 22.04, x86-64). The CPU code is
pure Python with no platform-specific parts, but macOS and Windows have not
been tested.

**Python.** Tested on Python 3.9, 3.10, 3.11 and 3.12.

**Dependencies** (installed automatically by `pip`), with the versions tested:

| Package | Tested versions |
|---|---|
| numpy | 1.26.4, 2.2.6, 2.5.3 |
| scipy | 1.13.1, 1.15.3, 1.18.1 |
| pandas | 2.3.3, 3.0.6 |
| networkx | 3.2.1, 3.4.2, 3.7 |
| matplotlib (optional, plotting) | 3.9.4, 3.10.9, 3.11.2 |
| torch (optional, `cits_rcit`) | 2.9.1 |

**Hardware.** No special hardware is needed for the CPU path; the demo below
uses about 110 MB of memory. The optional GPU backend needs an NVIDIA GPU and
the CUDA toolkit (`nvcc`) to build cuPC. It was tested on an NVIDIA RTX 5000
Ada (32 GB) with driver 535.261.03 and CUDA 12.2.

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
  This optional path is not covered by the automated test suite.

- **Optional, recommended for large graphs (more than ~100 variables):** the
  cuPC GPU backend. See "GPU setup (cuPC)" below.


## How to run CITS

Every function below runs the same CITS algorithm. They differ in only two
choices, and you pick each one separately:

1. **The conditional-independence test, which you match to your data.**
   Partial correlation (Fisher z) is fast and suits linear or approximately
   Gaussian data. The RCIT kernel test detects nonlinear and non-Gaussian
   dependence, such as in spike counts.
2. **The search, which you match to graph size and sampling rate.** The
   exhaustive conditioning-set search suits small graphs. The
   neighbor-restricted search (cuPC on a GPU) scales to about 1000 variables.
   The contemporaneous variant also infers same-time edges, which matters when
   sampling is slow relative to the interactions (e.g. calcium imaging).

| Search | Partial correlation (linear / Gaussian) | RCIT (nonlinear / non-Gaussian) |
|---|---|---|
| Exhaustive, lagged edges (small to moderate graphs) | `cits.methods.cits_full` | `cits.cits_rcit` |
| Neighbor-restricted, lagged edges (large graphs, GPU) | `cits.cits_gpu` | not available |
| Lagged + contemporaneous edges, signed weights | `cits.cits_contemporaneous` | not available |

All functions take the time series `X` with shape `(p, T)` (p variables by
T time points) and return a `(p, p)` matrix whose entry `[i, j]` is the
inferred influence of variable `i` on variable `j`. `cits.run(X, method)`
dispatches by name: `'base'`, `'rcit'`, `'gpu'` or `'contemporaneous'`.

### Exhaustive search with partial correlation: `cits.methods.cits_full`

The reference algorithm. It needs no GPU. The exhaustive search grows quickly
with the number of variables, so use it for small graphs (about 4 to 5
variables).

```python
import numpy as np
from cits import methods

X = np.random.randn(4, 1000)                                  # (p, T)
adj = methods.cits_full(X, tau=1, alpha=0.05)                 # binary adjacency
adj, eff = methods.cits_full_weighted(X, tau=1, alpha=0.05)   # plus weighted effects
```

### Exhaustive search with RCIT: `cits.cits_rcit`

The same search with the randomized conditional-independence test (RCIT;
Strobl, Zhang & Visweswaran 2019), a fast random-Fourier-feature approximation
of the kernel (HSIC) test. Use it for nonlinear or non-Gaussian data. The paper
uses it for its nonlinear autoregressive and spiking-network benchmarks. It
runs on a CUDA GPU when one is available and on the CPU otherwise, and gives
the same graph on both.

Install the extra with `pip install cits[rcit]` (adds PyTorch). If the default
PyTorch wheel targets a newer CUDA than your driver supports, PyTorch warns and
runs on the CPU. To match your driver, install PyTorch first from
https://pytorch.org (e.g. `pip install torch --index-url https://download.pytorch.org/whl/cu121`,
or `.../whl/cpu` for CPU only). The paper used torch 2.9.1 with CUDA 12.

```python
import cits

B = cits.cits_rcit(X, alpha=0.05, tau=1)   # (p, p) int lagged adjacency, self-lags included
```

`max_cond_size=None` (the default, as in the paper's benchmarks) searches
conditioning sets of every size. A smaller cap is faster on larger graphs.
`null='perm'` selects a NumPy-only permutation reference (slow, no PyTorch
needed).

### Neighbor-restricted search on a GPU: `cits.cits_gpu`

For large graphs. It replaces the exhaustive search with cuPC's
neighbor-restricted PC-stable search, which recovers the same graph under the
faithfulness assumption CITS already makes, and scales to about 1000
variables. It uses partial correlation, infers lagged edges, and requires
cuPC (see "GPU setup (cuPC)" below).

```python
import cits

B_lag = cits.cits_gpu(X, alpha=0.05, tau=1)   # (p, p) int lagged adjacency
```

### Lagged and contemporaneous edges: `cits.cits_contemporaneous`

When sampling is slow relative to the interactions, many interactions fall
within a single sample and appear as same-time (contemporaneous) dependence.
This function adds a contemporaneous PC step to the lagged CITS search, takes
the union of lagged and contemporaneous parents, and fits one set of signed
edge weights per target (a linear structural causal model refit, with local
IDA for edges whose direction is not identified). The paper uses it for its
calcium-imaging and Neuropixels analyses. It uses partial correlation and
supports `tau=1`.

The `backend` argument is `'auto'` by default: it uses cuPC when available and
the CPU otherwise. Both give the same skeleton. cuPC is recommended above about
100 variables.

```python
import cits

B = cits.cits_contemporaneous(X, alpha=0.05, tau=1)                 # backend='auto'
B = cits.cits_contemporaneous(X, alpha=0.05, tau=1, backend='cpu')  # no GPU needed
# B[parent, child] = signed weight; 0 = no edge; NaN = edge with unidentified sign

out = cits.cits_contemporaneous(X, alpha=0.05, tau=1, full_output=True)
# dict: 'weighted', 'skeleton', 'edge_type', 'sign_ambiguous', 'lagged', 'cpdag'
```

By default (`weight_lagged_only=False`) the output matches the paper's
analyses: edges found only in the lagged search stay in the skeleton without
a weight. Set `weight_lagged_only=True` to also give those edges a
least-squares weight.

## Demo (quickstart)

A runnable end-to-end example on a small simulated dataset from the bundled
simulator (4 variables, 2,000 time points). No data download and no GPU needed.

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

# 2) Lagged + contemporaneous edges with signed weights (CPU here).
B = cits.cits_contemporaneous(X, alpha=0.05, tau=1, backend='cpu')

# 3) At scale, prefer the GPU lagged skeleton (needs cuPC):
# B_lag = cits.cits_gpu(X, alpha=0.05, tau=1)

# Read the result: entry [i, j] is the i -> j causal influence.
# For cits_full: 1 = edge, 0 = no edge.
# For cits_contemporaneous: signed beta = edge weight, 0 = no edge,
#   NaN = edge present but sign-ambiguous (no identifiable weight).
parents_of_2 = np.where(adj[:, 2] != 0)[0]
print("inferred causes of variable 2:", parents_of_2)   # expect {0, 1} for lingauss1
```

**Run it.** Save the code above as `quickstart.py` and run `python quickstart.py`.

**Expected output:**

```
(4, 2000)
inferred causes of variable 2: [0 1]
```

In the simulated system, variables 0 and 1 drive variable 2 (ground-truth edges
0 → 2, 1 → 2 and 2 → 3), so CITS recovers the true causes of variable 2.

**Expected run time:** under a minute; about 50 s on a single CPU core (measured on an Intel
Xeon Gold 6430; a typical desktop is similar), using about 110 MB of memory.
If cuPC is installed, the commented `cits_gpu` line returns the same causes,
`[0 1]`, in about 10 s, most of which is one-time CUDA start-up. For larger
graphs, the paper's benchmark (random sparse linear-Gaussian graphs, one GPU)
runs 1,000 variables in about 33 s with 500 samples and about 61 s with 1,000.

### Running CITS on your own data

1. Arrange the recording as a NumPy array `X` of shape `(p, T)`: one row per
   variable (for example, a neuron), one column per time point. Values must be
   finite; `cits` raises an error on NaN.
2. Choose `tau`, the maximum interaction delay in time bins (`tau=1` is the
   default and is what the paper uses), and the significance level `alpha`.
3. Pick the function from the table in "How to run CITS": the column by your
   data (linear/Gaussian or not) and the row by graph size and whether you
   need same-time edges.
4. Read the result: entry `[i, j]` is the inferred influence of variable `i` on
   variable `j`.

### One entry point: `cits.run`

`cits.run(X, method, **kwargs)` dispatches by name, with `method` one of
`'base'`, `'gpu'`, `'contemporaneous'`, `'rcit'`. Keyword arguments pass straight through to the
underlying function.

```python
adj = cits.run(X, 'base', tau=1, alpha=0.05)          # -> methods.cits_full
B_lag = cits.run(X, 'gpu', tau=1)                      # -> cits_gpu
B = cits.run(X, 'contemporaneous', backend='cpu')      # -> cits_contemporaneous
B = cits.run(X, 'rcit')                                # -> cits_rcit
```

### Plotting (optional)

With the `viz` extra installed (`pip install cits[viz]`), draw the graph or
its adjacency in the paper's style. Grouping is optional:

```python
import matplotlib.pyplot as plt

# Basic: just pass the adjacency.
cits.plot_graph(B)
cits.plot_matrix(B)
plt.show()
```

If you have node labels/groups (e.g. brain regions, cell types), pass `groups`
for a clustered layout, per-group colors, side-strips, and a legend:

```python
groups = {0: 'src', 1: 'src', 2: 'mid', 3: 'sink'}
cits.plot_graph(B, labels=['a','b','c','d'], groups=groups)
cits.plot_matrix(B, labels=['a','b','c','d'], groups=groups)
plt.savefig('graph.png', bbox_inches='tight')   # legend sits outside the axes
```

`plot_graph` draws directed edges as curved arcs, reciprocal pairs as a single
double-headed arc, edge width proportional to |weight|, and (with `groups`)
lays nodes out in per-group clusters colored by group. `plot_matrix` renders
the adjacency as a `matshow` heatmap (diverging `bwr`) with a robust symmetric
color scale by default (a high percentile of |weights|, so faint edges stay
visible; pass `vmin=-0.1, vmax=0.1` for a fixed scale) and group side-strips
when `groups` is given. Both raise a clear error telling you to
`pip install cits[viz]` if matplotlib is missing; `import cits` never needs
matplotlib.

Grouping is most legible for a modest number of groups (guideline ~ up to
10-12). Up to 6 groups get a shaded hue family each; beyond that, each group
gets a distinct solid color. There is no hard limit; for full control over
many groups pass `group_colors={label: color}`.

## Which method should I use?

Start from the table in "How to run CITS". Practical limits:

- **`cits.methods.cits_full`**: the exhaustive search grows quickly with the
  number of variables. p=4 with T=2000 takes about 25 s on one CPU core;
  p=5 with T=3000 took about 20 min. For more variables with partial
  correlation, use `cits_gpu` or `cits_contemporaneous`.
- **`cits.cits_rcit`**: slower per test than partial correlation; practical
  for small to moderate graphs, faster on a GPU. Lower `max_cond_size` to
  trade search depth for speed.
- **`cits.cits_gpu`**: needs an NVIDIA GPU and cuPC; scales to about 1000
  variables.
- **`cits.cits_contemporaneous`**: CPU is fine for small to moderate graphs;
  cuPC is recommended above about 100 variables.

The 100-variable figure is a practical guideline, not a hard rule; benchmark
your own setup.

## Choosing parameters

- **`tau` (Markov order / maximum lag)** — the number of time steps over which
  one variable can influence another. Use `tau=1` when the interaction delay
  is about one time bin (the paper's setting for ~10 ms neural bins). Larger
  `tau` captures longer-lag dependencies but costs statistical power. Pick
  `tau` from the known interaction timescale / autocorrelation of your data.
  Note `cits_contemporaneous` currently supports `tau=1` only.
- **`alpha` (CI-test level)** — controls sparsity. Default `0.05`. Lower
  (`0.01`) gives a sparser, more conservative graph; higher (`0.1`) is denser
  and more sensitive. Check stability across `{0.01, 0.05, 0.1}`.
- **Time-bin / sampling** — finer bins resolve directionality better but give
  fewer counts per bin (less power) and can break Gaussianity (then use
  `cits_rcit`). Bins coarser than the interaction delay push effects into the
  contemporaneous slice (then use `cits_contemporaneous`). ~10 ms bins were used for
  spike data in the paper.
- **CI test** — partial correlation (fast) for approximately linear/Gaussian
  data; RCIT (`cits_rcit`) for nonlinear/non-Gaussian data. `cits_full` also
  accepts `cond_dep='cond_dep_hsic'`, an exact kernel test that needs R and
  `kpcalg`; RCIT is its faster approximation.
- **`backend` (`cits_contemporaneous`)** — `'cpu'` for small-to-moderate graphs;
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
**optional**: `cits_gpu` requires it, but base CITS and `cits_contemporaneous`
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

### Telling the package where `Skeleton.so` is

Point the package to the directory that contains `Skeleton.so`, in either way:

```
export CUPC_DIR=/path/to/cupc          # environment variable
```

```python
import cits
cits.set_cupc_dir("/path/to/cupc")     # checks the path and remembers it
```

`set_cupc_dir` remembers the location across sessions, so you only do this
once. If neither is set, the package checks a few default locations such as
`./cupc` and, in an interactive
terminal, asks once for the path. `import cits` never looks for cuPC, so
everything except the GPU backend works without it.

**Citation for cuPC** (`zarebavani2020cupc`): Zarebavani, B., Jafarinejad, F.,
Hashemi, M. & Salehkaleybar, S. (2020). cuPC: CUDA-based Parallel PC Algorithm
for Causal Structure Learning on GPU. *IEEE Transactions on Parallel and
Distributed Systems*, 31(3), 530-542.

## What you'll see / troubleshooting

- **Backend notice.** The first `cits_contemporaneous` run in a process emits one line
  (via the `cits` logger; stderr by default) saying which backend it chose
  and why:
  - `cits: using cuPC GPU backend.` — `backend='auto'` found a working cuPC.
  - `cits: cuPC not found; using the CPU backend (fine up to ~100 variables;
    see README 'GPU setup' to enable GPU acceleration).` — `auto` fell back
    to CPU (small graph).
  - For a CPU run above ~100 variables (explicit `backend='cpu'`, or `auto`
    with no cuPC): `cits: contemporaneous CITS on CPU with p=NN variables may be slow;
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

- **Input validation.** `cits_full`, `cits_gpu`, and `cits_contemporaneous` check X
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
  markers to stderr: `[cits contemporaneous] 1/4 lagged skeleton (p=NN)`,
  `2/4 contemporaneous PC`, `3/4 union`, `4/4 LSCM refit`.

- **`tau>1` with `cits_contemporaneous`.** Raises a clear error; the union step
  supports `tau=1` only. Use `cits_gpu(X, tau=...)` for higher-lag
  lagged-only inference.

## Running the tests

```bash
pip install pytest
python -m pytest test
```

Tests that need cuPC or a GPU are skipped automatically. `test_paper_repro.py` also
needs the authors' local benchmark data and is skipped elsewhere.

## Documentation

[Documentation is available at readthedocs.org](https://cits.readthedocs.io/en/latest/)

## Tutorial

Visit this [Google Colab](https://colab.research.google.com/drive/1TS_uVnbiW9Pb1ywBVjHdL-lnrdFkJ3wp?usp=sharing) for getting started with this package.

Alternatively, see the [Getting Started](https://cits.readthedocs.io/en/latest/gettingstarted.html) in the documentation. 

## Contributing

Your help is absolutely welcome! Please do reach out or create a future branch!

## Reproducing the paper

The scripts behind every figure and table of the CITS paper are in a separate
repository: <!-- TODO: add the analysis repository URL when it is published -->
[cits-paper](https://github.com/abbasilab/cits-paper).

## License

CITS is free for **noncommercial use** (academic research, teaching, nonprofit
and government work, personal use) under the
[PolyForm Noncommercial License 1.0.0](LICENSE). For commercial use, please
contact the authors.

Copyright (c) 2023-2026 Rahul Biswas.

The optional GPU backend calls cuPC, which is distributed separately under its
own license (GPL-3.0) and is not bundled with this package.

## Citation

If you use CITS in published work, please cite the paper:

Biswas, R., Sripada, S., Mukherjee, S. & Abbasi-Asl, R. CITS: Nonparametric
Statistical Causal Modeling for High-Resolution Neural Time Series.
arXiv:2508.01920. [https://arxiv.org/abs/2508.01920](https://arxiv.org/abs/2508.01920)

```bibtex
@article{biswas2025cits,
  title   = {CITS: Nonparametric Statistical Causal Modeling for High-Resolution Neural Time Series},
  author  = {Biswas, Rahul and Sripada, SuryaNarayana and Mukherjee, Somabha and Abbasi-Asl, Reza},
  journal = {arXiv preprint arXiv:2508.01920},
  year    = {2025},
  url     = {https://arxiv.org/abs/2508.01920}
}
```

`cits.cite()` prints this reference and `cits.cite(bibtex=True)` prints the
BibTeX entry. GitHub's "Cite this repository" button uses `CITATION.cff`. If
you use the GPU backend, please also cite cuPC (see "GPU setup (cuPC)").

The citation will be finalized upon publication; please check this section (or
the arXiv page) for the up-to-date reference before citing. `cits.cite()`
prints this pointer. A citeable Zenodo DOI for the software will accompany the
tagged release: `DOI: <to be assigned on release>`.
