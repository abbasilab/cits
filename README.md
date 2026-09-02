# Python Package for CITS algorithm: Causal Inference from Time Series data

CITS algorithm infers causal relationships in time series data based on structural causal model and Markovian condition of arbitrary but finite order. See the [paper](https://arxiv.org/abs/2508.01920) for details.

## Installation

You can get the latest version of CITS package as follows

`pip install cits`

## Requirements

- Python >= 3.6
- R >= 4.0
- R package `kpcalg` and its dependencies. They can be installed in R or RStudio as follows:

```
> install.packages("BiocManager")
> BiocManager::install("graph")
> BiocManager::install("RBGL")
> install.packages("pcalg")
> install.packages("kpcalg")
```


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

The faithful scalable CITS-lag skeleton. It matches base CITS exactly except
that the conditioning-set search is replaced by the neighbor-restricted cuPC
skeleton on the GPU, which is sound under the faithfulness assumption CITS
already requires. Returns the rolled lagged binary adjacency. Requires cuPC
(see "GPU setup" below).

```python
import cits

B_lag = cits.cits_gpu(X, alpha=0.05, tau=1)   # (p, p) int lagged adjacency
```

### 3. Contemporaneous / Version B CITS

Base CITS infers only lagged (directed-in-time) edges. Version B adds a
contemporaneous PC step, unions the lagged and contemporaneous parents, and
refits one coherent set of signed structural coefficients per child. It runs
the lagged skeleton (`cits_gpu`), a contemporaneous PC skeleton, v-structure
orientation (Meek optional), the union, and a signed LSCM refit with local
IDA for undirected edges. Requires cuPC.

```python
import cits

B = cits.cits_versionb(X, alpha=0.05, tau=1)   # signed weighted adjacency
# B[parent, child] = signed beta; 0 = non-edge; NaN = skeleton-only (sign-ambiguous)

out = cits.cits_versionb(X, alpha=0.05, tau=1, full_output=True)
# dict: 'weighted', 'skeleton', 'edge_type', 'sign_ambiguous', 'lagged', 'cpdag'
```

`import cits` and the base CPU algorithm always work, even without a GPU. The
`cits_gpu` and `cits_versionb` functions raise a clear error only when called
on a machine where cuPC is unavailable.

## GPU setup (cuPC)

The GPU and Version-B entry points call [cuPC](https://github.com/LIS-Laboratory/cupc)
(Zarebavani et al. 2020) for the Fisher-z skeleton search. cuPC is a
**required external dependency for the GPU and Version-B modes** (base CITS
does not need it). Its compiled shared library `Skeleton.so` is a GPU/CUDA
build, is not on PyPI, and is licensed separately under cuPC's own GPL-3.0
license, so it is **not bundled** with this package and is **not** in
`install_requires`.

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

## Documentation

[Documentation is available at readthedocs.org](https://cits.readthedocs.io/en/latest/)

## Tutorial

Visit this [Google Colab](https://colab.research.google.com/drive/1TS_uVnbiW9Pb1ywBVjHdL-lnrdFkJ3wp?usp=sharing) for getting started with this package.

Alternatively, see the [Getting Started](https://cits.readthedocs.io/en/latest/gettingstarted.html) in the documentation. 

## Contributing

Your help is absolutely welcome! Please do reach out or create a future branch!

## Citation

Biswas, R., Sripada, S., Mukherjee, S. & Abbasi-Asl, R. (2025) CITS: Nonparametric Statistical Causal Modeling for High-Resolution Neural Time Series. In Review. [https://arxiv.org/abs/2508.01920](https://arxiv.org/abs/2508.01920)
