"""
cits.contemporaneous

Contemporaneous / "Version B" CITS.

The base CITS-lag skeleton (``cits.gpu.cits_gpu``) infers only LAGGED
(directed-in-time) causal edges. It omits contemporaneous (lag-0)
structure. Version B augments the lagged skeleton with a contemporaneous PC
step and refits one coherent set of signed structural coefficients over the
UNION of lagged and contemporaneous parents.

Pipeline (per trial), given X of shape (p, T):
  1. Lagged skeleton: ``cits.gpu.cits_gpu(X, alpha, tau)`` -> rolled binary
     adjacency B_lag. For tau=1, B_lag[i, j] != 0 means neuron i at t-1 is a
     lag-1 parent of neuron j at t. Only the parent IDENTITY is used here;
     the signed weights are re-derived by the union LSCM refit below.
  2. Contemporaneous PC skeleton on the chi-subsampled lag-0 slice via
     ``cits._pc_raw.pc_skeleton_raw`` (same data CITS sees at lag 0).
  3. Orient the contemp skeleton with v-structure orientation
     (``cits._pc_orientation.orient_v_structures``). Meek propagation is
     OFF by default (``use_meek=False``): only collider-forced edges with
     direct conditional-independence evidence are oriented; everything else
     stays undirected and is handled by local IDA in the LSCM refit. Set
     ``use_meek=True`` to also apply Meek's R1-R3.
  4. Union: ``cits._union_cpdag.build_union`` merges the lagged parents and
     the contemp CPDAG into a single per-child parent specification plus a
     skeleton mask and an edge-type matrix.
  5. Signed LSCM refit: ``cits._lscm_refit.lscm_refit_cpdag`` fits one OLS
     per child on its union parent set. Directed edges get the signed beta;
     undirected contemp edges get a local-IDA conservative signed value (or
     NaN when sign-ambiguous).

Requires a compiled cuPC ``Skeleton.so`` and a CUDA-capable GPU (both the
lagged skeleton and the contemp PC skeleton run through cuPC). See
``cits._cupc_wrapper`` and the README "GPU setup (cuPC)" section.

Output convention for the signed weighted adjacency B:
    B[parent, child] = signed OLS beta   -- identifiable directed/undirected edge
    B[i, j] = 0                          -- non-edge
    B[i, j] = NaN                        -- skeleton-only edge (sign-ambiguous
                                            IDA multiset, or no valid orientation)
Downstream causal-weight analyses treat NaN as "edge present, no causal
weight": exclude from weighted analyses, include in skeleton-only analyses.
"""

from __future__ import annotations
import numpy as np

from .gpu import cits_gpu
from ._pc_raw import pc_skeleton_raw
from ._pc_orientation import (
    orient_v_structures,
    cpdag_from_skeleton,
    parents as _parents_in_G,
)
from ._union_cpdag import build_union
from ._lscm_refit import lscm_refit_cpdag, ols_beta_for_child


def cits_versionb(X, alpha: float = 0.05, tau: int = 1,
                  use_meek: bool = False, full_output: bool = False,
                  verbose: bool = False):
    """Contemporaneous / Version-B CITS: signed weighted adjacency over the
    union of lagged (CITS) and contemporaneous (PC) causal structure.

    Parameters
    ----------
    X : np.ndarray, shape (p, T)
        Time series, p variables (neurons) by T time points. Same input
        convention as ``cits.methods.cits_full`` and ``cits.gpu.cits_gpu``.
    alpha : float
        Significance level shared by the lagged skeleton and the contemp PC
        conditional-independence tests. Default 0.05 (paper default).
    tau : int
        CITS Markovian order / lag. Default 1 (paper default). The union
        step currently supports tau=1 only.
    use_meek : bool
        If False (default), orient the contemp skeleton with v-structures
        only (no Meek propagation) -- the paper's Version-B-safe orientation.
        If True, additionally apply Meek's R1-R3.
    full_output : bool
        If False (default), return only the signed weighted adjacency B.
        If True, return a dict with keys 'weighted', 'skeleton',
        'edge_type', 'sign_ambiguous', 'lagged', 'cpdag'.
    verbose : bool

    Returns
    -------
    B : np.ndarray, shape (p, p), float   (when full_output=False)
        Signed weighted adjacency (see module docstring for the
        beta / 0 / NaN convention).
    result : dict                          (when full_output=True)
        'weighted'       : (p, p) float signed weighted adjacency (as above)
        'skeleton'       : (p, p) int8 union skeleton presence (directed+undirected)
        'edge_type'      : (p, p) int8 per-edge type
                           (0 none, 1 CITS-lagged, 2 directed PC-contemp,
                            3 undirected PC-contemp IDA-identifiable,
                            4 undirected PC-contemp sign-ambiguous)
        'sign_ambiguous' : (p, p) bool IDA sign-ambiguity mask
        'lagged'         : (p, p) int rolled lagged adjacency from cits_gpu
        'cpdag'          : (p, p) int contemp PC CPDAG
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D (p, T); got shape {X.shape}")
    p, T = X.shape

    # ---- 1. Lagged skeleton (rolled binary adjacency) via GPU cuPC ----
    cits_lagged_B = cits_gpu(X, alpha=alpha, tau=tau, verbose=verbose)

    # LSCM refit and PC skeleton use (T, p) orientation.
    X_Tp = np.ascontiguousarray(X.T)

    # ---- 2. Contemporaneous PC skeleton ----
    pc_skel, _pc_r0, sep_sets, _inactive = pc_skeleton_raw(
        X_Tp, alpha=alpha, tau=tau, verbose=verbose)

    # ---- 3. Orient contemp skeleton (v-structures; optional Meek) ----
    if use_meek:
        pc_G = cpdag_from_skeleton(pc_skel, sep_sets)
    else:
        pc_G = orient_v_structures(pc_skel, sep_sets)

    # ---- 4. Union of lagged parents + contemp CPDAG ----
    union_parents, union_skel, _edge_type = build_union(
        cits_lagged_B, pc_skel, pc_G, sign_amb_mat=None, tau=tau)

    # ---- 5. Signed LSCM refit on the union parent set ----
    union_B, union_sa = lscm_refit_cpdag(
        X_Tp, pc_G, extra_parents_per_child=union_parents, verbose=verbose)

    # Recompute edge_type now that sign-ambiguity is known.
    _, _, edge_type = build_union(
        cits_lagged_B, pc_skel, pc_G, sign_amb_mat=union_sa, tau=tau)

    # Fill CITS-lagged-ONLY pairs (no contemp edge) into union_B: the LSCM
    # refit above iterates over PC-contemp CPDAG edges only, so lagged-only
    # edges need an explicit OLS on their union parent set.
    for child, lagged_parents in union_parents.items():
        cpdag_pa = _parents_in_G(pc_G, child)
        for (p_node, lag) in lagged_parents:
            if pc_skel[p_node, child] != 0:
                # Contemp skeleton also has this pair; already handled.
                continue
            full_parents = [(p_node, lag)]
            for k in cpdag_pa:
                full_parents.append((k, 0))
            for (other_p, other_lag) in lagged_parents:
                if (other_p, other_lag) == (p_node, lag):
                    continue
                full_parents.append((other_p, other_lag))
            betas = ols_beta_for_child(X_Tp, child, full_parents)
            union_B[p_node, child] = betas.get((p_node, lag), np.nan)

    if not full_output:
        return union_B

    return {
        'weighted': union_B,
        'skeleton': union_skel.astype(np.int8),
        'edge_type': edge_type,
        'sign_ambiguous': union_sa,
        'lagged': cits_lagged_B,
        'cpdag': pc_G,
    }
