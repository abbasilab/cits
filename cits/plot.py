"""
cits.plot

Publication-style plotting helpers for CITS causal graphs, in the style of the
paper's figures.

  plot_graph(A, ...)   -- directed causal graph (node-link) from a (p, p)
                          adjacency (signed weights or binary).
  plot_matrix(A, ...)  -- adjacency heatmap (montage style), diverging colormap
                          for signed weights, optional group separators.

matplotlib is an OPTIONAL dependency (extras_require['viz']). It (and
networkx) are imported lazily inside the functions, so ``import cits`` and the
base algorithm never require them. Calling a plot function without matplotlib
raises a clear ImportError telling you to ``pip install cits[viz]``.

Adjacency conventions (match cits_versionb / cits_gpu):
  A[i, j] != 0  -> directed edge i -> j (row = source/parent, col = target).
  A[i, j] and A[j, i] both != 0 -> reciprocal pair, drawn as one
      double-headed edge.
  NaN entries are treated as present-but-unweighted edges (skeleton-only).
"""

from __future__ import annotations
import numpy as np

# Okabe-Ito colorblind-safe qualitative palette.
_OKABE_ITO = [
    "#0072B2", "#E69F00", "#009E73", "#CC79A7",
    "#56B4E9", "#D55E00", "#F0E442", "#000000",
]


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt  # noqa: F401
        return plt
    except Exception as e:  # pragma: no cover - env dependent
        raise ImportError(
            "plotting requires matplotlib; pip install cits[viz]") from e


def _require_networkx():
    try:
        import networkx as nx  # noqa: F401
        return nx
    except Exception as e:  # pragma: no cover - env dependent
        raise ImportError(
            "plotting requires networkx (a core dependency); "
            "pip install networkx") from e


def _group_colors(groups, p):
    """Map node index -> color from `groups` (dict or length-p array), plus a
    {group_label: color} legend map. Returns (node_colors, legend)."""
    if groups is None:
        return ["#8c8c8c"] * p, {}
    if isinstance(groups, dict):
        gvals = [groups.get(i) for i in range(p)]
    else:
        gvals = list(np.asarray(groups).ravel())
        if len(gvals) != p:
            raise ValueError(
                f"groups length {len(gvals)} != number of nodes {p}")
    # Stable ordering of unique groups by first appearance.
    seen = []
    for g in gvals:
        if g is not None and g not in seen:
            seen.append(g)
    legend = {g: _OKABE_ITO[k % len(_OKABE_ITO)] for k, g in enumerate(seen)}
    node_colors = [legend.get(g, "#8c8c8c") for g in gvals]
    return node_colors, legend


def _edge_lists(A):
    """Split edges into directed and reciprocal (bidirectional) lists.

    Returns (directed, bidir) where each is a list of (u, v, mag, signed):
      directed: u -> v single arrow.
      bidir:    u <-> v double-headed (u < v).
    `mag` is |weight| (0.0 when NaN/unknown); `signed` is the signed weight
    used for sign coloring (nan -> 0.0)."""
    p = A.shape[0]

    def present(x):
        return bool(x != 0)  # NaN != 0 is True -> treated as present

    def mag(x):
        return 0.0 if (x != x) else abs(float(x))  # x!=x detects NaN

    def signed(x):
        return 0.0 if (x != x) else float(x)

    directed, bidir = [], []
    for i in range(p):
        for j in range(i + 1, p):
            aij, aji = A[i, j], A[j, i]
            eij, eji = present(aij), present(aji)
            if eij and eji:
                m = max(mag(aij), mag(aji))
                s = signed(aij) if mag(aij) >= mag(aji) else signed(aji)
                bidir.append((i, j, m, s))
            elif eij:
                directed.append((i, j, mag(aij), signed(aij)))
            elif eji:
                directed.append((j, i, mag(aji), signed(aji)))
    return directed, bidir


def _widths(mags, weight_widths, base=1.6, lo=0.8, hi=4.0):
    """Edge widths. Scaled by |weight| when weight_widths and magnitudes vary;
    otherwise uniform."""
    mags = np.asarray(mags, dtype=float)
    if mags.size == 0:
        return []
    mx = mags.max()
    if (not weight_widths) or mx <= 0 or np.allclose(mags, mags[0]):
        return [base] * len(mags)
    return list(lo + (hi - lo) * (mags / mx))


def _edge_colors(signs):
    """Sign-based edge colors: positive blue, negative red, unknown gray."""
    out = []
    for s in signs:
        if s > 0:
            out.append("#2166ac")
        elif s < 0:
            out.append("#b2182b")
        else:
            out.append("#8c8c8c")
    return out


def plot_graph(A, labels=None, groups=None, ax=None, layout="circular",
               weight_widths=True, title=None):
    """Render a directed causal graph from a (p, p) adjacency matrix.

    Parameters
    ----------
    A : array_like, shape (p, p)
        Adjacency. Signed weights or binary. A[i, j] != 0 is a directed edge
        i -> j; A[i, j] and A[j, i] both nonzero is drawn as one
        double-headed edge. NaN entries are treated as present but unweighted.
    labels : sequence of str, optional
        Node labels (length p). Defaults to node indices.
    groups : dict or array, optional
        Node -> group mapping (dict keyed by node index, or a length-p array).
        Nodes are colored by group with a colorblind-safe palette; a legend is
        added.
    ax : matplotlib Axes, optional
        Axes to draw into. Created if omitted.
    layout : str
        'circular' (default), 'spring', or 'kamada'.
    weight_widths : bool
        If True and weights vary, scale edge width by |weight|.
    title : str, optional

    Returns
    -------
    ax : matplotlib Axes
    """
    plt = _require_matplotlib()
    nx = _require_networkx()

    A = np.asarray(A)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"A must be square (p, p); got {A.shape}")
    p = A.shape[0]

    if ax is None:
        _fig, ax = plt.subplots(figsize=(6, 6))

    G = nx.DiGraph()
    G.add_nodes_from(range(p))

    if layout == "circular":
        pos = nx.circular_layout(G)
    elif layout == "spring":
        pos = nx.spring_layout(G, seed=0)
    elif layout == "kamada":
        pos = nx.kamada_kawai_layout(G) if G.number_of_edges() else \
            nx.circular_layout(G)
    else:
        raise ValueError(
            f"unknown layout {layout!r}; expected 'circular', 'spring', "
            f"or 'kamada'")

    node_colors, legend = _group_colors(groups, p)
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=520,
                           edgecolors="#333333", linewidths=1.0, ax=ax)

    lbls = {i: (str(labels[i]) if labels is not None else str(i))
            for i in range(p)}
    nx.draw_networkx_labels(G, pos, labels=lbls, font_size=9, ax=ax)

    directed, bidir = _edge_lists(A)
    if directed:
        eds = [(u, v) for (u, v, _m, _s) in directed]
        w = _widths([m for (_u, _v, m, _s) in directed], weight_widths)
        c = _edge_colors([s for (_u, _v, _m, s) in directed])
        nx.draw_networkx_edges(
            G, pos, edgelist=eds, width=w, edge_color=c, arrows=True,
            arrowstyle="-|>", arrowsize=16, node_size=520,
            connectionstyle="arc3,rad=0.06", ax=ax)
    if bidir:
        eds = [(u, v) for (u, v, _m, _s) in bidir]
        w = _widths([m for (_u, _v, m, _s) in bidir], weight_widths)
        c = _edge_colors([s for (_u, _v, _m, s) in bidir])
        nx.draw_networkx_edges(
            G, pos, edgelist=eds, width=w, edge_color=c, arrows=True,
            arrowstyle="<|-|>", arrowsize=16, node_size=520, ax=ax)

    if legend:
        handles = [plt.Line2D([0], [0], marker="o", linestyle="",
                              markerfacecolor=col, markeredgecolor="#333333",
                              markersize=9, label=str(g))
                   for g, col in legend.items()]
        ax.legend(handles=handles, loc="upper right", frameon=False,
                  fontsize=8)

    if title:
        ax.set_title(title)
    ax.set_axis_off()
    ax.margins(0.12)
    return ax


def plot_matrix(A, labels=None, groups=None, ax=None, title=None):
    """Adjacency heatmap in the montage style.

    Diverging colormap centered at 0 for signed weights; sequential for
    all-nonnegative input. Optional group separator lines when `groups` is
    given (assumes nodes are ordered by group). NaN entries are shown as a
    neutral 'no weight' color.

    Parameters
    ----------
    A : array_like, shape (p, p)
    labels : sequence of str, optional
    groups : dict or array, optional
        Node -> group mapping; separator lines are drawn where the group
        changes between consecutive node indices.
    ax : matplotlib Axes, optional
    title : str, optional

    Returns
    -------
    ax : matplotlib Axes
    """
    plt = _require_matplotlib()

    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"A must be square (p, p); got {A.shape}")
    p = A.shape[0]

    if ax is None:
        _fig, ax = plt.subplots(figsize=(6, 5))

    finite = A[np.isfinite(A)]
    has_neg = finite.size > 0 and (finite < 0).any()
    if has_neg:
        vmax = float(np.abs(finite).max()) or 1.0
        vmin, cmap = -vmax, plt.get_cmap("RdBu_r").copy()
    else:
        vmax = float(finite.max()) if finite.size else 1.0
        vmax = vmax or 1.0
        vmin, cmap = 0.0, plt.get_cmap("Reds").copy()
    cmap.set_bad(color="#f2f2f2")  # NaN cells

    masked = np.ma.masked_invalid(A)
    im = ax.imshow(masked, cmap=cmap, vmin=vmin, vmax=vmax,
                   interpolation="nearest", aspect="equal")
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=8)

    if labels is not None:
        ax.set_xticks(range(p))
        ax.set_yticks(range(p))
        ax.set_xticklabels([str(x) for x in labels], rotation=90, fontsize=7)
        ax.set_yticklabels([str(x) for x in labels], fontsize=7)
    else:
        ax.set_xticks([])
        ax.set_yticks([])
    ax.set_xlabel("target (child)")
    ax.set_ylabel("source (parent)")

    if groups is not None:
        if isinstance(groups, dict):
            gvals = [groups.get(i) for i in range(p)]
        else:
            gvals = list(np.asarray(groups).ravel())
        for k in range(p - 1):
            if gvals[k] != gvals[k + 1]:
                ax.axhline(k + 0.5, color="#333333", linewidth=0.8)
                ax.axvline(k + 0.5, color="#333333", linewidth=0.8)

    if title:
        ax.set_title(title)
    return ax
