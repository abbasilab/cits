"""
cits.plot

Publication-style plotting helpers for CITS causal graphs, in the style of the
paper's figures.

  plot_graph(A, ...)   -- directed causal graph (node-link) from a (p, p)
                          adjacency (signed weights or binary). With `groups`,
                          nodes are laid out in per-group clusters around a
                          ring, colored one hue-family per group, and edges are
                          drawn as curved arcs with arrowheads placed partway
                          along the arc (bidirectional pairs get one head near
                          each end).
  plot_matrix(A, ...)  -- adjacency heatmap, diverging colormap for signed
                          weights, optional subtle group separators.

matplotlib is an OPTIONAL dependency (extras_require['viz']). It (and
networkx) are imported lazily inside the functions, so ``import cits`` and the
base algorithm never require them. Calling a plot function without matplotlib
raises a clear ImportError telling you to ``pip install cits[viz]``.

Adjacency conventions (match cits_versionb / cits_gpu):
  A[i, j] != 0  -> directed edge i -> j (row = source/parent, col = target).
  A[i, j] and A[j, i] both != 0 -> reciprocal pair, drawn as one double-headed
      arc (one arrowhead near each end).
  A[i, i] != 0  -> self-loop.
  NaN entries are treated as present-but-unweighted edges (skeleton-only).

The grouped layout and arc rendering port the aesthetic of the paper figure
(regen_cfc_stimtypes.py: get_clustered_circle_positions_grouped, _arc), kept
general for arbitrary group labels.
"""

from __future__ import annotations
import numpy as np

# Okabe-Ito colorblind-safe qualitative palette (fallback when no groups).
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


# --------------------------------------------------------------------------
#  Grouping helpers
# --------------------------------------------------------------------------

def _group_values(groups, p):
    """Normalize `groups` to a length-p list of group labels (or None)."""
    if groups is None:
        return [None] * p
    if isinstance(groups, dict):
        return [groups.get(i) for i in range(p)]
    g = list(np.asarray(groups, dtype=object).ravel())
    if len(g) != p:
        raise ValueError(f"groups length {len(g)} != number of nodes {p}")
    return g


def _ordered_groups(gvals):
    """Unique group labels in first-appearance order, with member index lists."""
    order = []
    members = {}
    for i, g in enumerate(gvals):
        if g not in members:
            members[g] = []
            order.append(g)
        members[g].append(i)
    return order, members


def _base_colormaps(plt):
    """Colorblind-safe base colormaps, one hue family per group."""
    from matplotlib.colors import LinearSegmentedColormap
    teal = LinearSegmentedColormap.from_list(
        "cits_teal", ["#D6EFEA", "#5CB8A8", "#0E6B5B"])
    return [
        plt.get_cmap("Oranges"),
        teal,
        plt.get_cmap("Purples"),
        plt.get_cmap("Greens"),
        plt.get_cmap("Blues"),
        plt.get_cmap("Greys"),
    ]


def _grouped_cluster_positions(gvals, min_spacing, base_cluster_radius):
    """Cluster nodes by group around a large ring; members sit in a small
    sub-cluster around each group's centroid. Ports
    get_clustered_circle_positions_grouped, generalized to arbitrary labels."""
    order, members = _ordered_groups(gvals)
    num_groups = len(order)
    sizes = [len(members[g]) for g in order]
    max_cluster_radius = base_cluster_radius * (max(sizes) ** 0.5)
    circle_radius = min_spacing * num_groups + max_cluster_radius * 2.0
    if num_groups == 1:
        circle_radius = 0.0  # single group sits at the origin
    theta_groups = np.linspace(0, 2 * np.pi, num_groups, endpoint=False)

    pos = {}
    for g, theta in zip(order, theta_groups):
        idxs = members[g]
        n = len(idxs)
        cx = np.cos(theta) * circle_radius
        cy = np.sin(theta) * circle_radius
        if n == 1:
            pos[idxs[0]] = (cx, cy)
            continue
        cluster_radius = base_cluster_radius * (n ** 0.5)
        theta_nodes = np.linspace(0, 2 * np.pi, n, endpoint=False)
        for j, idx in enumerate(idxs):
            dx = np.cos(theta_nodes[j]) * cluster_radius
            dy = np.sin(theta_nodes[j]) * cluster_radius
            pos[idx] = (cx + dx, cy + dy)
    return pos


def _node_colors(gvals, group_colors, plt):
    """Per-node colors (one hue family per group, shaded 0.3->0.9 within group)
    and a {group: representative_color} legend map.

    group_colors : optional {group_label: color} override -> solid color for
    all members of that group."""
    order, members = _ordered_groups(gvals)
    cmaps = _base_colormaps(plt)
    colors = [None] * len(gvals)
    legend = {}
    for k, g in enumerate(order):
        idxs = members[g]
        n = len(idxs)
        if g is None and len(order) == 1:
            # No grouping at all: single neutral hue.
            for idx in idxs:
                colors[idx] = _OKABE_ITO[0]
            legend = {}
            continue
        if group_colors and g in group_colors:
            col = group_colors[g]
            for idx in idxs:
                colors[idx] = col
            legend[g] = col
            continue
        cmap = cmaps[k % len(cmaps)]
        shades = [0.6] if n == 1 else list(np.linspace(0.3, 0.9, n))
        for idx, s in zip(idxs, shades):
            colors[idx] = cmap(float(s))
        legend[g] = cmap(0.6)
    return colors, legend


# --------------------------------------------------------------------------
#  Edge helpers
# --------------------------------------------------------------------------

def _present(x):
    return bool(x != 0)  # NaN != 0 is True -> treated as present


def _mag(x):
    return 0.0 if (x != x) else abs(float(x))  # x != x detects NaN


def _sign_edge_color(signed, sign_color, default):
    if not sign_color:
        return default
    if signed > 0:
        return "#2166ac"
    if signed < 0:
        return "#b2182b"
    return default


# --------------------------------------------------------------------------
#  plot_graph
# --------------------------------------------------------------------------

def plot_graph(A, labels=None, groups=None, ax=None, layout="auto",
               group_colors=None, weight_widths=True, sign_color=False,
               edge_color="#333333", edge_width_range=(1.0, 5.0),
               node_size=250, min_spacing=6.0, base_cluster_radius=3.0,
               title=None):
    """Render a directed causal graph from a (p, p) adjacency matrix.

    Parameters
    ----------
    A : array_like, shape (p, p)
        Adjacency. Signed weights or binary. A[i, j] != 0 is a directed edge
        i -> j; A[i, j] and A[j, i] both nonzero is drawn as one double-headed
        arc; A[i, i] != 0 is a self-loop. NaN entries are treated as present
        but unweighted.
    labels : sequence of str, optional
        Node labels (length p). Defaults to node indices.
    groups : dict or array, optional
        Node -> group mapping (dict keyed by node index, or a length-p array).
        Drives the clustered layout and per-group node coloring, and adds a
        legend outside the axes.
    ax : matplotlib Axes, optional
        Axes to draw into. Created if omitted.
    layout : str
        'auto' (default) uses the grouped-cluster layout when `groups` is
        given, else 'circular'. May be forced to 'grouped', 'circular',
        'spring', or 'kamada'.
    group_colors : dict, optional
        {group_label: color} override. A group present here is drawn in that
        solid color instead of its shaded hue family.
    weight_widths : bool
        If True and weights vary, scale edge width by |weight| across
        `edge_width_range`; otherwise a single mid width.
    sign_color : bool
        If True, color edges by weight sign (blue positive, red negative).
        Default False (all edges `edge_color`).
    edge_color : str
        Default edge color (used when sign_color is False).
    edge_width_range : (float, float)
        (min, max) line widths for the weight interpolation.
    node_size : int
        Node marker size.
    min_spacing, base_cluster_radius : float
        Grouped-layout spacing controls (see the ported reference logic).
    title : str, optional

    Returns
    -------
    ax : matplotlib Axes

    Notes
    -----
    The legend is placed OUTSIDE the axes; save with ``bbox_inches='tight'``
    (e.g. ``fig.savefig(path, bbox_inches='tight')``) so it is not clipped.
    """
    plt = _require_matplotlib()
    nx = _require_networkx()
    from matplotlib.path import Path as _MP
    from matplotlib.patches import PathPatch as _PP, FancyArrowPatch as _FAP, \
        Circle as _Circle

    A = np.asarray(A)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"A must be square (p, p); got {A.shape}")
    p = A.shape[0]

    gvals = _group_values(groups, p)
    have_groups = groups is not None

    if ax is None:
        _fig, ax = plt.subplots(figsize=(7, 6))

    G = nx.DiGraph()
    G.add_nodes_from(range(p))

    use_grouped = (layout == "grouped") or (layout == "auto" and have_groups)
    if use_grouped:
        pos = _grouped_cluster_positions(gvals, min_spacing, base_cluster_radius)
    elif layout in ("auto", "circular"):
        pos = nx.circular_layout(G)
    elif layout == "spring":
        pos = nx.spring_layout(G, seed=0)
    elif layout == "kamada":
        pos = (nx.kamada_kawai_layout(G) if G.number_of_edges()
               else nx.circular_layout(G))
    else:
        raise ValueError(
            f"unknown layout {layout!r}; expected 'auto', 'grouped', "
            f"'circular', 'spring', or 'kamada'")

    node_colors, legend = _node_colors(gvals, group_colors, plt)

    # Scale references for arcs / self-loops.
    xs = np.array([pos[i][0] for i in range(p)])
    ys = np.array([pos[i][1] for i in range(p)])
    center = np.array([xs.mean(), ys.mean()])
    span = float(max(xs.ptp(), ys.ptp(), 1e-9))
    short_thresh = 0.15 * span
    loop_r = 0.035 * span if span > 0 else 0.1

    # Edge width interpolation bounds.
    mags = []
    for i in range(p):
        for j in range(p):
            if i != j and _present(A[i, j]):
                mags.append(_mag(A[i, j]))
    mags = [m for m in mags if m > 0]
    lo, hi = edge_width_range
    if weight_widths and len(mags) > 1 and max(mags) > min(mags):
        wmin, wmax = min(mags), max(mags)

        def _lw(m):
            if m <= 0:
                return (lo + hi) / 2.0
            return float(np.interp(m, [wmin, wmax], [lo, hi]))
    else:
        def _lw(m):
            return (lo + hi) / 2.0

    def _arc(a, b, rad, color, lw, alpha, headsize, heads):
        """Quadratic-Bezier arc a->b with arrowhead(s) placed partway along."""
        p0 = np.array(pos[a], float)
        p1 = np.array(pos[b], float)
        mid = (p0 + p1) / 2.0
        d = p1 - p0
        perp = np.array([d[1], -d[0]])
        pc = mid + rad * perp
        ax.add_patch(_PP(_MP([p0, pc, p1],
                             [_MP.MOVETO, _MP.CURVE3, _MP.CURVE3]),
                         fill=False, edgecolor=color, lw=lw, alpha=alpha,
                         zorder=2, capstyle="round"))
        bez = lambda t: (1 - t) ** 2 * p0 + 2 * (1 - t) * t * pc + t ** 2 * p1
        for t0, t1 in heads:
            ax.add_patch(_FAP(bez(t0), bez(t1), arrowstyle="-|>",
                              mutation_scale=headsize, color=color, lw=lw,
                              alpha=alpha, zorder=3))

    def _self_loop(i, color, lw, alpha, headsize):
        x, y = pos[i]
        v = np.array([x, y]) - center
        nrm = np.linalg.norm(v)
        dirn = v / nrm if nrm > 1e-9 else np.array([1.0, 0.0])
        lc = np.array([x, y]) + dirn * loop_r
        ax.add_patch(_Circle(lc, loop_r, fill=False, edgecolor=color, lw=lw,
                             alpha=alpha, zorder=2))
        # small arrowhead tangent to the loop
        tangent = np.array([-dirn[1], dirn[0]])
        h0 = lc + tangent * loop_r
        h1 = h0 + tangent * (loop_r * 0.25)
        ax.add_patch(_FAP(h0, h1, arrowstyle="-|>", mutation_scale=headsize,
                          color=color, lw=lw, alpha=alpha, zorder=3))

    alpha_val = 0.75
    arrowsize = 14
    # Self-loops first.
    for i in range(p):
        if _present(A[i, i]):
            _self_loop(i, _sign_edge_color(0.0 if A[i, i] != A[i, i]
                                           else float(A[i, i]), sign_color,
                                           edge_color),
                       _lw(_mag(A[i, i])), alpha_val, arrowsize)

    for i in range(p):
        for j in range(i + 1, p):
            aij, aji = A[i, j], A[j, i]
            eij, eji = _present(aij), _present(aji)
            if not (eij or eji):
                continue
            m = max(_mag(aij), _mag(aji))
            lw = _lw(m)
            signed = float(aij) if (eij and _mag(aij) >= _mag(aji)) else \
                (float(aji) if eji and aji == aji else 0.0)
            col = _sign_edge_color(signed if signed == signed else 0.0,
                                   sign_color, edge_color)
            if eij and eji:
                L = float(np.hypot(pos[j][0] - pos[i][0],
                                   pos[j][1] - pos[i][1]))
                heads = ([(0.85, 0.93), (0.15, 0.07)] if L < short_thresh
                         else [(0.72, 0.80), (0.28, 0.20)])
                _arc(i, j, 0.14, col, lw, alpha_val, arrowsize, heads)
            else:
                u, v = (i, j) if eij else (j, i)
                _arc(u, v, 0.12, col, lw, alpha_val, arrowsize, [(0.72, 0.80)])

    nc = nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                                node_size=node_size, edgecolors="#333333",
                                linewidths=0.6, ax=ax)
    nc.set_zorder(5)
    lbls = {i: (str(labels[i]) if labels is not None else str(i))
            for i in range(p)}
    tobj = nx.draw_networkx_labels(G, pos, labels=lbls, font_size=9, ax=ax)
    for t in tobj.values():
        t.set_zorder(6)

    if legend:
        handles = [plt.Line2D([0], [0], marker="o", linestyle="",
                              markerfacecolor=col, markeredgecolor="#333333",
                              markersize=9, label=str(g))
                   for g, col in legend.items()]
        ax.legend(handles=handles, loc="upper left",
                  bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=9,
                  title="group", title_fontsize=9)

    ax.set_aspect("equal")
    ax.margins(0.08)
    ax.axis("off")
    if title:
        ax.set_title(title)
    return ax


# --------------------------------------------------------------------------
#  plot_matrix
# --------------------------------------------------------------------------

def plot_matrix(A, labels=None, groups=None, ax=None, title=None):
    """Adjacency heatmap.

    Diverging colormap centered at 0 for signed weights; sequential for
    all-nonnegative input. Optional subtle group separator lines when `groups`
    is given (assumes nodes are ordered by group). NaN entries are shown as a
    neutral 'no weight' color. The colorbar is given its own space so it does
    not overlap the matrix.

    Parameters
    ----------
    A : array_like, shape (p, p)
    labels : sequence of str, optional
    groups : dict or array, optional
        Node -> group mapping; a subtle separator line is drawn where the
        group changes between consecutive node indices.
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
        _fig, ax = plt.subplots(figsize=(6.2, 5.2))

    finite = A[np.isfinite(A)]
    has_neg = finite.size > 0 and (finite < 0).any()
    if has_neg:
        vmax = float(np.abs(finite).max()) or 1.0
        vmin, cmap = -vmax, plt.get_cmap("RdBu_r").copy()
    else:
        vmax = (float(finite.max()) if finite.size else 1.0) or 1.0
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
        gvals = _group_values(groups, p)
        for k in range(p - 1):
            if gvals[k] != gvals[k + 1]:
                ax.axhline(k + 0.5, color="#999999", linewidth=0.6)
                ax.axvline(k + 0.5, color="#999999", linewidth=0.6)

    if title:
        ax.set_title(title)
    return ax
