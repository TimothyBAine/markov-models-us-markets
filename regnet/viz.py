"""Static (matplotlib) and interactive (plotly) figures."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, ListedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from .universe import SECTOR_GROUP, SECTOR_ORDER  # noqa: E402

# ------------------------------------------------------------------ palette
INK, INK2, GRID, AXIS, SURFACE = "#191919", "#666666", "#e7e7e7", "#c2c2c2", "#ffffff"

# Categorical slots in fixed order (validated reference palette), one per colour group.
GROUP_COLORS = {
    "Tech & Comm": "#257de0",
    "Financials": "#f47230",
    "Health Care": "#00b894",
    "Energy": "#f2a600",
    "Staples": "#ee7fac",
    "Discretionary": "#008a17",
    "Industrials & Materials": "#6a4ce2",
    "Utilities & Real Estate": "#ec4647",
}
GROUP_ORDER = list(GROUP_COLORS)

# Regimes are ordinal (ordered by volatility / coupling): one blue ramp, light -> dark.
_BLUE = ["#84b9f3", "#6aabf1", "#519ced", "#338ced", "#257de0",
         "#2370cd", "#1b64be", "#1757ac", "#0f4a96", "#0e3d7e"]
DIVERGING = LinearSegmentedColormap.from_list(
    "corr", ["#0e3d7e", "#338ced", "#9dc8f7", "#efefef", "#f5a8a8", "#ec4647", "#9e1f20"])
SEQUENTIAL = LinearSegmentedColormap.from_list("seq", ["#f4f9ff", "#9dc8f7", "#338ced", "#0e3d7e"])

POS_EDGE, NEG_EDGE = "#9c9c9c", "#191919"
# Glasso keeps many tiny edges; draw only |partial rho| >= this (metrics use the full graph).
PCORR_DISPLAY_MIN = 0.08


def regime_colors(k: int) -> list[str]:
    return [_BLUE[i] for i in np.linspace(0, len(_BLUE) - 1, k).round().astype(int)]


def node_color(t: str, sectors: dict[str, str]) -> str:
    return GROUP_COLORS[SECTOR_GROUP[sectors[t]]]


plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif", "font.size": 9,
    "text.color": INK, "axes.labelcolor": INK2, "axes.edgecolor": AXIS,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "legend.frameon": False, "lines.linewidth": 1.5, "axes.axisbelow": True,
})


def _shade(ax, states: pd.Series, colors: list[str], alpha: float = 0.35):
    for k, c in enumerate(colors):
        ax.fill_between(states.index, 0, 1, where=(states == k).to_numpy(), step="post",
                        transform=ax.get_xaxis_transform(), color=c, alpha=alpha, lw=0)


# ------------------------------------------------------------------ timeline


def plot_timeline(mkt_px: pd.Series, rm_b, feats_a: pd.DataFrame, rm_a, path: Path):
    kb, ka = rm_b.n_states, rm_a.n_states
    cb, ca = regime_colors(kb), regime_colors(ka)
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True,
                             gridspec_kw={"height_ratios": [2.2, 1, 1.4]})

    ax = axes[0]
    px = mkt_px.reindex(rm_b.states.index)
    _shade(ax, rm_b.states, cb)
    ax.plot(px.index, px, color=INK, lw=1.2)
    ax.set_yscale("log")
    ax.set_ylabel("SPY (log scale)")
    ax.set_title("Model B: regimes from market features (Viterbi path)")
    ax.legend(handles=[Patch(color=c, alpha=0.6, label=n) for c, n in zip(cb, rm_b.names)],
              loc="upper left", ncol=kb)

    ax = axes[1]
    post = rm_b.posteriors
    ax.stackplot(post.index, post.T.to_numpy(), colors=cb, lw=0)
    ax.set_ylim(0, 1)
    ax.set_ylabel("P(regime | data)")
    ax.set_title("Model B: smoothed regime probabilities")

    ax = axes[2]
    _shade(ax, rm_a.states, ca)
    ax.plot(feats_a.index, feats_a["mean_corr"], color=INK, lw=1.2, label="Mean pairwise correlation")
    ax.plot(feats_a.index, feats_a["absorption"], color=INK2, lw=1.2, ls="--", label="Absorption ratio")
    ax.set_ylabel("Rolling network stat")
    ax.set_title("Model A: regimes from rolling network topology")
    names_a = [f"A{k}: {n}" for k, n in enumerate(rm_a.names)]
    ax.legend(handles=[Patch(color=c, alpha=0.6, label=n) for c, n in zip(ca, names_a)]
              + [Line2D([], [], color=INK, label="Mean pairwise correlation"),
                 Line2D([], [], color=INK2, ls="--", label="Absorption ratio")],
              loc="upper left", ncol=ka + 2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


# ------------------------------------------------------------------ networks


def _sector_legend(fig):
    handles = [Line2D([], [], marker="o", ls="", markersize=8, markerfacecolor=c,
                      markeredgecolor=SURFACE, label=g) for g, c in GROUP_COLORS.items()]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=9)


def _draw_graph(ax, G, pos, sectors, signed: bool, title: str):
    colors = [node_color(n, sectors) for n in G.nodes]
    if signed:
        strong = [(u, v, d) for u, v, d in G.edges(data=True) if d["weight"] >= PCORR_DISPLAY_MIN]
        pos_e = [(u, v) for u, v, d in strong if d["pcorr"] > 0]
        neg_e = [(u, v) for u, v, d in strong if d["pcorr"] < 0]
        w = nx.get_edge_attributes(G, "weight")
        nx.draw_networkx_edges(G, pos, edgelist=pos_e, ax=ax, edge_color=POS_EDGE,
                               width=[0.3 + 6 * w[e] for e in pos_e], alpha=0.8)
        nx.draw_networkx_edges(G, pos, edgelist=neg_e, ax=ax, edge_color=NEG_EDGE, style="--",
                               width=[0.3 + 6 * w[e] for e in neg_e], alpha=0.8)
    else:
        corr = nx.get_edge_attributes(G, "corr")
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color=POS_EDGE,
                               width=[0.5 + 3 * max(corr[e], 0) for e in G.edges])
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=colors, node_size=150,
                           edgecolors=SURFACE, linewidths=1.0)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=5.5, font_color=INK)
    ax.set_title(title)
    ax.axis("off")


def plot_regime_networks(regnets, full_net, sectors, path: Path):
    """Row 1: MST per regime (own layout). Row 2: partial-correlation networks on a
    fixed layout so edges appearing / vanishing across regimes are comparable."""
    k = len(regnets)
    fixed_pos = nx.spring_layout(full_net.pgraph, weight="weight", seed=7, iterations=300)
    fig, axes = plt.subplots(2, k, figsize=(6.2 * k, 12.5))
    for j, rn in enumerate(regnets):
        m = rn.metrics
        pos = nx.kamada_kawai_layout(rn.mst, weight="weight")
        _draw_graph(axes[0, j], rn.mst, pos, sectors, False,
                    f"{rn.name}: MST\nmean ρ={m['mean_corr']:.2f}, "
                    f"tree length={m['mst_length']:.2f}, hub={m['mst_hub']}")
        _draw_graph(axes[1, j], rn.pgraph, fixed_pos, sectors, True,
                    f"{rn.name}: partial-correlation network (glasso)\n"
                    f"{m['pcorr_edges']} edges, sector modularity={m['pcorr_modularity']:.2f}, "
                    f"n_eff={m['n_eff']:.0f}")
    _sector_legend(fig)
    fig.text(0.5, 0.035, f"Edge width ∝ |ρ| (MST) or |partial ρ|; partial-corr panels show |partial ρ| ≥ "
             f"{PCORR_DISPLAY_MIN} (counts in titles are the full glasso graph); dashed = negative",
             ha="center", color=INK2, fontsize=8)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, dpi=170)
    plt.close(fig)


# ------------------------------------------------------------------ heatmaps


def sector_sorted(tickers, sectors):
    return sorted(tickers, key=lambda t: (SECTOR_ORDER.index(sectors[t]), t))


def _sector_axes(ax, order, sectors):
    secs = [sectors[t] for t in order]
    bounds = [i for i in range(1, len(secs)) if secs[i] != secs[i - 1]]
    for b in bounds:
        ax.axhline(b - 0.5, color=SURFACE, lw=1.2)
        ax.axvline(b - 0.5, color=SURFACE, lw=1.2)
    edges = [0, *bounds, len(secs)]
    centers = [(a + b - 1) / 2 for a, b in zip(edges[:-1], edges[1:])]
    labels = [secs[a] for a in edges[:-1]]
    ax.set_yticks(centers, labels, fontsize=7)
    ax.set_xticks([])
    ax.grid(False)


def plot_corr_heatmaps(regnets, tickers, sectors, path: Path):
    order = sector_sorted(tickers, sectors)
    idx = [tickers.index(t) for t in order]
    k = len(regnets)
    fig = plt.figure(figsize=(4.6 * (k + 1) + 1.5, 5.4))
    gs = fig.add_gridspec(1, k + 4, width_ratios=[1] * k + [0.05, 0.12, 1, 0.05], wspace=0.08)
    axes = [fig.add_subplot(gs[0, j]) for j in range(k)] + [fig.add_subplot(gs[0, k + 2])]
    cax1, cax2 = fig.add_subplot(gs[0, k]), fig.add_subplot(gs[0, k + 3])
    for j, rn in enumerate(regnets):
        C = rn.C[np.ix_(idx, idx)]
        im = axes[j].imshow(C, cmap=DIVERGING, vmin=-1, vmax=1)
        axes[j].set_title(f"{rn.name}  (mean ρ = {rn.metrics['mean_corr']:.2f})")
        _sector_axes(axes[j], order, sectors)
        if j:
            axes[j].set_yticks([])
    fig.colorbar(im, cax=cax1, label="Correlation")
    D = (regnets[-1].C - regnets[0].C)[np.ix_(idx, idx)]
    lim = np.abs(D).max()
    im2 = axes[k].imshow(D, cmap=DIVERGING, vmin=-lim, vmax=lim)
    axes[k].set_title(f"{regnets[-1].name} − {regnets[0].name}")
    _sector_axes(axes[k], order, sectors)
    axes[k].set_yticks([])
    fig.colorbar(im2, cax=cax2, label="Δ correlation")
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ metrics


METRIC_LABELS = {
    "mean_corr": "Mean pairwise correlation",
    "absorption": "Absorption ratio (λ₁ / N)",
    "mst_length": "Normalised MST length",
    "intra_minus_inter": "Intra − inter sector correlation",
    "pcorr_density": "Partial-corr network density",
    "pcorr_modularity": "Sector modularity (partial corr)",
    "louvain_nmi": "Louvain communities vs sectors (NMI)",
    "n_eff": "Effective sample size (days)",
}


def plot_metrics(metrics: pd.DataFrame, path: Path):
    cols = [c for c in METRIC_LABELS if c in metrics.columns]
    colors = regime_colors(len(metrics))
    fig, axes = plt.subplots(2, 4, figsize=(14, 6.2))
    for ax, c in zip(axes.flat, cols):
        vals = metrics[c].astype(float)
        bars = ax.bar(metrics["name"], vals, color=colors, width=0.6, edgecolor=SURFACE, linewidth=2)
        ax.bar_label(bars, fmt="%.0f" if c == "n_eff" else "%.2f", fontsize=8, color=INK, padding=2)
        ax.set_title(METRIC_LABELS[c], fontsize=9)
        ax.grid(axis="x", visible=False)
        ax.set_ylim(min(0, vals.min() * 1.15), vals.max() * 1.18)
        ax.tick_params(axis="x", labelsize=8)
    for ax in list(axes.flat)[len(cols):]:
        ax.axis("off")
    fig.suptitle("Network topology by regime (Model B)", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_model_comparison(ct: pd.DataFrame, ari_point: float, ari_window: float, path: Path):
    fig, ax = plt.subplots(figsize=(5.8, 4.4))
    ax.imshow(ct.to_numpy(), cmap=SEQUENTIAL, vmin=0, vmax=1)
    for (i, j), v in np.ndenumerate(ct.to_numpy()):
        ax.text(j, i, f"{v:.0%}", ha="center", va="center",
                color=SURFACE if v > 0.55 else INK, fontsize=9)
    ax.set_xticks(range(ct.shape[1]), [c.replace(": ", ":\n") for c in ct.columns])
    ax.set_yticks(range(ct.shape[0]), ct.index)
    ax.set_xlabel("Model A state (rolling topology)")
    ax.set_ylabel("Model B state (market features)")
    ax.grid(False)
    ax.set_title(f"Row-normalised agreement\nARI point-in-time={ari_point:.2f}, "
                 f"window-majority={ari_window:.2f}", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


# ------------------------------------------------------------------ interactive


def interactive_networks(regnets, full_net, sectors, path: Path):
    """Self-contained HTML: switch between regimes and between MST / partial-corr views."""
    import plotly.graph_objects as go

    fixed_pos = nx.spring_layout(full_net.pgraph, weight="weight", seed=7, iterations=300)
    fig = go.Figure()
    views = []   # (label, [trace indices])

    def add_view(label, G, pos, signed, metrics):
        start = len(fig.data)
        strength = dict(G.degree(weight="weight"))
        if signed:
            buckets = [("Positive partial ρ", POS_EDGE, "solid", lambda d: d["pcorr"] > 0),
                       ("Negative partial ρ", NEG_EDGE, "dash", lambda d: d["pcorr"] < 0)]
        else:
            buckets = [("MST edge", POS_EDGE, "solid", lambda d: True)]
        for name, color, dash, keep in buckets:
            for lo, hi, width in ([(PCORR_DISPLAY_MIN, 0.15, 1.0), (0.15, 0.3, 2.2), (0.3, 9, 3.8)] if signed
                                  else [(0, 0.1, 0.8), (0.1, 0.25, 2.0), (0.25, 9, 3.5)]):
                xs, ys = [], []
                for u, v, d in G.edges(data=True):
                    s = d["weight"] if signed else max(d["corr"], 0) / 2
                    if keep(d) and lo <= s < hi:
                        xs += [pos[u][0], pos[v][0], None]
                        ys += [pos[u][1], pos[v][1], None]
                fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", hoverinfo="skip",
                                         line=dict(color=color, width=width, dash=dash),
                                         name=name, legendgroup=name, showlegend=(width < 1.5),
                                         visible=False))
        for g, c in GROUP_COLORS.items():
            nodes = [n for n in G.nodes if SECTOR_GROUP[sectors[n]] == g]
            fig.add_trace(go.Scatter(
                x=[pos[n][0] for n in nodes], y=[pos[n][1] for n in nodes],
                mode="markers+text", text=nodes, textposition="top center",
                textfont=dict(size=9, color=INK),
                marker=dict(size=13, color=c, line=dict(color=SURFACE, width=1.5)),
                customdata=[[sectors[n], G.degree(n), strength[n]] for n in nodes],
                hovertemplate="<b>%{text}</b><br>%{customdata[0]}<br>degree %{customdata[1]}"
                              "<br>strength %{customdata[2]:.2f}<extra></extra>",
                name=g, legendgroup=g, visible=False))
        title = (f"{label} — mean ρ {metrics['mean_corr']:.2f}, absorption {metrics['absorption']:.2f}, "
                 f"glasso edges {metrics['pcorr_edges']} (showing |partial ρ| ≥ {PCORR_DISPLAY_MIN}), sector modularity {metrics['pcorr_modularity']:.2f}")
        views.append((label, list(range(start, len(fig.data))), title))

    for rn in regnets:
        add_view(f"{rn.name} · partial corr", rn.pgraph, fixed_pos, True, rn.metrics)
    for rn in regnets:
        add_view(f"{rn.name} · MST", rn.mst, nx.kamada_kawai_layout(rn.mst, weight="weight"),
                 False, rn.metrics)

    n = len(fig.data)
    buttons = []
    for label, idx, title in views:
        vis = [False] * n
        for i in idx:
            vis[i] = True
        buttons.append(dict(label=label, method="update",
                            args=[{"visible": vis}, {"title.text": title}]))
    for i in views[0][1]:
        fig.data[i].visible = True
    axis = dict(showgrid=False, zeroline=False, showticklabels=False)
    fig.update_layout(
        title=dict(text=views[0][2], x=0.01, font=dict(size=14, color=INK)),
        updatemenus=[dict(buttons=buttons, direction="down", x=0.01, y=1.06, xanchor="left")],
        xaxis=axis, yaxis={**axis, "scaleanchor": "x"},
        plot_bgcolor=SURFACE, paper_bgcolor=SURFACE, height=850,
        font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif", color=INK),
        legend=dict(orientation="h", y=-0.03), margin=dict(t=110, l=20, r=20, b=40),
    )
    fig.write_html(path, include_plotlyjs=True, full_html=True)
