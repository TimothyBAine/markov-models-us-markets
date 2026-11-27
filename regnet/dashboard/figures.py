"""Plotly figure builders for the dashboard."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..universe import SECTOR_GROUP
from ..viz import GROUP_COLORS, regime_colors

INK, INK2, GRID, AXIS, SURFACE = "#191919", "#666666", "#e7e7e7", "#c2c2c2", "#ffffff"
POS_EDGE, NEG_EDGE = "#8c8c8c", "#8c8c8c"   # sign is carried by dash style
FONT = dict(family="system-ui, -apple-system, Segoe UI, sans-serif", color=INK, size=12)

KIND_LABEL = {"pcorr": "partial ρ", "mst": "ρ", "thr": "ρ"}


def _hex_rgba(h: str, a: float) -> str:
    h = h.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{a})"


def empty_figure(msg: str = "") -> go.Figure:
    fig = go.Figure()
    fig.update_layout(xaxis=dict(visible=False), yaxis=dict(visible=False),
                      plot_bgcolor=SURFACE, paper_bgcolor=SURFACE, font=FONT,
                      annotations=[dict(text=msg, showarrow=False, font=dict(color=INK2))],
                      margin=dict(l=10, r=10, t=10, b=10))
    return fig


def network_figure(net, pos: dict, sectors: dict, draw_min: float,
                   regime_color: str | None = None, height: int = 560,
                   show_legend: bool = True, node_size: int = 14) -> go.Figure:
    """Edges bucketed into three width classes (plotly lines can't vary width within a
    trace); positive solid grey, negative dashed ink. Nodes coloured by sector group."""
    fig = go.Figure()
    label = KIND_LABEL[net.kind]
    shown = [(u, v, w) for u, v, w in net.edges if abs(w) >= draw_min or net.kind == "mst"]
    mags = np.array([abs(w) for *_, w in shown]) if shown else np.array([0.0])
    cuts = np.quantile(mags, [1 / 3, 2 / 3]) if len(mags) > 3 else [np.inf, np.inf]
    for sign, color, dash, name in [(1, POS_EDGE, "solid", f"Positive {label}"),
                                    (-1, NEG_EDGE, "dash", f"Negative {label}")]:
        first = True
        for lo, hi, width in [(-np.inf, cuts[0], 0.8), (cuts[0], cuts[1], 1.8), (cuts[1], np.inf, 3.4)]:
            xs, ys = [], []
            for u, v, w in shown:
                if np.sign(w) == sign and lo <= abs(w) < hi:
                    xs += [pos[u][0], pos[v][0], None]
                    ys += [pos[u][1], pos[v][1], None]
            if not xs:
                continue
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", hoverinfo="skip",
                                     line=dict(color=color, width=width, dash=dash),
                                     name=name, legendgroup=name, showlegend=first and show_legend))
            first = False

    deg = {n: 0 for n in pos}
    strength = {n: 0.0 for n in pos}
    for u, v, w in net.edges:
        deg[u] += 1
        deg[v] += 1
        strength[u] += abs(w)
        strength[v] += abs(w)
    for g, c in GROUP_COLORS.items():
        nodes = [n for n in pos if SECTOR_GROUP[sectors[n]] == g]
        fig.add_trace(go.Scatter(
            x=[pos[n][0] for n in nodes], y=[pos[n][1] for n in nodes],
            mode="markers+text", text=nodes, textposition="top center",
            textfont=dict(size=9, color=INK),
            marker=dict(size=node_size, color=c, line=dict(color=SURFACE, width=1.5)),
            customdata=[[sectors[n], deg[n], strength[n]] for n in nodes],
            hovertemplate="<b>%{text}</b> · %{customdata[0]}<br>degree %{customdata[1]}"
                          "<br>strength %{customdata[2]:.2f}<extra></extra>",
            name=g, legendgroup=g, showlegend=show_legend))

    axis = dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1.08, 1.08])
    fig.update_layout(
        height=height, plot_bgcolor=SURFACE, paper_bgcolor=SURFACE, font=FONT,
        xaxis=axis, yaxis={**axis, "range": [-1.1, 1.12]},
        margin=dict(l=8, r=8, t=8, b=8), hovermode="closest",
        legend=dict(orientation="h", y=-0.02, font=dict(size=10, color=INK2)),
        uirevision="network",
    )
    if regime_color:
        fig.add_shape(type="rect", xref="paper", yref="paper", x0=0, y0=0, x1=1, y1=1,
                      line=dict(color=regime_color, width=4), layer="below")
    return fig


def _runs(states: pd.Series):
    change = states.ne(states.shift()).cumsum()
    for _, s in states.groupby(change):
        yield int(s.iloc[0]), s.index[0], s.index[-1]


def timeline_figure(spy: pd.Series, rm, rolling: pd.DataFrame, sel: tuple, window: int) -> go.Figure:
    """SPY with regime shading (top) and rolling mean correlation raw vs residual
    (bottom; same unit, one axis). Shape 0 is the selected window, patched on slide."""
    colors = regime_colors(rm.n_states)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                        row_heights=[0.62, 0.38])
    shapes = [dict(type="rect", xref="x", yref="paper", x0=sel[0], x1=sel[1], y0=0, y1=1,
                   fillcolor="rgba(25,25,25,0.10)", line=dict(color=INK, width=1.2), layer="above")]
    for k, a, b in _runs(rm.states):
        shapes.append(dict(type="rect", xref="x", yref="paper", x0=a, x1=b, y0=0, y1=1,
                           fillcolor=_hex_rgba(colors[k], 0.45), line=dict(width=0), layer="below"))
    fig.add_trace(go.Scatter(x=spy.index, y=spy, mode="lines", line=dict(color=INK, width=1.3),
                             name="SPY", showlegend=False,
                             hovertemplate="%{x|%Y-%m-%d}<br>SPY %{y:.1f}<extra></extra>"), 1, 1)
    for k, (c, n) in enumerate(zip(colors, rm.names)):   # legend swatches for regimes
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=n,
                                 marker=dict(symbol="square", size=12, color=_hex_rgba(c, 0.7))), 1, 1)
    fig.add_trace(go.Scatter(x=rolling.index, y=rolling.raw, mode="lines", name="Mean ρ (raw)",
                             line=dict(color=INK, width=1.5),
                             hovertemplate="%{x|%Y-%m-%d}<br>raw %{y:.2f}<extra></extra>"), 2, 1)
    fig.add_trace(go.Scatter(x=rolling.index, y=rolling.residual, mode="lines",
                             name="Mean ρ (market removed)", line=dict(color=INK2, width=1.5, dash="dash"),
                             hovertemplate="%{x|%Y-%m-%d}<br>residual %{y:.2f}<extra></extra>"), 2, 1)
    fig.update_yaxes(type="log", title_text="SPY (log)", row=1, col=1, gridcolor=GRID,
                     tickvals=[50, 100, 200, 400, 800],
                     title_font=dict(color=INK2, size=11))
    fig.update_yaxes(title_text=f"{window}d mean ρ", row=2, col=1, gridcolor=GRID,
                     zeroline=True, zerolinecolor=AXIS, title_font=dict(color=INK2, size=11))
    fig.update_xaxes(gridcolor=GRID, linecolor=AXIS)
    fig.update_layout(
        shapes=shapes, height=330, plot_bgcolor=SURFACE, paper_bgcolor=SURFACE, font=FONT,
        margin=dict(l=60, r=10, t=36, b=10), hovermode="x unified",
        legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0, font=dict(size=11, color=INK2)),
        uirevision="timeline",
    )
    return fig
