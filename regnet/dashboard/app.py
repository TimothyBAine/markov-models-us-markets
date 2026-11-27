"""Dash app: regime-highlighted stock networks through time, with raw vs
market-removed comparison and sparsity controls."""
from pathlib import Path

import numpy as np
import pandas as pd
from dash import Dash, Input, Output, Patch, State, dcc, html, no_update

from ..viz import GROUP_COLORS, regime_colors
from .compute import Store
from .figures import network_figure, timeline_figure
from .text import CAPTIONS, CONTROL_TIPS, GUIDE, METRIC_TIPS

ALPHAS = [0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3]
STEPS = {"1 week": 1, "1 month": 4, "1 quarter": 13}
MODES = {"raw": "Raw returns", "residual": "Market removed", "compare": "Compare both"}
KINDS = {"pcorr": "Partial correlation (glasso)", "mst": "Minimum spanning tree", "thr": "Correlation threshold"}
SPARSITY = {"ebic": "Auto (EBIC)", "alpha": "Fixed penalty α", "topn": "Fixed edge count"}


# ------------------------------------------------------------------ small components


def _help(tip):
    """Small (i) icon; the tooltip shows on hover and on keyboard focus."""
    return html.Span("i", className="help", tabIndex="0", role="note", **{"data-tip": tip, "aria-label": tip})


def _control(label, child, cid=None, hidden=False, tip=None):
    kw = {"id": cid} if cid else {}
    head = html.Div([html.Label(label, className="ctl-label")] + ([_help(tip)] if tip else []), className="ctl-head")
    return html.Div([head, child], className="ctl", style={"display": "none"} if hidden else {}, **kw)


def _chip(label, value, tip=None):
    kw = {"data-tip": tip, "tabIndex": "0"} if tip else {}
    return html.Span([html.Span(label, className="chip-k"), html.Span(value, className="chip-v")],
                     className="chip has-tip" if tip else "chip", **kw)


def _fmt(x, nd=2):
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def _metric_chips(m: dict, kind: str):
    t = METRIC_TIPS
    chips = [_chip("edges", f"{m['edges']}", t["edges"]), _chip("density", _fmt(m["density"]), t["density"]),
             _chip("mean ρ", _fmt(m["mean_corr"]), t["mean_corr"]),
             _chip("λ₁/N", _fmt(m["absorption"]), t["absorption"]),
             _chip("sector modularity", _fmt(m["modularity"]), t["modularity"]),
             _chip("Louvain/sector NMI", _fmt(m["nmi"]), t["nmi"]),
             _chip("hub", f"{m['hub']} ({m['hub_degree']})", t["hub"])]
    if kind == "pcorr":
        chips += [_chip("glasso α", _fmt(m.get("alpha"), 3), t["alpha"]),
                  _chip("negative", f"{m['neg_share']:.0%}", t["neg"]) if not np.isnan(m["neg_share"]) else None]
    chips.append(_chip("n", f"{m['n_eff']:.0f}", t["n"]))
    return html.Div([c for c in chips if c], className="chips")


def _regime_badge(name, color, detail=""):
    return html.Span([html.Span(className="swatch", style={"background": color}),
                      html.B(name), html.Span(detail, className="muted")], className="badge")


def _sector_legend():
    return html.Div([html.Span([html.Span(className="dot", style={"background": c}), g], className="legend-item")
                     for g, c in GROUP_COLORS.items()]
                    + [html.Span([html.Span(className="line solid"), "positive"], className="legend-item"),
                       html.Span([html.Span(className="line dashed"), "negative"], className="legend-item")],
                    className="legend")


# ------------------------------------------------------------------ app


def create_app(data_dir: Path = Path("data")) -> Dash:
    store = Store(data_dir)
    store.regimes(3)
    store.fixed_layout()
    store.sector_layout()
    store.rolling_corr(126)
    frames, dates = store.frames, store.dates
    p = len(store.tickers)

    year_marks = {}
    for i, e in enumerate(frames):
        y = dates[e - 1].year
        if y not in year_marks.values() and y % 2 == 0:
            year_marks[i] = y
    year_marks = {i: str(y) for i, y in year_marks.items()}

    app = Dash(__name__, title="Regime Network Explorer",
               assets_folder=str(Path(__file__).parent / "assets"))

    controls = html.Div([
        _control("Regimes (K)", dcc.Dropdown(id="k", options=[2, 3, 4, 5], value=3, clearable=False),
                 tip=CONTROL_TIPS["k"]),
        _control("Market factor", dcc.RadioItems(id="mode", options=MODES, value="compare", inline=True),
                 tip=CONTROL_TIPS["mode"]),
        _control("Graph", dcc.Dropdown(id="kind", options=KINDS, value="pcorr", clearable=False),
                 tip=CONTROL_TIPS["kind"]),
        _control("Sparsity", dcc.RadioItems(id="sparsity", options=SPARSITY, value="topn", inline=True),
                 cid="ctl-sparsity", tip=CONTROL_TIPS["sparsity"]),
        _control("Penalty α", dcc.Slider(id="alpha", min=0, max=len(ALPHAS) - 1, step=1, value=4,
                                         marks={i: str(a) for i, a in enumerate(ALPHAS)}), cid="ctl-alpha", hidden=True, tip=CONTROL_TIPS["alpha"]),
        _control("Edges kept", dcc.Slider(id="topn", min=30, max=400, step=10, value=100,
                                          marks={m: str(m) for m in (30, 100, 200, 300, 400)}),
                 cid="ctl-topn", tip=CONTROL_TIPS["topn"]),
        _control("Correlation threshold τ", dcc.Slider(id="tau", min=0.1, max=0.8, step=0.05, value=0.5,
                                                 marks={v: f"{v:.1f}" for v in (0.1, 0.3, 0.5, 0.7)}),
                 cid="ctl-tau", hidden=True, tip=CONTROL_TIPS["tau"]),
        _control("Hide links weaker than (|partial ρ|)", dcc.Slider(id="drawmin", min=0, max=0.3, step=0.01, value=0.0,
                                                            marks={v: f"{v:.2f}" for v in (0, 0.1, 0.2, 0.3)}),
                 cid="ctl-drawmin", tip=CONTROL_TIPS["drawmin"]),
        _control("Window (days)", dcc.Dropdown(id="window", options=[63, 126, 252], value=126, clearable=False),
                 tip=CONTROL_TIPS["window"]),
        _control("Layout", dcc.RadioItems(id="layout", options={"ring": "Sector ring", "fixed": "Force (fixed)",
                                                                "adaptive": "Adaptive"},
                                          value="ring", inline=True), tip=CONTROL_TIPS["layout"]),
    ], className="controls")

    time_tab = html.Div([
        dcc.Graph(id="timeline", config={"displayModeBar": False}),
        html.P(CAPTIONS["timeline"], className="caption"),
        html.Div([
            html.Button("▶ Play", id="play", n_clicks=0, className="btn"),
            dcc.Dropdown(id="step", options=list(STEPS), value="1 month", clearable=False, className="step"),
            html.Div(dcc.Slider(id="frame", min=0, max=len(frames) - 1, step=1, value=len(frames) - 1,
                                marks=year_marks, updatemode="mouseup", allow_direct_input=False), className="frame-slider"),
        ], className="player"),
        html.Div(id="status", className="status"),
        dcc.Loading(html.Div(id="time-networks", className="net-grid"), type="dot", delay_show=400),
        dcc.Interval(id="tick", interval=900, disabled=True),
        dcc.Store(id="rendered"),
    ])

    regime_tab = html.Div([
        html.P(CAPTIONS["by_regime"], className="caption"),
        dcc.Loading(html.Div(id="regime-networks"), type="dot", delay_show=400),
    ])

    app.layout = html.Div([
        html.Header([html.H1("Regime Network Explorer"),
                     html.P(f"{p} US large caps · {dates[0]:%b %Y} – {dates[-1]:%b %Y} · "
                            "HMM regimes × dependence networks", className="muted")]),
        html.Details([html.Summary("How to read this dashboard"),
                      html.Ul([html.Li(g) for g in GUIDE])], className="guide"),
        controls,
        html.Div(id="warning", className="warning"),
        _sector_legend(),
        dcc.Tabs(id="tab", value="time", children=[
            dcc.Tab(label="Through time", value="time", children=time_tab),
            dcc.Tab(label="By regime", value="regime", children=regime_tab),
        ]),
    ], className="page")

    # ---------------------------------------------------------- helpers

    def params(kind, sparsity, alpha_i, topn, tau):
        """Canonical cache key: irrelevant settings are blanked out."""
        if kind != "pcorr":
            sparsity = alpha = topn = None
        else:
            alpha = ALPHAS[alpha_i] if sparsity == "alpha" else None
            topn = topn if sparsity == "topn" else None
        tau = tau if kind == "thr" else None
        return kind, sparsity, alpha, topn, tau

    def modes(mode):
        return [False, True] if mode == "compare" else [mode == "residual"]

    def positions(net, layout):
        if layout == "ring":
            return store.sector_layout()
        return store.fixed_layout() if layout == "fixed" else store.relayout(net)

    def panel(title, badge, net, layout, drawmin, color, height, kind):
        dm = drawmin if kind == "pcorr" else 0.0
        fig = network_figure(net, positions(net, layout), store.sectors, dm, color, height, show_legend=False)
        return html.Div([html.Div([html.H3(title), badge], className="panel-head"),
                         _metric_chips(net.metrics, kind),
                         dcc.Graph(figure=fig, config={"displayModeBar": False})], className="panel")

    # ---------------------------------------------------------- control visibility

    @app.callback(Output("ctl-sparsity", "style"), Output("ctl-alpha", "style"), Output("ctl-topn", "style"),
                  Output("ctl-tau", "style"), Output("ctl-drawmin", "style"),
                  Input("kind", "value"), Input("sparsity", "value"))
    def _visibility(kind, sparsity):
        show, hide = {}, {"display": "none"}
        pc = kind == "pcorr"
        return (show if pc else hide, show if pc and sparsity == "alpha" else hide,
                show if pc and sparsity == "topn" else hide, show if kind == "thr" else hide,
                show if pc else hide)

    @app.callback(Output("warning", "children"), Input("kind", "value"), Input("window", "value"),
                  Input("sparsity", "value"), Input("tab", "value"))
    def _warn(kind, window, sparsity, tab):
        msgs = []
        if tab == "time" and kind == "pcorr" and window < 1.5 * p:
            msgs.append(f"A {window}-day window is short for {p} stocks (under 1.5 days per stock), "
                        "so partial correlations are poorly determined. Use a larger α or a longer window.")
        if tab == "time" and kind == "pcorr" and sparsity == "ebic":
            msgs.append("Auto (EBIC) fits several penalties per frame (~1–3 s each), so playback is slow. "
                        "Fixed α or a fixed edge count animate smoothly and keep sparsity comparable across time.")
        return [html.Div(m) for m in msgs]

    # ---------------------------------------------------------- timeline

    def sel_range(i, window):
        end = int(frames[i])
        return dates[end - window], dates[end - 1]

    @app.callback(Output("timeline", "figure"), Input("k", "value"), Input("window", "value"),
                  State("frame", "value"))
    def _timeline(k, window, i):
        return timeline_figure(store.spy, store.regimes(k), store.rolling_corr(window),
                               sel_range(i, window), window)

    @app.callback(Output("timeline", "figure", allow_duplicate=True), Input("frame", "value"),
                  State("window", "value"), prevent_initial_call=True)
    def _timeline_sel(i, window):
        a, b = sel_range(i, window)
        pt = Patch()
        pt["layout"]["shapes"][0]["x0"] = a
        pt["layout"]["shapes"][0]["x1"] = b
        return pt

    @app.callback(Output("frame", "value", allow_duplicate=True), Input("timeline", "clickData"),
                  prevent_initial_call=True)
    def _click(click):
        if not click:
            return no_update
        t = pd.Timestamp(click["points"][0]["x"])
        ends = dates[frames - 1]
        return int(np.argmin(np.abs((ends - t).days)))

    # ---------------------------------------------------------- playback

    @app.callback(Output("tick", "disabled"), Output("play", "children"), Input("play", "n_clicks"),
                  State("tick", "disabled"), prevent_initial_call=True)
    def _toggle(_, disabled):
        playing = disabled          # was paused -> now playing
        return (not playing), ("❚❚ Pause" if playing else "▶ Play")

    @app.callback(Output("frame", "value"), Input("tick", "n_intervals"), State("frame", "value"),
                  State("step", "value"), State("rendered", "data"), prevent_initial_call=True)
    def _advance(_, i, step, rendered):
        if rendered != i:           # back-pressure: wait until the current frame has rendered
            return no_update
        if i >= len(frames) - 1:
            return 0                # loop back to the start
        return min(i + STEPS[step], len(frames) - 1)

    # ---------------------------------------------------------- networks through time

    @app.callback(Output("time-networks", "children"), Output("status", "children"), Output("rendered", "data"),
                  Input("frame", "value"), Input("k", "value"), Input("mode", "value"), Input("kind", "value"),
                  Input("sparsity", "value"), Input("alpha", "value"), Input("topn", "value"),
                  Input("tau", "value"), Input("drawmin", "value"), Input("window", "value"),
                  Input("layout", "value"), Input("tab", "value"))
    def _time_networks(i, k, mode, kind, sparsity, alpha_i, topn, tau, drawmin, window, layout, tab):
        if tab != "time":
            return no_update, no_update, no_update
        end = int(frames[i])
        key = params(kind, sparsity, alpha_i, topn, tau)
        reg = store.regime_at(k, end, window)
        color = regime_colors(k)[reg["state"]]
        mix = " · ".join(f"{n} {v:.0%}" for n, v in reg["mix"].items())
        status = html.Div([
            html.Span(f"Window {dates[end - window]:%d %b %Y} → {dates[end - 1]:%d %b %Y}", className="win"),
            _regime_badge(reg["name"], color, f"  P={reg['p_end']:.2f} at window end"),
            html.Span(f"Regime mix over window: {mix}", className="muted has-tip", tabIndex="0",
                      **{"data-tip": METRIC_TIPS["regime_mix"]}),
        ])
        panels = []
        for resid in modes(mode):
            net = store.window_network(end, window, resid, *key)
            title = "Market removed" if resid else "Raw returns"
            panels.append(panel(title, None, net, layout, drawmin, color,
                                560 if mode == "compare" else 680, kind))
        cls = "cols-2" if len(panels) == 2 else "cols-1"
        return html.Div(panels, className=f"net-grid {cls}"), status, i

    # ---------------------------------------------------------- networks by regime

    @app.callback(Output("regime-networks", "children"),
                  Input("k", "value"), Input("mode", "value"), Input("kind", "value"),
                  Input("sparsity", "value"), Input("alpha", "value"), Input("topn", "value"),
                  Input("tau", "value"), Input("drawmin", "value"), Input("layout", "value"),
                  Input("tab", "value"))
    def _regime_networks(k, mode, kind, sparsity, alpha_i, topn, tau, drawmin, layout, tab):
        if tab != "regime":
            return no_update
        rm = store.regimes(k)
        colors = regime_colors(k)
        key = params(kind, sparsity, alpha_i, topn, tau)
        summary = []
        blocks = []
        for resid in modes(mode):
            row = []
            for s in range(k):
                net = store.regime_network(k, s, resid, *key)
                share = float((rm.states == s).mean())
                badge = _regime_badge(rm.names[s], colors[s], f"  {share:.0%} of days")
                row.append(panel("", badge, net, layout, drawmin, colors[s], 440, kind))
                summary.append({"Market": "removed" if resid else "raw", "Regime": rm.names[s], **net.metrics})
            blocks.append(html.H2("Market removed" if resid else "Raw returns"))
            blocks.append(html.Div(row, className="net-grid", style={"gridTemplateColumns": f"repeat({k}, 1fr)"}))
        df = pd.DataFrame(summary)
        cols = ["Market", "Regime", "edges", "density", "mean_corr", "intra_minus_inter", "absorption",
                "modularity", "nmi", "neg_share", "hub"]
        df = df[cols]
        heads = ["Market", "Regime", "Edges", "Density", "Mean ρ", "Intra − inter ρ", "λ₁/N",
                 "Sector modularity", "Louvain NMI", "Negative edges", "Hub"]
        rows = [html.Tr([html.Td(r.Market), html.Td(r.Regime), html.Td(r.edges), html.Td(_fmt(r.density)),
                         html.Td(_fmt(r.mean_corr)), html.Td(_fmt(r.intra_minus_inter)), html.Td(_fmt(r.absorption)),
                         html.Td(_fmt(r.modularity)), html.Td(_fmt(r.nmi)),
                         html.Td("–" if np.isnan(r.neg_share) else f"{r.neg_share:.0%}"), html.Td(r.hub)])
                for r in df.itertuples()]
        table = html.Table([html.Thead(html.Tr([html.Th(h) for h in heads])), html.Tbody(rows)],
                           className="metrics")
        return blocks + [html.H2("Metrics"), table]

    return app
