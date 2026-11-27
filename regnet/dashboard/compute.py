"""Data + network computation for the dashboard, with caching.

Residualization here is *local*: within a rolling window, each stock's market beta
is estimated on that window; for a regime network, betas are posterior-weighted.
(The full-sample-beta version used by the batch pipeline gives near-identical
regime results; local betas are the right choice when stepping through time.)
"""
import warnings
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.covariance import graphical_lasso
from sklearn.exceptions import ConvergenceWarning

from ..data import clean_returns, load_prices
from ..features import market_features
from ..hmm import fit_hmm
from ..networks import (absorption_ratio, corr_from_cov, glasso_ebic, louvain_vs_sector,
                        mst_graph, partial_corr, sector_modularity)
from ..universe import MARKET, TICKER_SECTOR, TICKERS

FRAME_STEP = 5          # slider granularity (trading days)
MAX_WINDOW = 252        # first frame leaves room for the longest window
EBIC_ALPHAS = np.logspace(-1.7, -0.4, 6)


@dataclass
class Network:
    edges: list[tuple[str, str, float]]   # (u, v, signed weight)
    metrics: dict
    kind: str


# ------------------------------------------------------------------ helpers


def _weighted_resid_corr(X: np.ndarray, m: np.ndarray, w: np.ndarray | None,
                         residual: bool) -> tuple[np.ndarray, float]:
    """(Weighted) correlation of X, optionally after removing (weighted) market beta."""
    if w is None:
        w = np.ones(len(X))
    w = w / w.sum()
    Xc = X - w @ X
    if residual:
        mc = m - w @ m
        beta = (Xc * (w * mc)[:, None]).sum(0) / (w * mc * mc).sum()
        Xc = Xc - np.outer(mc, beta)
    S = (Xc * w[:, None]).T @ Xc
    return corr_from_cov(S), 1.0 / np.sum(w ** 2)


def _glasso(C: np.ndarray, alpha: float) -> tuple[np.ndarray, float]:
    """Graphical lasso that backs off to a larger alpha if the solver fails
    (happens when the window is short relative to the number of stocks)."""
    a = alpha
    for _ in range(8):
        try:
            with warnings.catch_warnings(), np.errstate(all="ignore"):
                warnings.simplefilter("ignore", ConvergenceWarning)
                warnings.simplefilter("ignore", RuntimeWarning)
                _, Th = graphical_lasso(C, alpha=a, max_iter=200, tol=1e-4)
            if np.all(np.isfinite(Th)):
                return Th, a
        except (FloatingPointError, ValueError):
            pass
        a *= 1.5
    raise RuntimeError("graphical lasso failed")


def _normalise(pos: dict) -> dict:
    """Scale each axis to [-1, 1] so the graph fills a landscape panel."""
    xy = np.array(list(pos.values()))
    lo, hi = xy.min(0), xy.max(0)
    xy = 2 * (xy - lo) / np.where(hi > lo, hi - lo, 1) - 1
    return {n: tuple(p) for n, p in zip(pos, xy)}


# ------------------------------------------------------------------ store


class Store:
    def __init__(self, data_dir: Path = Path("data"), start: str = "2005-01-01"):
        px = load_prices(TICKERS + [MARKET], start, None, data_dir)
        rets, _ = clean_returns(px)
        mkt = rets.pop(MARKET)
        self.feats = market_features(rets, mkt, 10)
        self.dates = self.feats.index
        self.rets = rets.loc[self.dates]
        self.mkt = mkt.loc[self.dates]
        self.spy = px[MARKET].reindex(self.dates).ffill()
        self.tickers = list(self.rets.columns)
        self.sectors = {t: TICKER_SECTOR[t] for t in self.tickers}
        self.X = self.rets.to_numpy()
        self.m = self.mkt.to_numpy()
        self.frames = np.arange(MAX_WINDOW, len(self.dates) + 1, FRAME_STEP)  # window end (exclusive)
        if self.frames[-1] != len(self.dates):
            self.frames = np.append(self.frames, len(self.dates))

    # ---- regimes
    @lru_cache(maxsize=8)
    def regimes(self, k: int):
        return fit_hmm(self.feats, k, order_by="log_vol", n_restarts=5, seed=0)

    def regime_at(self, k: int, end: int, window: int) -> dict:
        rm = self.regimes(k)
        post = rm.posteriors.iloc[end - window:end].mean().to_numpy()
        state = int(rm.states.iloc[end - 1])
        return {"state": state, "name": rm.names[state],
                "p_end": float(rm.posteriors.iloc[end - 1, state]),
                "mix": dict(zip(rm.names, post.round(3)))}

    # ---- correlation matrices
    @lru_cache(maxsize=4096)
    def window_corr(self, end: int, window: int, residual: bool) -> tuple[np.ndarray, float]:
        sl = slice(end - window, end)
        return _weighted_resid_corr(self.X[sl], self.m[sl], None, residual)

    @lru_cache(maxsize=64)
    def regime_corr(self, k: int, state: int, residual: bool) -> tuple[np.ndarray, float]:
        w = self.regimes(k).posteriors[state].to_numpy()
        return _weighted_resid_corr(self.X, self.m, w, residual)

    # ---- networks
    def build(self, C: np.ndarray, n: float, kind: str, sparsity: str, alpha: float,
              top_n: int, tau: float) -> Network:
        t = self.tickers
        info: dict = {}
        if kind == "pcorr":
            if sparsity == "ebic":
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    try:
                        Th, a, _ = glasso_ebic(C, n, alphas=EBIC_ALPHAS)
                    except RuntimeError:
                        Th, a = _glasso(C, EBIC_ALPHAS[-1])
            elif sparsity == "topn":
                Th, a = _glasso(C, 0.03)
            else:
                Th, a = _glasso(C, alpha)
            P = partial_corr(Th)
            i, j = np.where(np.abs(np.triu(P, 1)) > 1e-6)
            vals = P[i, j]
            if sparsity == "topn" and len(vals) > top_n:
                keep = np.argsort(-np.abs(vals))[:top_n]
                i, j, vals = i[keep], j[keep], vals[keep]
            edges = [(t[a_], t[b_], float(v)) for a_, b_, v in zip(i, j, vals)]
            info["alpha"] = a
        elif kind == "mst":
            G = mst_graph(C, t)
            edges = [(u, v, d["corr"]) for u, v, d in G.edges(data=True)]
        else:  # threshold
            i, j = np.where(np.triu(C, 1) > tau)
            edges = [(t[a_], t[b_], float(C[a_, b_])) for a_, b_ in zip(i, j)]
        return Network(edges, self._metrics(C, edges, n, info), kind)

    def _metrics(self, C, edges, n, info) -> dict:
        G = nx.Graph()
        G.add_nodes_from(self.tickers)
        for u, v, w in edges:
            G.add_edge(u, v, weight=abs(w), pcorr=w)
        off = ~np.eye(len(C), dtype=bool)
        sec = np.array([self.sectors[x] for x in self.tickers])
        same = np.equal.outer(sec, sec)
        deg = dict(G.degree())
        _, nmi = louvain_vs_sector(G, self.sectors)
        return {
            "edges": G.number_of_edges(),
            "density": nx.density(G),
            "mean_corr": float(C[off].mean()),
            "intra_minus_inter": float(C[off & same].mean() - C[off & ~same].mean()),
            "absorption": absorption_ratio(C),
            "modularity": sector_modularity(G, self.sectors),
            "nmi": nmi,
            "neg_share": float(np.mean([w < 0 for *_, w in edges])) if edges else float("nan"),
            "hub": max(deg, key=deg.get),
            "hub_degree": max(deg.values()),
            "n_eff": n,
            **info,
        }

    @lru_cache(maxsize=2048)
    def window_network(self, end, window, residual, kind, sparsity, alpha, top_n, tau) -> Network:
        C, n = self.window_corr(end, window, residual)
        return self.build(C, n, kind, sparsity, alpha, top_n, tau)

    @lru_cache(maxsize=256)
    def regime_network(self, k, state, residual, kind, sparsity, alpha, top_n, tau) -> Network:
        C, n = self.regime_corr(k, state, residual)
        return self.build(C, n, kind, sparsity, alpha, top_n, tau)

    # ---- time series of cheap metrics (for the timeline)
    @lru_cache(maxsize=8)
    def rolling_corr(self, window: int) -> pd.DataFrame:
        rows = []
        for end in self.frames:
            Cr, _ = self.window_corr(int(end), window, False)
            Cs, _ = self.window_corr(int(end), window, True)
            off = ~np.eye(len(Cr), dtype=bool)
            rows.append((self.dates[end - 1], Cr[off].mean(), Cs[off].mean()))
        return pd.DataFrame(rows, columns=["date", "raw", "residual"]).set_index("date")

    # ---- layouts
    @lru_cache(maxsize=1)
    def sector_layout(self) -> dict:
        """Each GICS sector is a small cluster on a ring; related sectors sit next to each
        other. Positions never move, so within- vs cross-sector edges read at a glance."""
        ring = ["Information Technology", "Communication Services", "Consumer Discretionary",
                "Consumer Staples", "Health Care", "Utilities", "Real Estate", "Financials",
                "Industrials", "Materials", "Energy"]
        pos = {}
        for i, sec in enumerate(ring):
            th = np.pi / 2 - 2 * np.pi * i / len(ring)
            cx, cy = np.cos(th), np.sin(th)
            members = sorted(t for t in self.tickers if self.sectors[t] == sec)
            r = 0.055 * np.sqrt(len(members)) + 0.05
            for j, t in enumerate(members):
                ph = th + np.pi + 2 * np.pi * (j + 0.5) / len(members)
                pos[t] = (cx + r * np.cos(ph), cy + r * np.sin(ph))
        return _normalise(pos)

    @lru_cache(maxsize=1)
    def fixed_layout(self) -> dict:
        """Positions from the full-sample residual partial-correlation network: sector
        clusters sit together and nodes stay put as you move through time."""
        C, _ = _weighted_resid_corr(self.X, self.m, None, True)
        Th, _ = _glasso(C, 0.05)
        P = partial_corr(Th)
        G = nx.Graph()
        G.add_nodes_from(self.tickers)
        i, j = np.where(np.triu(P, 1) > 0.02)
        G.add_weighted_edges_from((self.tickers[a], self.tickers[b], P[a, b]) for a, b in zip(i, j))
        return _normalise(nx.spring_layout(G, weight="weight", seed=7, iterations=300))

    def relayout(self, net: Network) -> dict:
        """Force layout of the current graph, initialised from the fixed layout
        so successive frames move smoothly instead of jumping."""
        G = nx.Graph()
        G.add_nodes_from(self.tickers)
        G.add_weighted_edges_from((u, v, abs(w)) for u, v, w in net.edges)
        if net.kind == "mst":
            pos = nx.kamada_kawai_layout(G)
        else:
            pos = nx.spring_layout(G, pos=self.fixed_layout(), weight="weight",
                                   iterations=60, seed=7)
        return _normalise(pos)
