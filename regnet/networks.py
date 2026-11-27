"""Network construction and topology metrics.

Two graph types per correlation structure:
- MST of the Mantegna distance d_ij = sqrt(2(1 - rho_ij)): a backbone of the
  strongest (unconditional) co-movements.
- Partial-correlation graph from a graphical-lasso precision matrix: an edge
  means conditional dependence given all other stocks (removes the common
  market mode that dominates raw correlations in stress).
"""
import warnings
from dataclasses import dataclass, field

import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.covariance import graphical_lasso
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import normalized_mutual_info_score

# ---------------------------------------------------------------- matrices


def corr_from_cov(S: np.ndarray) -> np.ndarray:
    d = np.sqrt(np.diag(S))
    C = S / np.outer(d, d)
    np.fill_diagonal(C, 1.0)
    return C


def weighted_corr(X: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, float]:
    """Correlation of rows of X weighted by w (e.g. HMM posteriors).

    Returns (C, n_eff) where n_eff = (sum w)^2 / sum w^2 is the Kish effective
    sample size, which governs how well-conditioned the estimate is.
    """
    w = np.asarray(w, float)
    w = w / w.sum()
    Xc = X - w @ X
    S = (Xc * w[:, None]).T @ Xc
    return corr_from_cov(S), 1.0 / np.sum(w ** 2)


def residualize(rets: pd.DataFrame, mkt: pd.Series) -> pd.DataFrame:
    """Remove each stock's full-sample market beta exposure."""
    m = mkt.reindex(rets.index)
    mc = m - m.mean()
    beta = rets.sub(rets.mean()).mul(mc, axis=0).sum() / (mc ** 2).sum()
    return rets - np.outer(m, beta)


def mantegna_distance(C: np.ndarray) -> np.ndarray:
    return np.sqrt(np.clip(2.0 * (1.0 - C), 0.0, None))


def absorption_ratio(C: np.ndarray, k: int = 1) -> float:
    ev = np.linalg.eigvalsh(C)
    return float(ev[-k:].sum() / ev.sum())


def mean_offdiag(C: np.ndarray, mask: np.ndarray | None = None) -> float:
    off = ~np.eye(len(C), dtype=bool)
    if mask is not None:
        off &= mask
    return float(C[off].mean())


def glasso_ebic(C: np.ndarray, n: float, alphas=None, gamma: float = 0.5):
    """Graphical lasso on a correlation matrix with alpha chosen by extended BIC
    (Foygel & Drton 2010). Returns (precision, alpha, path DataFrame)."""
    p = C.shape[0]
    if alphas is None:
        alphas = np.logspace(-2.3, -0.5, 16)
    best, path = None, []
    for a in alphas:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                warnings.simplefilter("ignore", RuntimeWarning)   # diverging small-alpha fits
                with np.errstate(all="ignore"):
                    _, Th = graphical_lasso(C, alpha=a, max_iter=300, tol=1e-4)
        except (FloatingPointError, ValueError):
            continue
        if not np.all(np.isfinite(Th)):
            continue                      # diverged at this (too small) alpha
        with np.errstate(all="ignore"):
            sign, logdet = np.linalg.slogdet(Th)
        if sign <= 0 or not np.isfinite(logdet):
            continue
        ll = 0.5 * n * (logdet - np.trace(C @ Th))
        e = int((np.abs(np.triu(Th, 1)) > 1e-6).sum())
        ebic = -2 * ll + e * np.log(n) + 4 * gamma * e * np.log(p)
        path.append({"alpha": a, "edges": e, "ebic": ebic})
        if best is None or ebic < best[2]:
            best = (Th, a, ebic)
    if best is None:
        raise RuntimeError("graphical lasso failed for every alpha")
    return best[0], best[1], pd.DataFrame(path)


def partial_corr(Th: np.ndarray) -> np.ndarray:
    d = np.sqrt(np.diag(Th))
    P = -Th / np.outer(d, d)
    np.fill_diagonal(P, 0.0)
    return P


# ---------------------------------------------------------------- graphs


def mst_graph(C: np.ndarray, tickers: list[str]) -> nx.Graph:
    D = mantegna_distance(C)
    T = minimum_spanning_tree(D + 1e-12).tocoo()   # tiny offset: csgraph treats 0 as no-edge
    G = nx.Graph()
    G.add_nodes_from(tickers)
    for i, j, d in zip(T.row, T.col, T.data):
        G.add_edge(tickers[i], tickers[j], weight=float(d), corr=float(C[i, j]))
    return G


def mst_length(C: np.ndarray) -> float:
    """Normalised tree length (Onnela et al. 2003): mean MST edge distance.
    Shrinks when the market contracts into a tight, highly correlated cluster."""
    T = minimum_spanning_tree(mantegna_distance(C) + 1e-12)
    return float(T.sum() / (C.shape[0] - 1))


def threshold_graph(C: np.ndarray, tickers: list[str], tau: float) -> nx.Graph:
    G = nx.Graph()
    G.add_nodes_from(tickers)
    i, j = np.where(np.triu(C, 1) > tau)
    G.add_weighted_edges_from((tickers[a], tickers[b], float(C[a, b])) for a, b in zip(i, j))
    return G


def pcorr_graph(P: np.ndarray, tickers: list[str], eps: float = 1e-6) -> nx.Graph:
    G = nx.Graph()
    G.add_nodes_from(tickers)
    i, j = np.where(np.abs(np.triu(P, 1)) > eps)
    for a, b in zip(i, j):
        G.add_edge(tickers[a], tickers[b], weight=abs(float(P[a, b])), pcorr=float(P[a, b]))
    return G


def _positive_subgraph(G: nx.Graph) -> nx.Graph:
    H = nx.Graph()
    H.add_nodes_from(G.nodes)
    H.add_edges_from((u, v, d) for u, v, d in G.edges(data=True) if d.get("pcorr", 1.0) > 0)
    return H


def sector_modularity(G: nx.Graph, sectors: dict[str, str]) -> float:
    """Newman modularity of the GICS sector partition (positive edges only)."""
    H = _positive_subgraph(G)
    if H.number_of_edges() == 0:
        return float("nan")
    groups: dict[str, set] = {}
    for n in H.nodes:
        groups.setdefault(sectors[n], set()).add(n)
    return float(nx.community.modularity(H, groups.values(), weight="weight"))


def louvain_vs_sector(G: nx.Graph, sectors: dict[str, str], seed: int = 0) -> tuple[int, float]:
    """Number of Louvain communities and their NMI with GICS sectors."""
    H = _positive_subgraph(G)
    if H.number_of_edges() == 0:
        return 0, float("nan")
    comms = nx.community.louvain_communities(H, weight="weight", seed=seed)
    lab = {n: i for i, c in enumerate(comms) for n in c}
    nodes = list(H.nodes)
    nmi = normalized_mutual_info_score([sectors[n] for n in nodes], [lab[n] for n in nodes])
    return len(comms), float(nmi)


# ---------------------------------------------------------------- Model A: rolling features


def rolling_network_features(rets: pd.DataFrame, sectors: dict[str, str], window: int = 63,
                             step: int = 5, tau: float = 0.5) -> pd.DataFrame:
    """Topology of the rolling-window correlation network, sampled every `step` days.
    Each row is stamped with the *last* date of its window (no look-ahead)."""
    X = rets.to_numpy()
    tickers = list(rets.columns)
    same = np.equal.outer(np.array([sectors[t] for t in tickers]),
                          np.array([sectors[t] for t in tickers]))
    rows = []
    for end in range(window, len(X) + 1, step):
        C = np.corrcoef(X[end - window:end], rowvar=False)
        G = threshold_graph(C, tickers, tau)
        rows.append({
            "date": rets.index[end - 1],
            "mean_corr": mean_offdiag(C),
            "absorption": absorption_ratio(C),
            "mst_length": mst_length(C),
            "intra_minus_inter": mean_offdiag(C, same) - mean_offdiag(C, ~same),
            "thr_density": nx.density(G),
            "thr_modularity": sector_modularity(G, sectors),
        })
    df = pd.DataFrame(rows).set_index("date")
    return df


# ---------------------------------------------------------------- Model B: regime networks


@dataclass
class RegimeNetwork:
    state: int
    name: str
    C: np.ndarray
    P: np.ndarray
    n_eff: float
    alpha: float
    mst: nx.Graph
    pgraph: nx.Graph
    metrics: dict
    hubs: pd.DataFrame
    ebic_path: pd.DataFrame = field(default_factory=pd.DataFrame)


def network_metrics(C: np.ndarray, P: np.ndarray, tickers: list[str],
                    sectors: dict[str, str], tau: float) -> tuple[dict, nx.Graph, nx.Graph, pd.DataFrame]:
    sec = np.array([sectors[t] for t in tickers])
    same = np.equal.outer(sec, sec)
    mst = mst_graph(C, tickers)
    pg = pcorr_graph(P, tickers)
    thr = threshold_graph(C, tickers, tau)
    n_comm, nmi = louvain_vs_sector(pg, sectors)
    pcs = np.array([d["pcorr"] for *_, d in pg.edges(data=True)])
    intra = mean_offdiag(C, same)
    inter = mean_offdiag(C, ~same)

    strength = pd.Series(dict(pg.degree(weight="weight")))
    mst_deg = pd.Series(dict(mst.degree()))
    hubs = pd.DataFrame({"pcorr_strength": strength, "pcorr_degree": pd.Series(dict(pg.degree())),
                         "mst_degree": mst_deg, "sector": pd.Series(sectors)}).loc[tickers]
    hubs = hubs.sort_values("pcorr_strength", ascending=False)

    metrics = {
        "mean_corr": mean_offdiag(C),
        "absorption": absorption_ratio(C),
        "mst_length": mst_length(C),
        "intra_sector_corr": intra,
        "inter_sector_corr": inter,
        "intra_minus_inter": intra - inter,
        "thr_density": nx.density(thr),
        "thr_modularity": sector_modularity(thr, sectors),
        "pcorr_edges": pg.number_of_edges(),
        "pcorr_density": nx.density(pg),
        "pcorr_mean_abs": float(np.abs(pcs).mean()) if len(pcs) else float("nan"),
        "pcorr_neg_share": float((pcs < 0).mean()) if len(pcs) else float("nan"),
        "pcorr_modularity": sector_modularity(pg, sectors),
        "louvain_communities": n_comm,
        "louvain_nmi": nmi,
        "mst_hub": mst_deg.idxmax(),
        "mst_hub_degree": int(mst_deg.max()),
    }
    return metrics, mst, pg, hubs


def estimate_network(rets: pd.DataFrame, w: np.ndarray, sectors: dict[str, str], tau: float,
                     alpha: float | None = None, gamma: float = 0.5, state: int = -1,
                     name: str = "Full sample") -> RegimeNetwork:
    tickers = list(rets.columns)
    C, n_eff = weighted_corr(rets.to_numpy(), w)
    if n_eff < 5 * len(tickers):
        warnings.warn(f"{name}: effective sample size {n_eff:.0f} is < 5x the number of "
                      f"stocks ({len(tickers)}); the network estimate will be noisy.")
    if alpha is None:
        Th, a, path = glasso_ebic(C, n_eff, gamma=gamma)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            _, Th = graphical_lasso(C, alpha=alpha, max_iter=300)
        a, path = alpha, pd.DataFrame()
    P = partial_corr(Th)
    metrics, mst, pg, hubs = network_metrics(C, P, tickers, sectors, tau)
    metrics = {"n_eff": n_eff, "glasso_alpha": a, **metrics}
    return RegimeNetwork(state, name, C, P, n_eff, a, mst, pg, metrics, hubs, path)


def estimate_regime_networks(rets: pd.DataFrame, posteriors: pd.DataFrame, names: list[str],
                             sectors: dict[str, str], tau: float, alpha: float | None = None,
                             gamma: float = 0.5) -> list[RegimeNetwork]:
    """One network per regime, using smoothed posteriors as soft observation weights.

    Soft weights use every day for every regime (down-weighted by membership
    probability) instead of hard-assigning days, which keeps covariance
    estimates better conditioned near regime boundaries."""
    idx = rets.index.intersection(posteriors.index)
    R, W = rets.loc[idx], posteriors.loc[idx]
    return [estimate_network(R, W[k].to_numpy(), sectors, tau, alpha, gamma, k, names[k])
            for k in W.columns]
