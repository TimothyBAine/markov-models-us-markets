"""End-to-end run: data -> two HMM couplings -> regime networks -> figures + tables."""
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from . import viz
from .config import Config
from .data import clean_returns, load_prices
from .features import market_features
from .hmm import describe_regimes, fit_hmm, regime_runs, select_k
from .networks import (estimate_network, estimate_regime_networks, residualize,
                       rolling_network_features)
from .universe import MARKET, TICKER_SECTOR, TICKERS

MODEL_A_FEATURES = ["mean_corr", "mst_length", "thr_modularity"]


def compare_models(rm_a, rm_b, window: int) -> dict:
    """Agreement between Model A (rolling topology) and Model B (market features).

    Model A's state at date t summarises the window (t-window, t], so we compare it
    both with Model B's state *at* t and with B's majority state over that window."""
    a = rm_a.states
    b_point = rm_b.states.reindex(a.index, method="ffill")
    b_major = (rm_b.states.rolling(window, min_periods=window // 2)
               .apply(lambda s: np.bincount(s.astype(int)).argmax(), raw=True)
               .reindex(a.index, method="ffill"))
    ok = b_point.notna() & b_major.notna()
    a, b_point, b_major = a[ok], b_point[ok].astype(int), b_major[ok].astype(int)
    ct = pd.crosstab(b_point.map(dict(enumerate(rm_b.names))),
                     a.map(lambda k: f"A{k}: {rm_a.names[k]}"), normalize="index")
    ct = ct.reindex(index=[n for n in rm_b.names if n in ct.index])
    return {"ari_point": adjusted_rand_score(b_point, a),
            "ari_window": adjusted_rand_score(b_major, a),
            "crosstab": ct}


def run(cfg: Config) -> dict:
    out = cfg.out_dir
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "tables").mkdir(parents=True, exist_ok=True)

    # ---- data
    px = load_prices(TICKERS + [MARKET], cfg.start, cfg.end, cfg.data_dir, cfg.refresh)
    rets, dropped = clean_returns(px)
    mkt = rets.pop(MARKET)
    tickers = list(rets.columns)
    sectors = {t: TICKER_SECTOR[t] for t in tickers}
    mkt_px = px[MARKET].ffill()
    print(f"Data: {len(tickers)} stocks, {rets.index[0].date()} -> {rets.index[-1].date()} "
          f"({len(rets)} days). Dropped: {dropped or 'none'}")

    # ---- Model B: HMM on low-lag market features -> regime-conditional networks
    f_b = market_features(rets, mkt, cfg.vol_halflife)
    bic = select_k(f_b, cfg.k_grid, n_restarts=max(3, cfg.n_restarts // 2), seed=cfg.seed)
    bic.to_csv(out / "tables" / "model_b_k_selection.csv", index=False)
    rm_b = fit_hmm(f_b, cfg.n_states, order_by="log_vol", n_restarts=cfg.n_restarts, seed=cfg.seed)
    summary_b = describe_regimes(rm_b, mkt, rets)
    summary_b.to_csv(out / "tables" / "model_b_regimes.csv")
    regime_runs(rm_b).to_csv(out / "tables" / "model_b_episodes.csv", index=False)
    rm_b.posteriors.set_axis(rm_b.names, axis=1).assign(viterbi=rm_b.states).to_csv(
        out / "tables" / "model_b_posteriors.csv")

    net_rets = residualize(rets, mkt) if cfg.residualize else rets
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", UserWarning)
        regnets = estimate_regime_networks(net_rets, rm_b.posteriors, rm_b.names, sectors,
                                           cfg.corr_threshold, cfg.glasso_alpha, cfg.ebic_gamma)
        full = estimate_network(net_rets.loc[rm_b.posteriors.index],
                                np.ones(len(rm_b.posteriors)), sectors, cfg.corr_threshold,
                                cfg.glasso_alpha, cfg.ebic_gamma)
    for w in caught:
        if issubclass(w.category, UserWarning):
            print(f"  warning: {w.message}")

    metrics = pd.DataFrame([{"state": rn.state, "name": rn.name, **rn.metrics} for rn in regnets]
                           + [{"state": -1, "name": "Full sample", **full.metrics}]).set_index("state")
    metrics.to_csv(out / "tables" / "regime_network_metrics.csv")
    pd.concat({rn.name: rn.hubs.head(10) for rn in regnets}).to_csv(out / "tables" / "regime_hubs.csv")
    for rn in regnets:
        pd.DataFrame(rn.C, index=tickers, columns=tickers).to_csv(
            out / "tables" / f"corr_{rn.name.lower().replace(' ', '_')}.csv")
        pd.DataFrame(rn.P, index=tickers, columns=tickers).to_csv(
            out / "tables" / f"pcorr_{rn.name.lower().replace(' ', '_')}.csv")

    # ---- Model A: HMM on rolling network topology
    f_a_all = rolling_network_features(rets, sectors, cfg.roll_window, cfg.roll_step, cfg.corr_threshold)
    f_a_all.to_csv(out / "tables" / "model_a_rolling_features.csv")
    rm_a = fit_hmm(f_a_all[MODEL_A_FEATURES], cfg.n_states, order_by="mean_corr",
                   n_restarts=cfg.n_restarts, seed=cfg.seed)
    rm_a.names = [f"{n} coupling" for n in
                  (["Low", "High"] if cfg.n_states == 2 else
                   ["Low", "Mid", "High"] if cfg.n_states == 3 else
                   [f"Level {i}" for i in range(cfg.n_states)])]
    means_a = rm_a.means.assign(name=rm_a.names, expected_duration_weeks=rm_a.expected_duration
                                * cfg.roll_step / 5, share=rm_a.states.value_counts(normalize=True)
                                .sort_index().to_numpy())
    means_a.to_csv(out / "tables" / "model_a_regimes.csv")

    cmp = compare_models(rm_a, rm_b, cfg.roll_window)
    cmp["crosstab"].to_csv(out / "tables" / "model_comparison_crosstab.csv")

    # ---- figures
    fig_dir = out / "figures"
    viz.plot_timeline(mkt_px, rm_b, f_a_all, rm_a, fig_dir / "01_regime_timeline.png")
    viz.plot_regime_networks(regnets, full, sectors, fig_dir / "02_regime_networks.png")
    viz.plot_corr_heatmaps(regnets, tickers, sectors, fig_dir / "03_regime_correlations.png")
    viz.plot_metrics(metrics.loc[metrics.index >= 0], fig_dir / "04_regime_metrics.png")
    viz.plot_model_comparison(cmp["crosstab"], cmp["ari_point"], cmp["ari_window"],
                              fig_dir / "05_model_comparison.png")
    viz.interactive_networks(regnets, full, sectors, out / "regime_networks.html")

    # ---- summary
    summary = {
        "config": {k: str(v) for k, v in vars(cfg).items()},
        "n_stocks": len(tickers), "dropped": dropped,
        "sample": [str(rets.index[0].date()), str(rets.index[-1].date())],
        "model_b": {
            "features": rm_b.feature_names, "loglik": rm_b.loglik, "bic": rm_b.bic,
            "names": rm_b.names, "transmat": rm_b.transmat.round(4).tolist(),
            "emission_means": rm_b.means.round(4).to_dict(orient="index"),
        },
        "model_a": {"features": MODEL_A_FEATURES, "names": rm_a.names,
                    "transmat": rm_a.transmat.round(4).tolist()},
        "comparison": {"ari_point": cmp["ari_point"], "ari_window": cmp["ari_window"]},
        "current_regime": {"date": str(rm_b.states.index[-1].date()),
                           "model_b": rm_b.names[int(rm_b.states.iloc[-1])],
                           "model_b_probs": dict(zip(rm_b.names, rm_b.posteriors.iloc[-1].round(3))),
                           "model_a": rm_a.names[int(rm_a.states.iloc[-1])]},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))

    pd.set_option("display.width", 160, "display.precision", 3)
    print("\nModel B K-selection (lower BIC is better):\n", bic.round(1).to_string(index=False))
    print("\nModel B regimes:\n", summary_b.to_string())
    print("\nTransition matrix (rows = from):\n",
          pd.DataFrame(rm_b.transmat, index=rm_b.names, columns=rm_b.names).round(4).to_string())
    show = ["n_eff", "glasso_alpha", "mean_corr", "absorption", "mst_length", "intra_minus_inter",
            "pcorr_edges", "pcorr_modularity", "louvain_communities", "louvain_nmi", "mst_hub"]
    print("\nRegime network metrics:\n", metrics.set_index("name")[show].to_string())
    print("\nModel A regimes (emission means):\n", means_a.to_string())
    print(f"\nA vs B agreement: ARI point={cmp['ari_point']:.3f}, window-majority={cmp['ari_window']:.3f}")
    print(cmp["crosstab"].round(2).to_string())
    print(f"\nCurrent regime ({summary['current_regime']['date']}): "
          f"B={summary['current_regime']['model_b']} {summary['current_regime']['model_b_probs']}, "
          f"A={summary['current_regime']['model_a']}")
    print(f"\nOutputs written to {out.resolve()}")
    return summary
