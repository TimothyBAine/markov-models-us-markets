"""Gaussian HMM fitting with restarts, BIC selection and volatility-ordered state labels."""
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

logging.getLogger("hmmlearn").setLevel(logging.ERROR)

STATE_NAMES = {
    2: ["Calm", "Stress"],
    3: ["Calm", "Moderate", "Stress"],
    4: ["Calm", "Moderate", "Elevated", "Stress"],
    5: ["Calm", "Low-moderate", "Moderate", "Elevated", "Stress"],
}


def state_names(k: int) -> list[str]:
    return STATE_NAMES.get(k, [f"Regime {i}" for i in range(k)])


@dataclass
class RegimeModel:
    model: GaussianHMM
    feature_names: list[str]
    states: pd.Series           # Viterbi path, relabelled 0..K-1 in order of `order_by`
    posteriors: pd.DataFrame    # smoothed P(s_t = k | all data), same labels
    transmat: np.ndarray        # relabelled transition matrix
    means: pd.DataFrame         # emission means in original feature units
    loglik: float
    bic: float
    names: list[str]

    @property
    def n_states(self) -> int:
        return self.transmat.shape[0]

    @property
    def expected_duration(self) -> np.ndarray:
        return 1.0 / (1.0 - np.diag(self.transmat))


def _n_params(k: int, d: int) -> int:
    return (k - 1) + k * (k - 1) + k * d + k * d * (d + 1) // 2


def _fit_best(Z: np.ndarray, k: int, n_restarts: int, seed: int, n_iter: int):
    best, best_ll = None, -np.inf
    for r in range(n_restarts):
        m = GaussianHMM(n_components=k, covariance_type="full", n_iter=n_iter,
                        tol=1e-4, random_state=seed + r, min_covar=1e-3)
        try:
            m.fit(Z)
            ll = m.score(Z)
        except (ValueError, np.linalg.LinAlgError):
            continue
        if np.isfinite(ll) and ll > best_ll:
            best, best_ll = m, ll
    if best is None:
        raise RuntimeError(f"All {n_restarts} HMM restarts failed for K={k}")
    return best, best_ll


def fit_hmm(X: pd.DataFrame, n_states: int, order_by: str, n_restarts: int = 10,
            seed: int = 0, n_iter: int = 500) -> RegimeModel:
    """Fit a full-covariance Gaussian HMM on standardised features.

    States are relabelled so that state 0 has the lowest mean of `order_by`
    (e.g. volatility), making labels stable and interpretable across runs.
    """
    mu, sd = X.mean(), X.std()
    Z = ((X - mu) / sd).to_numpy()
    m, ll = _fit_best(Z, n_states, n_restarts, seed, n_iter)

    j = X.columns.get_loc(order_by)
    perm = np.argsort(m.means_[:, j])           # new label k  <- old label perm[k]
    inv = np.empty_like(perm)
    inv[perm] = np.arange(n_states)

    post = m.predict_proba(Z)[:, perm]
    states = inv[m.predict(Z)]
    means = pd.DataFrame(m.means_[perm] * sd.to_numpy() + mu.to_numpy(), columns=X.columns)

    return RegimeModel(
        model=m,
        feature_names=list(X.columns),
        states=pd.Series(states, index=X.index, name="state"),
        posteriors=pd.DataFrame(post, index=X.index, columns=range(n_states)),
        transmat=m.transmat_[np.ix_(perm, perm)],
        means=means,
        loglik=ll,
        bic=-2 * ll + _n_params(n_states, X.shape[1]) * np.log(len(X)),
        names=state_names(n_states),
    )


def select_k(X: pd.DataFrame, ks, n_restarts: int = 5, seed: int = 0) -> pd.DataFrame:
    Z = ((X - X.mean()) / X.std()).to_numpy()
    rows = []
    for k in ks:
        _, ll = _fit_best(Z, k, n_restarts, seed, 500)
        p = _n_params(k, Z.shape[1])
        rows.append({"K": k, "loglik": ll, "n_params": p,
                     "bic": -2 * ll + p * np.log(len(Z)),
                     "aic": -2 * ll + 2 * p})
    return pd.DataFrame(rows)


def _runs(states: pd.Series) -> pd.DataFrame:
    change = states.ne(states.shift()).cumsum()
    g = states.groupby(change)
    return pd.DataFrame({"state": g.first(), "start": g.apply(lambda s: s.index[0]),
                         "end": g.apply(lambda s: s.index[-1]), "length": g.size()})


def describe_regimes(rm: RegimeModel, mkt_ret: pd.Series, stock_rets: pd.DataFrame) -> pd.DataFrame:
    """Per-regime summary: frequency, persistence and market behaviour."""
    runs = _runs(rm.states)
    r = mkt_ret.reindex(rm.states.index)
    disp = stock_rets.reindex(rm.states.index).std(axis=1)
    rows = []
    for k in range(rm.n_states):
        mask = rm.states == k
        kr = runs[runs.state == k]
        rows.append({
            "state": k,
            "name": rm.names[k],
            "share_of_days": mask.mean(),
            "n_episodes": len(kr),
            "mean_episode_days": kr.length.mean(),
            "expected_duration_days": rm.expected_duration[k],
            "self_transition_prob": rm.transmat[k, k],
            "mkt_ann_return": r[mask].mean() * 252,
            "mkt_ann_vol": r[mask].std() * np.sqrt(252),
            "mean_dispersion": disp[mask].mean(),
        })
    return pd.DataFrame(rows).set_index("state")


def regime_runs(rm: RegimeModel) -> pd.DataFrame:
    runs = _runs(rm.states)
    runs["name"] = runs.state.map(dict(enumerate(rm.names)))
    return runs.reset_index(drop=True)
