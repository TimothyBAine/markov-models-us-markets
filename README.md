# Regime-switching stock networks (HMM × network science)

How does the dependence network of US large caps rewire across market regimes?
This project couples hidden Markov models with network science in the two ways
sketched in the design notes, and compares them.

| | **Model A: network-as-observation** | **Model B: regime-conditional graphs** (primary) |
|---|---|---|
| HMM input | Rolling-window (63d, sampled weekly) network topology: mean correlation, normalised MST length, sector modularity of the threshold graph | Low-lag daily market features: SPY return, log EWMA vol (half-life 10d), log cross-sectional dispersion |
| Networks | Implicit in the features | One network per regime from **posterior-weighted** correlation → MST + graphical-lasso partial-correlation graph (alpha by EBIC) |
| Question answered | "Does network structure itself define regimes?" | "Given a regime, what does the dependence network look like?" |
| Weakness | Window lag + overlapping-window autocorrelation in o_t | Needs enough effective observations per regime (n_eff reported; warns if < 5·N) |

Design choices worth knowing:

- **Soft weights.** Regime networks use smoothed posteriors γ_t(k) as observation
  weights, so days near a regime boundary count partially toward each regime instead of being hard-assigned.
  The Kish effective sample size `n_eff = (Σγ)² / Σγ²` is reported per regime.
- **Two graph views.** The *MST* of the Mantegna distance `√(2(1−ρ))` shows the
  unconditional backbone. The *partial-correlation* graph (glasso) shows conditional
  dependence after controlling for every other stock, which mostly removes the market
  mode that dominates raw correlations in stress.
- **Stable labels.** HMM states are relabelled by mean volatility (Model B) or mean
  correlation (Model A), so "Calm/Moderate/Stress" mean the same thing on every run.
  The best of 10 random restarts (by log-likelihood) is kept.

## Run

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m regnet                    # defaults: 2005→today, 3 regimes, ~1 min
.venv/bin/python -m regnet --states 4         # more regimes
.venv/bin/python -m regnet --residualize      # strip market beta first: sector structure only
.venv/bin/python -m regnet --alpha 0.05       # fixed glasso penalty (comparable density across regimes)
.venv/bin/python -m regnet --refresh          # re-download prices (cached in data/)
```

## Outputs (`outputs/`)

| File | What |
|---|---|
| `figures/01_regime_timeline.png` | SPY with Model B regimes, posterior probabilities, Model A regimes over rolling correlation |
| `figures/02_regime_networks.png` | Per-regime MSTs (top) and partial-correlation networks on a fixed layout (bottom) |
| `figures/03_regime_correlations.png` | Sector-ordered correlation matrices per regime, plus Stress − Calm |
| `figures/04_regime_metrics.png` | Topology metrics by regime |
| `figures/05_model_comparison.png` | Agreement between Model A and Model B states |
| `regime_networks.html` | Interactive networks with hover and a dropdown over regimes × {partial corr, MST} |
| `tables/*.csv` | Regime summaries, episodes, posteriors, BIC grid, per-regime correlation / partial-correlation matrices, hubs |
| `summary.json` | Transition matrices, emission means, ARI, current regime |

## First results (2005-02 → 2026-11, 66 stocks, K = 3)

- **Regimes:** Calm (38% of days, 8.5% annualised vol), Moderate (35%, 13.8%) and Stress (27%, 31%).
  All three are persistent, with expected durations of 43, 29 and 67 days. Stress almost never jumps
  straight to Calm (P ≈ 0.000); it unwinds through Moderate.
- **Network contraction under stress:** mean correlation goes 0.18 → 0.26 → 0.52, the absorption
  ratio goes 0.20 → 0.53, and the MST length shrinks 1.03 → 0.73. The tree collapses from a long
  sector chain into a star around industrials/tech hubs (HON, CSCO).
- **Sector structure mostly survives:** the intra-minus-inter sector correlation gap only narrows
  (0.21 → 0.15). Partial-correlation sector modularity barely moves (0.50 → 0.47), and Louvain
  communities still largely match GICS (NMI 0.82 in both Calm and Stress). The stress "coupling" is mainly a common
  market factor layered on top of an intact sector skeleton.
- **The two couplings only partly agree** (ARI ≈ 0.2). 80% of Model B's Stress days fall in Model A's
  High-coupling state, but Calm and Moderate spread across all three, and high correlation can persist
  after volatility subsides. Model A's 63-day window also lags: on 2026-11-25 Model B says Stress
  while Model A still says Low coupling.
- **BIC keeps falling as K grows** (K = 5 is best on BIC). With 5,000+ daily observations this is
  expected, because fat tails get absorbed as extra states. K = 3 is chosen for interpretability; try `--states 4`.

### Market-residualized networks (`--residualize`)

Each stock's full-sample SPY beta is removed before building the networks; the regimes stay the same.

- **The stress "contraction" is almost entirely the market factor.** Mean residual correlation is
  about 0.04–0.06 in every regime (raw: 0.18 → 0.52). MST length stays flat (1.12 → 1.06, raw: 1.03 → 0.73)
  and the HON star disappears.
- **Beyond the market, sectors become more distinct in stress.** The intra-minus-inter gap widens
  (0.24 → 0.28), whereas in the raw data it narrows. Tech becomes more internally coupled (+0.10),
  and Staples and Health Care move together as a defensive block.
- **Real Estate switches clusters.** Its residual correlation with Financials rises by +0.29 and its
  correlation with Utilities falls by −0.26. The Financials link holds outside the 2007–09 financial
  crisis (−0.01 → 0.19), and in the COVID and 2022 stress spells too. The Utilities decoupling is
  GFC-only: during the GFC it is −0.13, outside it 0.32 → 0.24.
- Robustness: re-residualizing with posterior-weighted betas estimated per regime gives the same
  block changes to ±0.03. The regime-specific shift in betas is therefore not driving the pattern.

## Dashboard

```bash
.venv/bin/python -m regnet.dashboard        # http://127.0.0.1:8050 (first load ~20 s while the HMM fits)
```

- **Through time:** pick a date on the slider (or click the SPY timeline), or press Play to step through
  rolling-window networks. The timeline shades the regimes, and each network's border takes the colour
  of the regime at the window's end. The status line shows the regime mix inside the window.
- **By regime:** one posterior-weighted network per regime, plus a metrics table.
- **Market factor:** raw returns, market removed (betas estimated locally per window, or weighted by
  posterior per regime), or both side by side.
- **Graph:** partial correlation (glasso), MST, or correlation threshold τ.
- **Sparsity:** Auto (EBIC), fixed penalty α, or fixed edge count (the default). A fixed edge count is
  the right choice for comparing frames, because it holds density constant, so what changes is *which*
  stocks connect. EBIC refits several penalties per frame and plays back slowly.
- **Layout:** Sector ring (default: each GICS sector is a cluster on a ring, so within-sector vs
  cross-sector edges read at a glance), Force (a fixed force layout from the full-sample network), or
  Adaptive (a force layout recomputed per frame, seeded from the fixed positions so clusters drift
  smoothly). In both fixed layouts, nodes stay put and edges visibly appear and vanish.
- Fixed-edge-count mode fits glasso at α = 0.03 and keeps the N strongest partial correlations.
- **Explainers:** every control and metric chip has a tooltip (hover or Tab to focus), and a collapsed
  "How to read this dashboard" panel sits under the title. The text lives in `regnet/dashboard/text.py`.
- **Regimes K and window length** are adjustable. HMM fits and networks are cached, so revisiting
  a frame is instant.

## Caveats

- **Look-ahead:** smoothed posteriors and the Viterbi path use the full sample. They are fine
  for describing history but not for trading. A real-time version needs filtered probabilities and
  an expanding-window refit.
- **Survivorship bias:** the universe is today's large caps that have traded since 2005.
- Raw correlations in stress are dominated by the market mode. Use `--residualize` or the
  partial-correlation view to see structure beyond it.
- EBIC picks fairly dense glasso graphs because n_eff ≈ 1,500–2,100. The figures draw only edges with
  |partial ρ| ≥ 0.08, while the metrics use the full graph. Glasso density varies with n_eff as well
  as with structure, so pass a fixed `--alpha` for like-for-like density comparisons.

## Possible extensions

- Joint emissions: an HMM on the top-k PCA factor returns with full covariance, so the
  network enters the likelihood directly (a true Markov-switching graphical model).
- Bootstrap edge stability per regime (block bootstrap within regime episodes).
- A sticky HDP-HMM to learn K instead of fixing it.
- Drive the dashboard from filtered (real-time) regime probabilities instead of smoothed ones.

## Layout

```
regnet/
  universe.py   tickers, GICS sectors, colour groups
  data.py       yfinance download + parquet cache, log returns
  features.py   Model B market features
  hmm.py        Gaussian HMM (restarts, BIC, relabelling, regime summaries)
  networks.py   weighted corr, glasso+EBIC, MST, modularity, Louvain, rolling features
  viz.py        matplotlib figures + plotly interactive network
  pipeline.py   end-to-end run
  dashboard/    Dash app: compute.py (cached networks), figures.py, app.py (layout + callbacks)
```
