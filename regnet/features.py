import numpy as np
import pandas as pd


def market_features(stock_rets: pd.DataFrame, mkt_ret: pd.Series, halflife: int = 10) -> pd.DataFrame:
    """Low-lag daily features for the regime HMM (Model B).

    - mkt_ret:  market log return (%)
    - log_vol:  log annualised EWMA volatility of the market (short half-life -> small lag)
    - log_disp: log cross-sectional dispersion of stock returns (same-day, no window)
    """
    vol = np.sqrt((mkt_ret ** 2).ewm(halflife=halflife, adjust=False).mean() * 252)
    disp = stock_rets.std(axis=1)
    f = pd.DataFrame({
        "mkt_ret": mkt_ret * 100,
        "log_vol": np.log(vol),
        "log_disp": np.log(disp),
    })
    return f.iloc[3 * halflife:].dropna()
