from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf


def load_prices(tickers: list[str], start: str, end: str | None, data_dir: Path,
                refresh: bool = False) -> pd.DataFrame:
    """Adjusted close prices, cached to parquet."""
    data_dir.mkdir(parents=True, exist_ok=True)
    cache = data_dir / "prices.parquet"
    if cache.exists() and not refresh:
        px = pd.read_parquet(cache)
        have_all = set(tickers) <= set(px.columns)
        covers = px.index[0] <= pd.Timestamp(start) + pd.Timedelta(days=7)
        if have_all and covers:
            return px.loc[start:end, tickers]
    px = yf.download(tickers, start=start, end=end, auto_adjust=True,
                     progress=False, threads=True)["Close"]
    px.index = pd.DatetimeIndex(px.index).tz_localize(None)
    px.to_parquet(cache)
    return px.loc[:, tickers]


def clean_returns(prices: pd.DataFrame, max_missing: float = 0.02):
    """Daily log returns on a common calendar. Drops tickers with too much missing data."""
    miss = prices.isna().mean()
    dropped = miss[miss > max_missing].index.tolist()
    px = prices.drop(columns=dropped).ffill(limit=3).dropna()
    rets = np.log(px).diff().dropna()
    return rets, dropped
