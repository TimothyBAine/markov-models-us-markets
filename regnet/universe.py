"""Stock universe: large-cap US names with continuous history since 2005, by GICS sector.

Note: this is a *current* large-cap list, so the sample carries survivorship bias
(firms that failed or shrank out of the large-cap set are absent).
"""

MARKET = "SPY"

UNIVERSE: dict[str, list[str]] = {
    "Information Technology": ["AAPL", "MSFT", "INTC", "CSCO", "ORCL", "IBM", "TXN", "QCOM"],
    "Communication Services": ["GOOGL", "VZ", "T", "DIS", "CMCSA"],
    "Consumer Discretionary": ["AMZN", "HD", "MCD", "NKE", "LOW", "SBUX"],
    "Consumer Staples": ["PG", "KO", "PEP", "WMT", "COST", "CL"],
    "Health Care": ["JNJ", "PFE", "MRK", "UNH", "ABT", "AMGN", "MDT"],
    "Financials": ["JPM", "BAC", "WFC", "C", "GS", "AXP", "USB"],
    "Industrials": ["GE", "CAT", "HON", "MMM", "UPS", "BA", "LMT"],
    "Energy": ["XOM", "CVX", "COP", "SLB", "OXY"],
    "Materials": ["APD", "SHW", "ECL", "NEM", "FCX"],
    "Utilities": ["NEE", "DUK", "SO", "D", "AEP"],
    "Real Estate": ["SPG", "PLD", "PSA", "O", "WELL"],   # EQR/AVB: no Yahoo history after Aug 2026
}

SECTOR_ORDER = list(UNIVERSE)
TICKER_SECTOR = {t: s for s, ts in UNIVERSE.items() for t in ts}
TICKERS = [t for ts in UNIVERSE.values() for t in ts]

# Colour groups: 11 sectors exceed a CVD-safe categorical palette (8 hues), so
# sectors fold into 8 economically coherent groups for colour only. All analysis
# (modularity, NMI) uses the full GICS sector labels.
SECTOR_GROUP = {
    "Information Technology": "Tech & Comm",
    "Communication Services": "Tech & Comm",
    "Financials": "Financials",
    "Health Care": "Health Care",
    "Energy": "Energy",
    "Consumer Staples": "Staples",
    "Consumer Discretionary": "Discretionary",
    "Industrials": "Industrials & Materials",
    "Materials": "Industrials & Materials",
    "Utilities": "Utilities & Real Estate",
    "Real Estate": "Utilities & Real Estate",
}
