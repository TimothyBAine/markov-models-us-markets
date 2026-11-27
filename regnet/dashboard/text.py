"""User-facing explainer text for the dashboard (edited for plain language)."""

CONTROL_TIPS = {
    'k': 'How many market states the model splits history into. States are sorted by volatility, so the first is always the calmest. A new value refits the model, which takes a few seconds the first time.',
    'mode': "Raw returns include the market-wide move that lifts or sinks every stock at once. Market removed strips out each stock's sensitivity to SPY first, leaving what stocks share beyond the market. Compare both shows the two side by side.",
    'kind': 'Partial correlation links two stocks that still move together after accounting for every other stock. Minimum spanning tree keeps only the strongest links needed to connect every stock, with no loops. Correlation threshold links every pair above τ.',
    'sparsity': 'Controls how many links the partial-correlation network keeps. Auto picks the penalty that best balances fit and simplicity, so density changes from frame to frame. Fixed edge count keeps density constant, which makes frames easier to compare over time.',
    'alpha': 'Higher values prune more links. Start around 0.05 to 0.1.',
    'topn': 'Keeps this many of the strongest links.',
    'tau': 'Links two stocks when their correlation is above this value. With the market removed, correlations are much smaller, so lower τ.',
    'drawmin': 'Hides weak links in the picture only. The numbers above each network still count every link.',
    'window': 'Trading days behind each network in the Through time view. Longer windows give steadier networks but react later. 126 days is about six months.',
    'layout': 'Sector ring keeps each sector in a fixed cluster, so links within and across sectors are easy to tell apart. Force (fixed) places stocks by their long-run relationships. Adaptive rearranges every frame, so clusters drift as relationships change.',
}

METRIC_TIPS = {
    'edges': 'Number of links in the network.',
    'density': 'Share of all possible stock pairs that are linked.',
    'mean_corr': 'Average correlation across every pair of stocks.',
    'absorption': 'Share of total variance explained by the single biggest common factor. Close to 1 means everything moves as one.',
    'modularity': 'How much the links stay inside sectors. 0 means no sector pattern; higher means links cluster by sector.',
    'nmi': 'How well clusters found from the links match the official sectors, from 0 (unrelated) to 1 (identical).',
    'hub': 'The stock with the most links, with its link count.',
    'alpha': 'Penalty used by the graphical lasso. In fixed edge count mode it stays at 0.03, and the dashboard then keeps the strongest links.',
    'neg': 'Share of links where the two stocks move in opposite directions once everything else is accounted for.',
    'n': 'Days behind the estimate. In By regime it is an effective count, because days are weighted by regime probability.',
    'regime_mix': "Share of the window's days the model assigns to each regime, on average.",
}

CAPTIONS = {
    'timeline': 'Click the chart to jump to a date. The boxed area is the window behind the networks below.',
    'by_regime': 'Each network uses every day in the sample, weighted by how likely that day is to belong to the regime.',
}

GUIDE = [
    "The shading behind SPY is the regime the model assigns to each day, from light (calm) to dark (stressed). It is based on SPY's daily return, short-term volatility and how far individual stocks' returns spread apart.",
    'Each dot is a stock, coloured by sector. A link means two stocks move together: thicker is stronger, and dashed means they move in opposite directions once the rest of the market is accounted for.',
    "Each network's border takes the colour of the regime at the end of its window.",
    'In stressed markets nearly every pair of stocks rises and falls together. Switch the market factor to Market removed to see which relationships remain.',
    "Regimes are estimated with hindsight from the full history, so a day's label can depend on what happened after it.",
]
