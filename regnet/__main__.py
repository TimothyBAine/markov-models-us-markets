import argparse
from pathlib import Path

from .config import Config
from .pipeline import run


def main():
    p = argparse.ArgumentParser(prog="python -m regnet",
                                description="Regime-switching stock networks (HMM x network science)")
    p.add_argument("--start", default=Config.start)
    p.add_argument("--end", default=None)
    p.add_argument("--states", type=int, default=Config.n_states, help="number of HMM regimes")
    p.add_argument("--restarts", type=int, default=Config.n_restarts)
    p.add_argument("--window", type=int, default=Config.roll_window, help="Model A rolling window (days)")
    p.add_argument("--step", type=int, default=Config.roll_step, help="Model A sampling step (days)")
    p.add_argument("--alpha", type=float, default=None, help="fixed glasso alpha (default: EBIC)")
    p.add_argument("--residualize", action="store_true", help="strip market beta before networks")
    p.add_argument("--refresh", action="store_true", help="re-download prices")
    p.add_argument("--out", type=Path, default=Path("outputs"))
    a = p.parse_args()
    run(Config(start=a.start, end=a.end, n_states=a.states, n_restarts=a.restarts,
               roll_window=a.window, roll_step=a.step, glasso_alpha=a.alpha,
               residualize=a.residualize, refresh=a.refresh, out_dir=a.out))


if __name__ == "__main__":
    main()
