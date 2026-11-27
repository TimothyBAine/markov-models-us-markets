from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    start: str = "2005-01-01"
    end: str | None = None
    refresh: bool = False

    # HMM
    n_states: int = 3
    k_grid: tuple[int, ...] = (2, 3, 4, 5)
    n_restarts: int = 10
    seed: int = 0
    vol_halflife: int = 10          # EWMA half-life (days) for market vol feature

    # Model A: rolling network features
    roll_window: int = 63           # ~1 quarter
    roll_step: int = 5              # weekly sampling to thin overlap
    corr_threshold: float = 0.5     # threshold graph |rho| > tau

    # Model B: regime-conditional networks
    glasso_alpha: float | None = None   # None -> choose by EBIC per regime
    ebic_gamma: float = 0.5
    residualize: bool = False           # remove market beta before building networks

    data_dir: Path = field(default_factory=lambda: Path("data"))
    out_dir: Path = field(default_factory=lambda: Path("outputs"))
