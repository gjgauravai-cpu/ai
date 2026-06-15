"""Universe, cost, and model configuration for the leveraged-ETF TS engine.

Single source of truth. Everything downstream reads from here so there are no
hard-coded tickers/params scattered across modules.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class LetfSpec:
    """Static description of one leveraged ETF.

    underlying is the *tradable proxy* for the index the LETF tracks; it drives
    the synthetic daily-reset reconstruction and the trend/regime signal.
    half_spread_bps is one-side spread+slippage applied per unit of turnover.
    """

    symbol: str
    leverage: float
    underlying: str
    expense_ratio: float          # annual, e.g. 0.0095 = 0.95%
    half_spread_bps: float        # one-side cost in basis points
    name: str = ""


# Leveraged-ETF universe (broad index + sector/thematic).
# Expense ratios and spreads are realistic estimates as of 2026; tune freely.
UNIVERSE: dict[str, LetfSpec] = {
    # --- broad equity index ---
    "TQQQ": LetfSpec("TQQQ", 3.0, "QQQ", 0.0084, 3.0, "ProShares UltraPro QQQ"),
    "QLD":  LetfSpec("QLD",  2.0, "QQQ", 0.0095, 3.0, "ProShares Ultra QQQ"),
    "UPRO": LetfSpec("UPRO", 3.0, "SPY", 0.0091, 3.0, "ProShares UltraPro S&P 500"),
    "SSO":  LetfSpec("SSO",  2.0, "SPY", 0.0089, 3.0, "ProShares Ultra S&P 500"),
    "SPXL": LetfSpec("SPXL", 3.0, "SPY", 0.0091, 4.0, "Direxion Daily S&P 500 Bull 3x"),
    "TNA":  LetfSpec("TNA",  3.0, "IWM", 0.0107, 6.0, "Direxion Daily Small Cap Bull 3x"),
    "UDOW": LetfSpec("UDOW", 3.0, "DIA", 0.0095, 6.0, "ProShares UltraPro Dow30"),
    # --- sector & thematic ---
    "SOXL": LetfSpec("SOXL", 3.0, "SOXX", 0.0075, 8.0, "Direxion Daily Semiconductor Bull 3x"),
    "TECL": LetfSpec("TECL", 3.0, "XLK", 0.0094, 8.0, "Direxion Daily Technology Bull 3x"),
    "FAS":  LetfSpec("FAS",  3.0, "XLF", 0.0096, 8.0, "Direxion Daily Financial Bull 3x"),
    "LABU": LetfSpec("LABU", 3.0, "XBI", 0.0097, 12.0, "Direxion Daily S&P Biotech Bull 3x"),
    "NUGT": LetfSpec("NUGT", 2.0, "GDX", 0.0119, 12.0, "Direxion Daily Gold Miners Bull 2x"),
}

# Default "survivable core" recommended from the analysis.
CORE = ["QLD", "SSO"]
# Default comparison set for the headline run.
DEFAULT_SET = ["QLD", "SSO", "TQQQ", "UPRO", "SOXL"]


@dataclass(frozen=True)
class EngineConfig:
    start: str = "2010-01-01"      # extend to inception where data allows
    end: str | None = None         # None -> today
    # financing
    financing_spread: float = 0.0040   # swap spread over the risk-free rate
    fallback_rf: float = 0.045         # used if the T-bill series is unavailable
    trading_days: int = 252
    # vol model
    garch_refit_every: int = 21        # refit GARCH params every N days (walk-forward)
    garch_min_obs: int = 252           # min history before GARCH engages
    target_vol: float = 0.25           # annualized vol target for vol-targeted strat
    max_weight: float = 1.0            # cap on LETF exposure (1.0 = no stacking)
    # regime
    sma_window: int = 200              # trend filter on the underlying
    hmm_states: int = 2
    autocorr_window: int = 63          # rolling AR(1) window (LETF-dynamics, arXiv 2504.20116)
    autocorr_gain: float = 2.0         # maps AR(1) coefficient to an exposure tilt
    # implied-vol floor for the HAR sizer (multi-agent panel rank 2)
    iv_floor_beta: float = 0.65        # floor = beta * L * VIX/100; tuned to bind ~10-20% of days
    vix_symbol: str = "^VIX"           # forward implied-vol index used for the floor
    # live deployment: low-turnover rebalancing (T+1 cash-account safe)
    live_rebalance_days: int = 5       # only revisit the target weight weekly
    live_band: float = 0.08            # and only trade when |target - held| exceeds this
    # backtest
    starting_capital: float = 100.0    # matches the funded agentic account
    rf_for_sharpe: float = 0.045       # flat rf for Sharpe (overridden by series if present)
    seed: int = 7

    strategies: tuple[str, ...] = field(
        default_factory=lambda: (
            "buy_hold",
            "regime_sma",
            "vol_target",
            "vol_target_regime",
        )
    )


DEFAULT = EngineConfig()
