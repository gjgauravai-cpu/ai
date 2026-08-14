"""Performance and risk metrics. Pure functions over a daily-return Series."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import kurtosis as _kurtosis
from scipy.stats import norm
from scipy.stats import skew as _skew

EULER_GAMMA = 0.5772156649015329


def equity_curve(returns: pd.Series, start_value: float = 1.0) -> pd.Series:
    return start_value * (1.0 + returns.fillna(0.0)).cumprod()


def cagr(returns: pd.Series, trading_days: int = 252) -> float:
    r = returns.dropna()
    if len(r) < 2:
        return float("nan")
    growth = float((1.0 + r).prod())
    years = len(r) / trading_days
    if growth <= 0:      # wiped out -> total loss
        return -1.0
    return growth ** (1.0 / years) - 1.0


def ann_vol(returns: pd.Series, trading_days: int = 252) -> float:
    return float(returns.dropna().std(ddof=0) * np.sqrt(trading_days))


def sharpe(returns: pd.Series, rf: float = 0.0, trading_days: int = 252) -> float:
    r = returns.dropna()
    if r.std(ddof=0) == 0:
        return float("nan")
    excess = r - rf / trading_days
    return float(excess.mean() / r.std(ddof=0) * np.sqrt(trading_days))


def sortino(returns: pd.Series, rf: float = 0.0, trading_days: int = 252) -> float:
    r = returns.dropna()
    downside = r[r < 0].std(ddof=0)
    if downside == 0 or np.isnan(downside):
        return float("nan")
    excess = r.mean() - rf / trading_days
    return float(excess / downside * np.sqrt(trading_days))


def drawdown_series(returns: pd.Series, start_value: float = 1.0) -> pd.Series:
    eq = equity_curve(returns, start_value)
    peak = eq.cummax()
    return eq / peak - 1.0


def max_drawdown(returns: pd.Series) -> float:
    return float(drawdown_series(returns).min())


def max_drawdown_duration_days(returns: pd.Series) -> int:
    """Longest stretch (in bars) spent below a prior peak."""
    dd = drawdown_series(returns)
    underwater = dd < 0
    longest = cur = 0
    for u in underwater:
        cur = cur + 1 if u else 0
        longest = max(longest, cur)
    return int(longest)


def calmar(returns: pd.Series, trading_days: int = 252) -> float:
    mdd = abs(max_drawdown(returns))
    if mdd == 0:
        return float("nan")
    return cagr(returns, trading_days) / mdd


def probabilistic_sharpe(returns: pd.Series, sr_benchmark: float = 0.0,
                         trading_days: int = 252) -> float:
    """Probabilistic Sharpe Ratio (Bailey & Lopez de Prado 2014).

    P(true per-period Sharpe > sr_benchmark), correcting for sample length,
    skewness and (non-excess) kurtosis. sr_benchmark is in per-period units.
    """
    r = returns.dropna()
    n = len(r)
    sd = r.std(ddof=1)
    if n < 8 or sd == 0:
        return float("nan")
    sr = r.mean() / sd                              # per-period Sharpe
    g3 = float(_skew(r, bias=False))
    g4 = float(_kurtosis(r, fisher=False, bias=False))   # non-excess kurtosis
    denom = np.sqrt(max(1.0 - g3 * sr + (g4 - 1.0) / 4.0 * sr ** 2, 1e-12))
    return float(norm.cdf((sr - sr_benchmark) * np.sqrt(n - 1) / denom))


def expected_max_sharpe(n_trials: int, var_sr: float) -> float:
    """Expected maximum per-period Sharpe across n_trials iid strategies (null)."""
    if n_trials < 2 or var_sr <= 0:
        return 0.0
    z1 = norm.ppf(1.0 - 1.0 / n_trials)
    z2 = norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    return float(np.sqrt(var_sr) * ((1.0 - EULER_GAMMA) * z1 + EULER_GAMMA * z2))


def deflated_sharpe(returns: pd.Series, trial_sharpes_annual: list[float],
                    trading_days: int = 252) -> float:
    """Deflated Sharpe Ratio: PSR evaluated against the expected-max Sharpe from
    the full set of strategies tried (multiple-testing correction). >0.95 ~ real.
    """
    arr = np.asarray([s for s in trial_sharpes_annual if np.isfinite(s)], dtype=float)
    if arr.size < 2:
        return float("nan")
    daily = arr / np.sqrt(trading_days)
    sr_star = expected_max_sharpe(arr.size, float(np.var(daily, ddof=1)))
    return probabilistic_sharpe(returns, sr_benchmark=sr_star, trading_days=trading_days)


def summary(returns: pd.Series, rf: float = 0.045, trading_days: int = 252,
            start_value: float = 100.0) -> dict[str, float]:
    eq = equity_curve(returns, start_value)
    return {
        "CAGR": cagr(returns, trading_days),
        "AnnVol": ann_vol(returns, trading_days),
        "Sharpe": sharpe(returns, rf, trading_days),
        "Sortino": sortino(returns, rf, trading_days),
        "MaxDD": max_drawdown(returns),
        "Calmar": calmar(returns, trading_days),
        "MaxDD_days": max_drawdown_duration_days(returns),
        "FinalValue": float(eq.iloc[-1]) if len(eq) else float("nan"),
    }
