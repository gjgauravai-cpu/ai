"""Synthetic daily-reset LETF reconstruction and return decomposition.

A daily-reset L-times ETF earns, each day:
    r_letf = L * r_index  -  (L-1) * (rf + spread)/252  -  expense/252
i.e. L times the underlying's *daily* return, minus financing on the borrowed
(L-1) notional, minus the fund fee. Compounding this daily series reproduces the
real LETF closely; the gap vs. the naive L*(cumulative index) is volatility decay.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import LetfSpec


def synthetic_letf_returns(index_ret: pd.Series, spec: LetfSpec,
                           rf: pd.Series | float, financing_spread: float,
                           trading_days: int = 252) -> pd.Series:
    """Reconstruct the LETF's daily return series from the underlying."""
    if isinstance(rf, pd.Series):
        rf_d = rf.reindex(index_ret.index).ffill().fillna(0.0) / trading_days
    else:
        rf_d = pd.Series(rf / trading_days, index=index_ret.index)
    financing = (spec.leverage - 1.0) * (rf_d + financing_spread / trading_days)
    fee = spec.expense_ratio / trading_days
    return (spec.leverage * index_ret - financing - fee).rename(spec.symbol)


def decompose(index_ret: pd.Series, spec: LetfSpec, rf: pd.Series | float,
              financing_spread: float, trading_days: int = 252) -> dict[str, float]:
    """Break the LETF's total log-return over the window into its drivers.

    log(LETF) ~= L*log(index) - 0.5*L*(L-1)*RealizedVar - financing - fees
    All terms are reported as total log-return contribution over the window.
    """
    idx = index_ret.dropna()
    L = spec.leverage
    # geometric (log) returns
    g_index = float(np.log1p(idx).sum())
    realized_var = float((np.log1p(idx) ** 2).sum())         # sum of squared daily log-rets
    var_drag = -0.5 * L * (L - 1.0) * realized_var

    if isinstance(rf, pd.Series):
        rf_d = rf.reindex(idx.index).ffill().fillna(0.0) / trading_days
    else:
        rf_d = pd.Series(rf / trading_days, index=idx.index)
    financing = -float(((L - 1.0) * (rf_d + financing_spread / trading_days)).sum())
    fees = -float((spec.expense_ratio / trading_days) * len(idx))

    naive = L * g_index
    modeled = naive + var_drag + financing + fees
    years = len(idx) / trading_days
    return {
        "years": years,
        "index_log_ret": g_index,
        "naive_Lx_log_ret": naive,                   # what people *think* they get
        "var_drag_log": var_drag,                    # volatility decay
        "financing_log": financing,
        "fees_log": fees,
        "modeled_letf_log_ret": modeled,
        "naive_Lx_total_%": (np.exp(naive) - 1.0) * 100.0,
        "modeled_total_%": (np.exp(modeled) - 1.0) * 100.0,
        "decay+cost_drag_%": (np.exp(modeled) - np.exp(naive)) / np.exp(naive) * 100.0,
        "ann_index_vol": float(np.log1p(idx).std(ddof=0) * np.sqrt(trading_days)),
    }


def tracking_error(actual_ret: pd.Series, synth_ret: pd.Series,
                   trading_days: int = 252) -> dict[str, float]:
    """How well the synthetic reconstruction matches the real LETF."""
    a, s = actual_ret.align(synth_ret, join="inner")
    diff = (a - s).dropna()
    a_eq = float((1 + a.loc[diff.index]).prod())
    s_eq = float((1 + s.loc[diff.index]).prod())
    return {
        "n_days": len(diff),
        "daily_corr": float(a.loc[diff.index].corr(s.loc[diff.index])),
        "ann_tracking_err": float(diff.std(ddof=0) * np.sqrt(trading_days)),
        "actual_total_%": (a_eq - 1) * 100.0,
        "synth_total_%": (s_eq - 1) * 100.0,
    }
