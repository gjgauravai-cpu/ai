"""Backtest engine: turn a target-weight series into honest net returns.

Net daily return =
    weight_t * letf_ret_t                 (invested in the LETF)
  + (1 - weight_t) * rf_daily_t           (remainder earns cash interest)
  - |Δweight_t| * half_spread             (round-trip cost on rebalancing)

The spread is the dominant real-world cost on a $100 account; commissions are
zero on Robinhood and regulatory fees are negligible, so they are omitted.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import EngineConfig, LetfSpec
import metrics


def run_backtest(letf_ret: pd.Series, weight: pd.Series, spec: LetfSpec,
                 cfg: EngineConfig, rf: pd.Series | float) -> dict:
    """Return net/gross daily returns plus turnover for one strategy."""
    w = weight.reindex(letf_ret.index).fillna(0.0).clip(0.0, cfg.max_weight)
    if isinstance(rf, pd.Series):
        rf_d = rf.reindex(letf_ret.index).ffill().fillna(cfg.fallback_rf) / cfg.trading_days
    else:
        rf_d = pd.Series(rf / cfg.trading_days, index=letf_ret.index)

    half_spread = spec.half_spread_bps / 1e4
    turn = w.diff().abs().fillna(w.iloc[0] if len(w) else 0.0)   # first day = entry cost
    cost = turn * half_spread

    gross = w * letf_ret + (1.0 - w) * rf_d
    net = gross - cost

    ann_turnover = float(turn.sum() / (len(turn) / cfg.trading_days))
    return {
        "net": net.rename(weight.name),
        "gross": gross.rename(weight.name),
        "weight": w,
        "ann_turnover": ann_turnover,
        "avg_exposure": float(w.mean()),
    }


def evaluate(letf_ret: pd.Series, weights: dict[str, pd.Series], spec: LetfSpec,
             cfg: EngineConfig, rf: pd.Series | float) -> pd.DataFrame:
    """Backtest every strategy and assemble a metrics table (net of costs)."""
    rows = []
    for name, w in weights.items():
        bt = run_backtest(letf_ret, w, spec, cfg, rf)
        s = metrics.summary(bt["net"], rf=cfg.rf_for_sharpe,
                            trading_days=cfg.trading_days,
                            start_value=cfg.starting_capital)
        s_gross = metrics.summary(bt["gross"], rf=cfg.rf_for_sharpe,
                                  trading_days=cfg.trading_days,
                                  start_value=cfg.starting_capital)
        rows.append({
            "strategy": name,
            "CAGR_net": s["CAGR"], "CAGR_gross": s_gross["CAGR"],
            "AnnVol": s["AnnVol"], "Sharpe_net": s["Sharpe"],
            "MaxDD": s["MaxDD"], "MaxDD_days": s["MaxDD_days"],
            "Calmar": s["Calmar"], "AvgExp": bt["avg_exposure"],
            "Turnover": bt["ann_turnover"],
            f"Final_${int(cfg.starting_capital)}": s["FinalValue"],
        })
    return pd.DataFrame(rows).set_index("strategy")


def net_return_curves(letf_ret: pd.Series, weights: dict[str, pd.Series],
                      spec: LetfSpec, cfg: EngineConfig,
                      rf: pd.Series | float) -> pd.DataFrame:
    """Equity curves ($-scaled) for plotting."""
    curves = {}
    for name, w in weights.items():
        bt = run_backtest(letf_ret, w, spec, cfg, rf)
        curves[name] = metrics.equity_curve(bt["net"], cfg.starting_capital)
    return pd.DataFrame(curves)
