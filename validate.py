"""Validation harness: tell real timing skill from leverage-in-disguise.

The Deflated Sharpe Ratio as wired greenlights every strategy (even naked
buy&hold scores >0.95) because its null is not matched to a strategy's exposure
and turnover. This module fixes the ruler with two ex-post, strictly causal
tests over the already-computed weight/return series:

1. Exposure/turnover-matched null (circular rotation). Rotate the weight series
   by random offsets against the return series. Each rotation keeps the SAME mean
   exposure, the SAME turnover and the SAME weight autocorrelation, but its
   alignment to returns is destroyed -- so genuine timing skill vanishes while the
   risk budget is held fixed. The real Sharpe's p-value against this null answers
   "is the timing real, or just the equity-risk-premium a high-beta book earns?".

2. Regime-split sign-flip. Does the strategy's edge OVER buy&hold keep its sign
   across pre/post-2020 and high/low realized-vol regimes? A sign flip is the
   signature of a period-specific (decaying) edge, not a durable one -- an
   alpha-half-life detector, the right answer to documented ~18-month AI decay.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backtest import run_backtest
from config import EngineConfig, LetfSpec
from metrics import sharpe


def matched_null(letf_ret: pd.Series, weight: pd.Series, spec: LetfSpec,
                 cfg: EngineConfig, rf: pd.Series | float,
                 n_rotations: int = 400) -> dict[str, float]:
    """Circular-rotation null: real Sharpe vs same-weights-different-timing."""
    w = weight.reindex(letf_ret.index).fillna(0.0)
    real_sr = sharpe(run_backtest(letf_ret, w, spec, cfg, rf)["net"],
                     cfg.rf_for_sharpe, cfg.trading_days)
    vals = w.values
    n = len(vals)
    lo, hi = cfg.trading_days, max(cfg.trading_days + 1, n - cfg.trading_days)
    rng = np.random.default_rng(cfg.seed)
    null = np.empty(n_rotations)
    for i, off in enumerate(rng.integers(lo, hi, size=n_rotations)):
        w_rot = pd.Series(np.roll(vals, int(off)), index=w.index, name=w.name)
        null[i] = sharpe(run_backtest(letf_ret, w_rot, spec, cfg, rf)["net"],
                         cfg.rf_for_sharpe, cfg.trading_days)
    null = null[np.isfinite(null)]
    if null.size == 0:
        return {"real_sharpe": real_sr, "null_p95": float("nan"),
                "percentile": float("nan"), "p_value": float("nan")}
    return {
        "real_sharpe": float(real_sr),
        "null_p95": float(np.quantile(null, 0.95)),
        "percentile": float((null < real_sr).mean()),
        "p_value": float((null >= real_sr).mean()),
    }


def regime_split(letf_ret: pd.Series, weight: pd.Series, spec: LetfSpec,
                 cfg: EngineConfig, rf: pd.Series | float, und_close: pd.Series,
                 split_date: str = "2020-01-01") -> dict[str, float | bool]:
    """Sign of edge-over-buy&hold across date and vol-regime splits."""
    w = weight.reindex(letf_ret.index).fillna(0.0)
    net = run_backtest(letf_ret, w, spec, cfg, rf)["net"]
    bh = run_backtest(letf_ret, pd.Series(1.0, index=letf_ret.index),
                      spec, cfg, rf)["net"]
    excess = (net - bh).dropna()

    cut = pd.Timestamp(split_date)
    rv = und_close.pct_change().rolling(21).std().shift(1).reindex(excess.index)
    med = rv.median()

    def _sr(x: pd.Series) -> float:
        return sharpe(x, 0.0, cfg.trading_days) if len(x) > 20 else float("nan")

    pre, post = excess[excess.index < cut], excess[excess.index >= cut]
    hivol, lovol = excess[rv > med], excess[rv <= med]
    parts = {"pre2020": _sr(pre), "post2020": _sr(post),
             "hivol": _sr(hivol), "lovol": _sr(lovol)}
    signs = {np.sign(round(v, 6)) for v in parts.values() if np.isfinite(v) and v != 0}
    return {**{f"xsSR_{k}": v for k, v in parts.items()},
            "sign_consistent": len(signs) <= 1}


def validate_strategies(letf_ret: pd.Series, weights: dict[str, pd.Series],
                        spec: LetfSpec, cfg: EngineConfig, rf: pd.Series | float,
                        und_close: pd.Series, n_rotations: int = 400,
                        split_date: str = "2020-01-01") -> pd.DataFrame:
    """Run both tests for every strategy; return a tidy verdict table."""
    rows = []
    for name, w in weights.items():
        mn = matched_null(letf_ret, w, spec, cfg, rf, n_rotations)
        rs = regime_split(letf_ret, w, spec, cfg, rf, und_close, split_date)
        verdict = ("benchmark" if name == "buy_hold"
                   else "REAL timing edge" if (mn["p_value"] <= 0.05
                                               and rs["sign_consistent"])
                   else "weak/period-biased" if mn["p_value"] <= 0.05
                   else "not distinguishable from matched-null")
        rows.append({"strategy": name, **mn, **rs, "verdict": verdict})
    return pd.DataFrame(rows).set_index("strategy")
