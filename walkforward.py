"""Out-of-sample and execution-friction checks for promotion candidates.

This module does not search parameters and cannot place orders.  It evaluates
pre-registered strategy rules over contiguous, non-overlapping out-of-sample
windows and reruns the same windows under wider execution costs.  That keeps
the promotion process focused on robust behaviour rather than the best
in-sample backtest.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import pandas as pd

import data
from backtest import run_backtest
from config import EngineConfig, LetfSpec
from metrics import summary
from run_synthetic import run_one


def fold_bounds(n_obs: int, min_train_days: int = 756, test_days: int = 252,
                step_days: int = 252) -> list[tuple[int, int]]:
    """Return contiguous OOS windows after a fixed initial training history."""
    if min_train_days < 1 or test_days < 21 or step_days < test_days:
        raise ValueError("invalid walk-forward window configuration")
    if n_obs < min_train_days + test_days:
        return []
    return [(start, start + test_days)
            for start in range(min_train_days, n_obs - test_days + 1, step_days)]


def evaluate_walk_forward(letf_ret: pd.Series, weight: pd.Series, spec: LetfSpec,
                          cfg: EngineConfig, rf: pd.Series | float,
                          min_train_days: int = 756, test_days: int = 252,
                          step_days: int = 252) -> tuple[pd.DataFrame, dict[str, bool]]:
    """Score a fixed rule across non-overlapping OOS folds and cost stresses."""
    bounds = fold_bounds(len(letf_ret), min_train_days, test_days, step_days)
    rows: list[dict[str, float | int | str]] = []
    for fold, (start, end) in enumerate(bounds, start=1):
        ret = letf_ret.iloc[start:end]
        w = weight.reindex(letf_ret.index).fillna(0.0).iloc[start:end]
        net = run_backtest(ret, w, spec, cfg, rf)["net"]
        bh = run_backtest(ret, pd.Series(1.0, index=ret.index, name="buy_hold"),
                          spec, cfg, rf)["net"]
        s, b = summary(net, cfg.rf_for_sharpe, cfg.trading_days), summary(
            bh, cfg.rf_for_sharpe, cfg.trading_days)
        rows.append({
            "fold": fold,
            "start": str(ret.index[0].date()),
            "end": str(ret.index[-1].date()),
            "CAGR_net": s["CAGR"],
            "Sharpe_net": s["Sharpe"],
            "MaxDD": s["MaxDD"],
            "CAGR_vs_buy_hold": s["CAGR"] - b["CAGR"],
            "Sharpe_vs_buy_hold": s["Sharpe"] - b["Sharpe"],
        })
    folds = pd.DataFrame(rows)
    if folds.empty:
        return folds, {
            "walk-forward has at least 3 OOS folds": False,
            "median OOS Sharpe is positive": False,
            "at least two-thirds of OOS folds have positive CAGR": False,
        }
    checks = {
        "walk-forward has at least 3 OOS folds": len(folds) >= 3,
        "median OOS Sharpe is positive": float(folds["Sharpe_net"].median()) > 0.0,
        "at least two-thirds of OOS folds have positive CAGR":
            float((folds["CAGR_net"] > 0.0).mean()) >= 2.0 / 3.0,
    }
    return folds, checks


def stress_costs(letf_ret: pd.Series, weight: pd.Series, spec: LetfSpec,
                 cfg: EngineConfig, rf: pd.Series | float,
                 multipliers: tuple[float, ...] = (1.0, 1.5, 2.0)) -> pd.DataFrame:
    """Reprice identical decisions under conservative one-way execution costs."""
    rows = []
    for multiplier in multipliers:
        stressed = replace(spec, half_spread_bps=spec.half_spread_bps * multiplier)
        net = run_backtest(letf_ret, weight, stressed, cfg, rf)["net"]
        rows.append({"cost_multiplier": multiplier,
                     **summary(net, cfg.rf_for_sharpe, cfg.trading_days)})
    return pd.DataFrame(rows).set_index("cost_multiplier")


def promotion_checks(letf_ret: pd.Series, weight: pd.Series, spec: LetfSpec,
                     cfg: EngineConfig, rf: pd.Series | float) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, bool]]:
    """Return OOS evidence and the additional hard checks required for promotion."""
    folds, checks = evaluate_walk_forward(letf_ret, weight, spec, cfg, rf)
    stress = stress_costs(letf_ret, weight, spec, cfg, rf)
    checks["2x modeled execution cost retains positive CAGR"] = (
        not stress.empty and float(stress.loc[2.0, "CAGR"]) > 0.0
    )
    return folds, stress, checks


def main() -> None:
    ap = argparse.ArgumentParser(description="Research-only walk-forward promotion checks")
    ap.add_argument("strategy", nargs="?", default="vol_target_har_live")
    ap.add_argument("--start", default="1999-01-01")
    args = ap.parse_args()
    cfg = EngineConfig(start=args.start)
    prices = data.load_prices(["QQQ", cfg.vix_symbol], cfg.start, None)
    rf = data.risk_free_series(cfg.start, None, cfg.fallback_rf)
    vix = prices[cfg.vix_symbol]["close"] if cfg.vix_symbol in prices else None
    result = run_one("QQQ", 3.0, prices["QQQ"], rf, cfg,
                     [args.strategy, "buy_hold"], vix)
    folds, stress, checks = promotion_checks(result["letf_ret"], result["weights"][args.strategy],
                                              result["spec"], cfg, rf)
    print("WALK-FORWARD FOLDS")
    print(folds.to_string(index=False))
    print("\nCOST STRESS")
    print(stress.to_string())
    print("\nPROMOTION CHECKS")
    print(json.dumps(checks, indent=2))
    out = Path(__file__).parent / "output"
    folds.to_csv(out / f"walkforward_{args.strategy}_folds.csv", index=False)
    stress.to_csv(out / f"walkforward_{args.strategy}_cost_stress.csv")


if __name__ == "__main__":
    main()
