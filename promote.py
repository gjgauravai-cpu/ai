"""Research-to-production pipeline: backtest a strategy full-cycle, validate it
(matched-null), and run it through the promotion gate. PASS -> cleared to execute
(paper first); FAIL -> stays in research.

Usage:
    python promote.py vol_target_har_live
    python promote.py vol_target_gjr
This is the ONLY path to execution: nothing goes live without clearing the gate.
"""
from __future__ import annotations

import sys

import data
import gate
import validate
import walkforward
from config import EngineConfig
from run_synthetic import run_one

if __name__ == "__main__":
    name = (sys.argv[1] if len(sys.argv) > 1 else "vol_target_har_live")
    cfg = EngineConfig(start="1999-01-01")

    prices = data.load_prices(["QQQ", cfg.vix_symbol], cfg.start, None)
    rf = data.risk_free_series(cfg.start, None, cfg.fallback_rf)
    vix = prices[cfg.vix_symbol]["close"] if cfg.vix_symbol in prices else None

    print(f"Backtesting '{name}' full-cycle (synthetic 3x QQQ since 1999)...")
    r = run_one("QQQ", 3.0, prices["QQQ"], rf, cfg, [name, "buy_hold"], vix)
    row = r["table"].loc[name]

    print("Validating (matched-null, 400 rotations)...")
    mn = validate.matched_null(r["letf_ret"], r["weights"][name], r["spec"], cfg, rf,
                               n_rotations=400)
    _, _, oos_checks = walkforward.promotion_checks(
        r["letf_ret"], r["weights"][name], r["spec"], cfg, rf)

    verdict = gate.evaluate(
        name,
        net_sharpe=float(row["Sharpe_net"]),
        pvalue=float(mn["p_value"]),
        net_cagr=float(row["CAGR_net"]),
        maxdd=float(row["MaxDD"]),
        additional_checks=oos_checks,
    )
    print("\n" + gate.render(verdict))
