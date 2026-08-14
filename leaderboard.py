"""Run EVERY registered strategy through the promotion gate and rank them — the
definitive "what is cleared to trade" leaderboard. Writes the PASS set to
output/cleared.json, which is the execution ALLOW-LIST (live.py / alpaca_exec.py
refuse to trade anything not on it).
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import data
import gate
import validate
import walkforward
from config import EngineConfig
from run_synthetic import run_one
from strategies import REGISTRY

cfg = EngineConfig(start="1999-01-01")
prices = data.load_prices(["QQQ", cfg.vix_symbol], cfg.start, None)
rf = data.risk_free_series(cfg.start, None, cfg.fallback_rf)
vix = prices[cfg.vix_symbol]["close"] if cfg.vix_symbol in prices else None

names = [n for n in REGISTRY if n != "buy_hold"]          # buy_hold = benchmark, not a candidate
print(f"Backtesting {len(names)} strategies full-cycle (synthetic 3x QQQ since 1999)...")
r = run_one("QQQ", 3.0, prices["QQQ"], rf, cfg, names + ["buy_hold"], vix)

rows, cleared = [], {}
for name in names:
    row = r["table"].loc[name]
    mn = validate.matched_null(r["letf_ret"], r["weights"][name], r["spec"], cfg, rf,
                               n_rotations=300)
    _, _, oos_checks = walkforward.promotion_checks(
        r["letf_ret"], r["weights"][name], r["spec"], cfg, rf)
    v = gate.evaluate(name, float(row["Sharpe_net"]), float(mn["p_value"]),
                      float(row["CAGR_net"]), float(row["MaxDD"]),
                      additional_checks=oos_checks)
    rows.append({"strategy": name, "PASS": v["PASS"], **v["metrics"],
                 "OOS_checks_pass": all(oos_checks.values())})
    if v["PASS"]:
        cleared[name] = v["metrics"]

df = pd.DataFrame(rows).sort_values(["PASS", "net_sharpe"], ascending=[False, False])
pd.set_option("display.width", 200)
print("\n" + "=" * 84)
print("PROMOTION-GATE LEADERBOARD — what is cleared to execute")
print("=" * 84)
print(df.to_string(index=False))

out = Path(__file__).parent / "output" / "cleared.json"
out.write_text(json.dumps({"cleared_strategies": list(cleared.keys()),
                           "detail": cleared}, indent=2))
print(f"\nCLEARED TO EXECUTE ({len(cleared)}): {list(cleared.keys())}")
print(f"REJECTED ({len(names) - len(cleared)}): "
      f"{[n for n in names if n not in cleared]}")
print(f"allow-list written -> output/cleared.json")
