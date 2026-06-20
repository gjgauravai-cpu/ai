"""Promotion gate — the final checkpoint between backtest and execution.

A strategy is cleared to EXECUTE only when its full-cycle backtest clears a
pre-committed bar (the "appropriate level"). Otherwise it stays in research.
Promotion goes to PAPER first; live real money still needs an explicit human flip
— we never push a strategy to real money on a backtest alone.

PRE-COMMITTED CRITERIA — do NOT relax these to force a strategy through. Loosening
the gate to make something pass is exactly how overfit strategies reach production
and lose real money. If a candidate fails, it fails.
"""
from __future__ import annotations

import json
from pathlib import Path

MIN_NET_SHARPE = 0.40      # full-cycle, AFTER costs
MAX_PVALUE = 0.05          # matched-null: real timing edge, not luck
MIN_MAXDD = -0.85          # survivable (a -90%+ path fails even with a nice Sharpe)
MAX_CORR_NEW = 0.50        # a NEW sleeve must actually diversify the live book


def evaluate(name: str, net_sharpe: float, pvalue: float, net_cagr: float,
             maxdd: float, corr_to_live: float | None = None) -> dict:
    checks = {
        f"net Sharpe >= {MIN_NET_SHARPE}": net_sharpe >= MIN_NET_SHARPE,
        f"matched-null p < {MAX_PVALUE} (real edge)": pvalue < MAX_PVALUE,
        "net CAGR > 0 (pays after costs)": net_cagr > 0,
        f"max DD survivable (> {MIN_MAXDD:.0%})": maxdd > MIN_MAXDD,
    }
    if corr_to_live is not None:
        checks[f"|corr to live| < {MAX_CORR_NEW} (diversifies)"] = abs(corr_to_live) < MAX_CORR_NEW
    passed = all(checks.values())
    return {
        "strategy": name,
        "metrics": {"net_sharpe": round(net_sharpe, 3), "pvalue": round(pvalue, 3),
                    "net_cagr": round(net_cagr, 4), "maxdd": round(maxdd, 3),
                    "corr_to_live": None if corr_to_live is None else round(corr_to_live, 2)},
        "checks": checks,
        "PASS": passed,
        "action": ("PROMOTE -> deploy to PAPER (Alpaca), then human flip to live"
                   if passed else "REJECT -> keep in research, do NOT execute"),
    }


def render(v: dict) -> str:
    out = [f"=== PROMOTION GATE: {v['strategy']} ===", f"metrics: {v['metrics']}"]
    for k, ok in v["checks"].items():
        out.append(f"  [{'PASS' if ok else 'FAIL'}] {k}")
    flag = "*** PASS *** -> " if v["PASS"] else "*** FAIL *** -> "
    out.append(f">>> {flag}{v['action']}")
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# Execution allow-list — the hard, fail-closed precondition for trading.       #
# Only strategies written here by the leaderboard (i.e. that PASSED the gate)  #
# may execute. Missing file / missing name => execution refused.               #
# --------------------------------------------------------------------------- #
def _cleared_path() -> Path:
    here = Path(__file__).parent
    for p in (here / "output" / "cleared.json", here / "cleared.json"):
        if p.exists():
            return p
    return here / "output" / "cleared.json"


def cleared_set() -> set[str]:
    try:
        return set(json.loads(_cleared_path().read_text()).get("cleared_strategies", []))
    except Exception:          # noqa: BLE001 - fail closed
        return set()


def is_cleared(name: str) -> bool:
    return name in cleared_set()


def assert_cleared(name: str) -> None:
    """Raise unless `name` is on the promotion-gate allow-list. Fail-closed:
    no allow-list, or name not on it, => no execution."""
    if not is_cleared(name):
        raise SystemExit(
            f"EXECUTION BLOCKED: '{name}' has not cleared the promotion gate "
            f"(not in output/cleared.json). Run `python leaderboard.py` and let it "
            f"PASS the gate before any paper or live execution.")
