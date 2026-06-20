"""Autonomous executor — runs in GitHub Actions with your laptop OFF.

It computes the validated vol_target_har_live target for TQQQ and rebalances via
Alpaca's OFFICIAL trading API. PAPER trading by default (ALPACA_PAPER=true) — flip
to live only after the paper loop is proven.

WHY ALPACA, NOT ROBINHOOD: Robinhood has no official trading API. Automating it
means the unofficial robin_stocks library + a stored password = Terms-of-Service
violation, account-ban risk, and your brokerage login sitting in CI. Alpaca is
built for this — REVOCABLE API keys (never a password), within ToS, and it runs
from a cloud cron 24/5 with no machine of yours involved.

Secrets come from GitHub Actions secrets (never hardcoded):
  ALPACA_API_KEY, ALPACA_SECRET_KEY   (+ optional ALPACA_PAPER=false to go live)
"""
from __future__ import annotations

import os
import uuid

import requests

import live
from config import EngineConfig

TICKER = "TQQQ"
BAND = 0.08
PAPER = os.environ.get("ALPACA_PAPER", "true").lower() != "false"
BASE = "https://paper-api.alpaca.markets" if PAPER else "https://api.alpaca.markets"
KEY = os.environ.get("ALPACA_API_KEY", "")
SEC = os.environ.get("ALPACA_SECRET_KEY", "")
HDR = {"APCA-API-KEY-ID": KEY, "APCA-API-SECRET-KEY": SEC}


def _get(path: str) -> requests.Response:
    return requests.get(BASE + path, headers=HDR, timeout=30)


def position_value() -> float:
    r = _get(f"/v2/positions/{TICKER}")
    return float(r.json()["market_value"]) if r.status_code == 200 else 0.0


def run() -> None:
    if not (KEY and SEC):
        raise SystemExit("Set ALPACA_API_KEY / ALPACA_SECRET_KEY as GitHub secrets.")
    mode = "PAPER" if PAPER else "LIVE"
    clock = _get("/v2/clock").json()
    if not clock.get("is_open"):
        print(f"[{mode}] market closed — nothing to do.")
        return

    cfg = EngineConfig()
    target = live.latest_target(TICKER, cfg, refresh=True)["target_weight"]
    equity = float(_get("/v2/account").json()["equity"])
    pos = position_value()
    target_dollars = target * equity
    delta = target_dollars - pos

    print(f"[{mode}] target {target:.1%} (${target_dollars:,.2f}) vs position "
          f"${pos:,.2f} | equity ${equity:,.2f} | delta ${delta:,.2f}")

    if equity <= 0 or abs(delta) / equity < BAND:
        print(f"[{mode}] HOLD — within {BAND:.0%} no-trade band.")
        return

    side = "buy" if delta > 0 else "sell"
    notional = round(min(abs(delta), pos) if side == "sell" else abs(delta), 2)
    body = {"symbol": TICKER, "notional": str(notional), "side": side,
            "type": "market", "time_in_force": "day",
            "client_order_id": str(uuid.uuid4())}
    r = requests.post(BASE + "/v2/orders", headers=HDR, json=body, timeout=30)
    if r.status_code < 300:
        print(f"[{mode}] {side.upper()} ${notional:,.2f} TQQQ — order {r.json().get('id')}")
    else:
        print(f"[{mode}] ORDER FAILED {r.status_code}: {r.text}")


if __name__ == "__main__":
    run()
