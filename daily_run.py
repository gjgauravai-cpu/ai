"""Daily cloud run — the single entry point that executes online every weekday.

Runs in GitHub Actions (machine-off). It ALWAYS does two things so the bot can
never be "silently dead":
  1. Computes the validated vol_target_har_live target for TQQQ and reports it.
  2. Reports whether it is able to trade, and why not if it cannot.

If Alpaca API keys are present it also REBALANCES via Alpaca's official API
(paper by default). Without keys it still reports the target and states loudly
that no broker is wired — the failure is visible, never silent.

Robinhood is deliberately NOT automated here: it has no official trading API, so
cloud automation would mean an unofficial library plus a stored password, which
violates their terms and risks the account.
"""
from __future__ import annotations

import os
import sys
import uuid

import requests

import data
import gate
import models
from config import UNIVERSE, EngineConfig
from strategies import build_weights

TICKER = "TQQQ"
STRATEGY = "vol_target_har_live"
BAND = 0.08                       # no-trade band, matches the validated backtest

PAPER = os.environ.get("ALPACA_PAPER", "true").lower() != "false"
KEY = os.environ.get("ALPACA_API_KEY", "").strip()
SEC = os.environ.get("ALPACA_SECRET_KEY", "").strip()
BASE = "https://paper-api.alpaca.markets" if PAPER else "https://api.alpaca.markets"
HDR = {"APCA-API-KEY-ID": KEY, "APCA-API-SECRET-KEY": SEC}


def compute_target() -> dict:
    """Causal target weight from the gate-cleared strategy."""
    gate.assert_cleared(STRATEGY)          # fail-closed: uncleared strategy cannot run
    cfg = EngineConfig()
    spec = UNIVERSE[TICKER]
    letf_df = data.load_one(TICKER, cfg.start, None, refresh=True)
    und_df = data.load_one(spec.underlying, cfg.start, None, refresh=True)
    ctx = {
        "letf_ret": data.daily_returns(letf_df).dropna(),
        "letf_df": letf_df,
        "underlying_close": und_df["close"],
        "har_vol": models.har_vol(letf_df, cfg.garch_refit_every,
                                  cfg.garch_min_obs, cfg.trading_days),
    }
    w = build_weights(STRATEGY, ctx, cfg).dropna()
    return {
        "weight": float(w.iloc[-1]),
        "har_vol": float(ctx["har_vol"].dropna().iloc[-1]),
        "price": float(letf_df["close"].iloc[-1]),
        "asof": str(w.index[-1].date()),
    }


def _get(path: str) -> requests.Response:
    return requests.get(BASE + path, headers=HDR, timeout=30)


def trade_via_alpaca(target_weight: float, out: list[str]) -> None:
    """Rebalance to the target weight. Appends a report to `out`."""
    mode = "PAPER" if PAPER else "LIVE"
    clock = _get("/v2/clock")
    if clock.status_code != 200:
        out.append(f"- ❌ Alpaca auth/API error ({clock.status_code}). Check the secrets.")
        return
    if not clock.json().get("is_open"):
        out.append(f"- 🕒 [{mode}] Market closed at run time — no action (correct).")
        return

    acct = _get("/v2/account").json()
    equity = float(acct["equity"])
    pos_r = _get(f"/v2/positions/{TICKER}")
    pos = float(pos_r.json()["market_value"]) if pos_r.status_code == 200 else 0.0
    target_dollars = target_weight * equity
    delta = target_dollars - pos

    out.append(f"- [{mode}] equity **${equity:,.2f}** | {TICKER} **${pos:,.2f}** "
               f"({pos / equity:.1%}) | target **${target_dollars:,.2f}** "
               f"({target_weight:.1%}) | delta **${delta:,.2f}**")

    if equity <= 0 or abs(delta) / equity < BAND:
        out.append(f"- ✅ **HOLD** — inside the {BAND:.0%} no-trade band. No order placed.")
        return

    side = "buy" if delta > 0 else "sell"
    notional = round(min(abs(delta), pos) if side == "sell" else abs(delta), 2)
    body = {"symbol": TICKER, "notional": str(notional), "side": side,
            "type": "market", "time_in_force": "day",
            "client_order_id": str(uuid.uuid4())}
    r = requests.post(BASE + "/v2/orders", headers=HDR, json=body, timeout=30)
    if r.status_code < 300:
        out.append(f"- 🟢 **{side.upper()} ${notional:,.2f} {TICKER}** placed "
                   f"(order `{r.json().get('id')}`).")
    else:
        out.append(f"- ❌ Order rejected {r.status_code}: `{r.text[:300]}`")


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:                       # noqa: BLE001
        pass

    out: list[str] = []
    t = compute_target()
    out.append(f"**Target weight: {t['weight']:.1%}** of account in {TICKER}")
    out.append("")
    out.append(f"- HAR volatility (annualized): {t['har_vol']:.1%}")
    out.append(f"- Last close: ${t['price']:,.2f}  (data as of {t['asof']})")
    out.append(f"- Strategy `{STRATEGY}` — matched-null validated (p=0.03), "
               f"rebalances only outside a {BAND:.0%} band")
    out.append("")

    if KEY and SEC:
        out.append("### Execution (Alpaca, official API)")
        try:
            trade_via_alpaca(t["weight"], out)
        except Exception as exc:            # noqa: BLE001 - surface, never swallow
            out.append(f"- ❌ Execution error: `{exc}`")
    else:
        out.append("### ⚠️ NOT TRADING — no broker wired")
        out.append("")
        out.append("This run computed the target but **placed no order**, because no "
                   "Alpaca API keys are configured.")
        out.append("")
        out.append("To make it trade itself online every day:")
        out.append("1. Create a free Alpaca account and switch to **Paper**.")
        out.append("2. Generate an API key + secret.")
        out.append("3. Repo → Settings → Secrets and variables → Actions → add "
                   "`ALPACA_API_KEY` and `ALPACA_SECRET_KEY`.")
        out.append("")
        out.append("Until then this is a signal-only bot. (Robinhood is not automated "
                   "here on purpose: no official API means an unofficial library plus a "
                   "stored password, which breaches their terms and risks the account.)")

    print("\n".join(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
