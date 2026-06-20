"""Live deployment loop: compute today's target weight for the validated
strategy (vol_target_har_live) and size the rebalancing order for the funded
Robinhood agentic account.

This module does DATA + MODEL + ORDER-MATH only. It NEVER places an order — by
design. Order submission stays a human-gated, reviewed step outside this file.

Usage:
    python live.py --ticker TQQQ                              # just the target weight
    python live.py --ticker TQQQ --account-value 100 \
                   --position-value 0 --price 77.5            # full order math
"""
from __future__ import annotations

import argparse
import json

import data
import models
from config import UNIVERSE, EngineConfig, DEFAULT
from strategies import build_weights

STRATEGY = "vol_target_har_live"          # the validated, low-turnover deployable strategy


def latest_target(ticker: str, cfg: EngineConfig, refresh: bool = False) -> dict:
    """Compute the current target weight from causal history (no account state)."""
    if ticker not in UNIVERSE:
        raise SystemExit(f"Unknown ticker '{ticker}'. Known: {list(UNIVERSE)}")
    spec = UNIVERSE[ticker]
    letf_df = data.load_one(ticker, cfg.start, None, refresh=refresh)
    und_df = data.load_one(spec.underlying, cfg.start, None, refresh=refresh)
    letf_ret = data.daily_returns(letf_df).dropna()

    ctx = {
        "letf_ret": letf_ret,
        "letf_df": letf_df,
        "underlying_close": und_df["close"],
        "har_vol": models.har_vol(letf_df, cfg.garch_refit_every,
                                  cfg.garch_min_obs, cfg.trading_days),
    }
    w = build_weights(STRATEGY, ctx, cfg).dropna()
    har = ctx["har_vol"].dropna()
    return {
        "ticker": ticker,
        "strategy": STRATEGY,
        "as_of": str(w.index[-1].date()),
        "last_close": round(float(letf_df["close"].iloc[-1]), 4),
        "har_vol_annual": round(float(har.iloc[-1]), 4),
        "target_vol": cfg.target_vol,
        "target_weight": round(float(w.iloc[-1]), 4),
    }


def build_order(target_weight: float, account_value: float, position_value: float,
                price: float, band: float) -> dict:
    """Size the rebalancing trade; HOLD if inside the no-trade band."""
    target_dollars = target_weight * account_value
    delta = target_dollars - position_value
    within_band = account_value > 0 and abs(delta) / account_value < band
    order = {
        "target_weight": round(target_weight, 4),
        "target_dollars": round(target_dollars, 2),
        "current_dollars": round(position_value, 2),
        "delta_dollars": round(delta, 2),
        "no_trade_band": band,
    }
    if within_band:
        order.update(action="HOLD",
                     reason=f"|delta| {abs(delta)/account_value:.1%} < band {band:.0%}")
        return order
    side = "buy" if delta > 0 else "sell"
    order.update(action=side.upper(), side=side,
                 notional=round(abs(delta), 2),
                 shares=round(abs(delta) / price, 6),
                 price=round(price, 4))
    return order


def main() -> None:
    ap = argparse.ArgumentParser(description="Live target + order sizing (no placement)")
    ap.add_argument("--ticker", default="TQQQ")
    ap.add_argument("--account-value", type=float, default=None)
    ap.add_argument("--position-value", type=float, default=0.0)
    ap.add_argument("--price", type=float, default=None, help="live price; default=last close")
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    cfg = EngineConfig()
    tgt = latest_target(args.ticker, cfg, refresh=args.refresh)
    print(json.dumps({"target": tgt}, indent=2))

    if args.account_value is not None:
        price = args.price if args.price is not None else tgt["last_close"]
        order = build_order(tgt["target_weight"], args.account_value,
                            args.position_value, price, cfg.live_band)
        print(json.dumps({"order": order}, indent=2))
        print("\n*** This is a PROPOSED order. No order has been placed. "
              "Human review + explicit approval required before submission. ***")


if __name__ == "__main__":
    main()
