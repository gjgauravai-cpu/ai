"""Cloud signal bot — compute today's vol_target_har_live target for TQQQ and
print a markdown report.

Runs in GitHub Actions on a weekday cron (no credentials, NO trading). The
workflow posts this output as a GitHub issue so the daily target reaches you even
when your own machine is off. Execution stays human / MCP-gated — this only tells
you what the validated strategy wants to hold.
"""
from __future__ import annotations

import datetime as dt
import sys

import data
import models
from config import UNIVERSE, EngineConfig
from strategies import build_weights

TICKER = "TQQQ"
STRATEGY = "vol_target_har_live"


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")     # emojis on Windows + Linux
    except Exception:                                # noqa: BLE001
        pass
    cfg = EngineConfig()
    spec = UNIVERSE[TICKER]
    letf_df = data.load_one(TICKER, cfg.start, None, refresh=True)
    und_df = data.load_one(spec.underlying, cfg.start, None, refresh=True)
    letf_ret = data.daily_returns(letf_df).dropna()

    ctx = {
        "letf_ret": letf_ret,
        "letf_df": letf_df,
        "underlying_close": und_df["close"],
        "har_vol": models.har_vol(letf_df, cfg.garch_refit_every,
                                  cfg.garch_min_obs, cfg.trading_days),
    }
    w = build_weights(STRATEGY, ctx, cfg).dropna()
    tw = float(w.iloc[-1])
    har = float(ctx["har_vol"].dropna().iloc[-1])
    price = float(letf_df["close"].iloc[-1])
    asof = str(w.index[-1].date())
    today = dt.datetime.utcnow().date().isoformat()

    print(f"## 📡 {TICKER} target — generated {today} (data as of {asof})")
    print()
    print(f"- **Target weight: {tw:.1%}** of account in {TICKER}")
    print(f"- HAR volatility (annualized): {har:.1%}")
    print(f"- Last close: ${price:,.2f}")
    print(f"- On a $100 account → hold ~**${tw * 100:,.2f}** {TICKER}, rest in cash")
    print(f"- On a $200 account → hold ~**${tw * 200:,.2f}** {TICKER}, rest in cash")
    print()
    print(f"Strategy: `{STRATEGY}` (matched-null validated, p=0.03). "
          f"Rebalance only if your current weight is off target by >8%.")
    print()
    print("> ⚠️ MONITOR ONLY — no order was placed. Execute via your broker / the "
          "MCP daily task. This bot just delivers the number when your machine is off.")


if __name__ == "__main__":
    main()
