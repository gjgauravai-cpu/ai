# tqqq-signal-bot

A **safe, always-on signal monitor** for the validated TQQQ vol-targeting strategy.

A GitHub Actions cron runs every weekday morning, computes the `vol_target_har_live`
target weight for TQQQ, and posts it as a **GitHub issue** — so you get the daily
target even when your own computer is off.

## What it does (and does NOT do)

- ✅ Computes the daily target weight from public market data (yfinance).
- ✅ Posts the number to you (a GitHub issue → email notification).
- ✅ Runs in the cloud, 24/7, no machine of yours required.
- ❌ Does **NOT** place any trades.
- ❌ Stores **no** brokerage credentials. No secrets needed (uses the built-in
  `GITHUB_TOKEN` only to open the issue).

Execution stays human / MCP-gated: you read the target and place the order
yourself (or let the local Robinhood MCP task do it). This bot only solves
"my machine is off so I don't know today's target."

## Schedule

`.github/workflows/signal.yml` runs `0 13 * * 1-5` (09:00 ET weekdays). Trigger it
manually anytime from the **Actions** tab → *TQQQ daily signal* → *Run workflow*.

## Files

| File | Role |
|------|------|
| `cloud_signal.py` | computes + prints the target (markdown) |
| `config.py`, `data.py`, `models.py`, `strategies.py` | the strategy engine (subset) |
| `.github/workflows/signal.yml` | the weekday cron + issue poster |

## The strategy

`vol_target_har_live`: HAR-RV volatility-targeted exposure to TQQQ, low-turnover
(weekly revisit + 8% no-trade band). Matched-null validated (p=0.03) — the only
leveraged-ETF sleeve with certified timing edge in testing. Research lives in the
parent `leveraged_etf_ts_engine`.

> Not investment advice. Leveraged ETFs carry severe drawdown risk.
