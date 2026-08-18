# Research Agenda (internal, pre-vetted backlog)

The monthly board works TOP-DOWN through this list when ideas.md has no
gate-worthy web idea. One item per month MAX. promote.py is the judge. Record
the verdict + date in the table below after testing. NEVER tune-to-pass.

## Untested candidates (ranked by a 13-agent review panel, Jun 2026)

| # | Candidate | Spec + causality warnings | Verdict |
|---|---|---|---|
| 1 | Drawdown-throttle overlay | Hysteretic tiers 1.0/0.7/0.5/0.0 applied to vol_target_har_live by trailing drawdown of the BASE strategy's own equity proxy. CAUSALITY: equity proxy = cumprod of (base_w * letf_ret) SHIFTED BY 1 DAY before cummax/drawdown - using same-day data is the look-ahead trap the panel flagged. Pre-commit tiers at -15/-30/-45%. | UNTESTED |
| 2 | VIX term-structure gate | ^VIX/^VIX3M backwardation as soft de-leverage multiplier; causal rolling-quantile threshold (never a hardcoded 1.0); must show INCREMENTAL Calmar over regime_sma. | UNTESTED |
| 3 | Markov-switching regime (Hamilton) | statsmodels MarkovRegression, 2-state, walk-forward refit; replaces 200-DMA as hold/cash. | UNTESTED |
| 4 | Fractional-Kelly ceiling | w = min(vol_target_w, kelly_cap) from slow EWMA drift / slow var; CEILING-ONLY (never increases exposure). | UNTESTED |
| 5 | CVaR throttle | scale = clip(cvar_budget/|trailing 63d 5% CVaR|, 0, 1), smooth clip; overlaps vol-target - must show incremental maxDD benefit. | UNTESTED |
| 6 | BOCPD changepoint floor | Adams-MacKay, hazard 1/250 fixed a priori, de-risk floor 0.5, standalone before any stacking. | UNTESTED |
| 7 | HARQ / realized-GARCH | ONLY if the weekly review flags HAR-vs-realized calibration bias > 15%. | GATED-ON-FLAG |

## REJECTED - do not retest without new data or a regime change
GJR-GARCH-t (p=0.165, worst of family) | IV-floored HAR (neutral after warmup fix)
| sector stat-arb MR (real signal, net Sharpe 0.01 at 2bps) | 9-asset dual momentum
(p=0.243, corr +0.66 to live sleeve) | any additional LETF sleeve (corr 0.83-0.93,
all failed matched-null) | intraday trading (session Sharpe 0.41 vs 0.96 overnight)
| 4-signal combiners / regime routers (overfit trap per panel).
