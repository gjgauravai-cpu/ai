# Research Agenda (internal, pre-vetted backlog)

The monthly board works TOP-DOWN through this list when ideas.md has no
gate-worthy web idea. One item per month MAX. promote.py is the judge. Record
the verdict + date in the table below after testing. NEVER tune-to-pass.

## Untested candidates (ranked by a 13-agent review panel, Jun 2026)

| # | Candidate | Spec + causality warnings | Verdict |
|---|---|---|---|
| 1 | Drawdown-throttle overlay | Hysteretic tiers 1.0/0.7/0.5/0.0 on the base rule's SHIFTED equity-proxy drawdown; tiers -15/-30/-45%, 5pp hysteresis. Implemented as vol_target_har_live_dd (causality unit-tested). | **FAIL 2026-08-18** - real-data gate: Sharpe 0.404, CAGR 11.2%, maxDD -41.9%, but matched-null p=0.177 (timing not distinguishable from random de-risking); 7/8 checks passed, killed by the null test. DD-vs-CAGR trade only, no edge. |
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

## Board notes
- 2026-08-18: Cloud board implemented item #1 but was blocked (sandbox: all
  market-data hosts 403, repo write 403) AND found promote.py had been crashing
  since c343884 (gate.evaluate lacked additional_checks). Both fixed in this
  commit: gate.py now accepts additional_checks (walkforward + cost-stress
  checks active again), and the REAL gate was run locally with real data ->
  verdict recorded above. promote.yml added so future gate runs can be
  dispatched in CI, where market data IS reachable.

## Literature review: HFT / microstructure corpus (2026-08-20)

21 papers now in `agentic_trading/quant_library/` (LOB dynamics, Hawkes, market
impact, optimal execution, deep-LOB, SEC reports). Reviewed against CRITERIA.md
gate 2 (implementable HERE). **Verdict: the HFT/LOB branch is NOT implementable
in this account and the board should not spend cycles on it.** Reasons, concrete:

- DeepLOB / queue-reactive / Hawkes-microstructure / optimal market making all
  require Level-2 order-book data, sub-second execution, and shorting. This
  account has daily bars, T+1 cash settlement, long-only, fractional shares.
- Our own intraday test already settled the horizon question empirically:
  intraday session Sharpe 0.41 vs overnight 0.96, no intraday momentum
  (hourly autocorr -0.115), and ~1500 trades/yr of cost. See REJECTED list.
- Market-impact papers (Almgren-Chriss, Toth, Lillo, Donier) matter at size.
  At this account size the order is a rounding error - impact is ~0, so the
  optimal-execution machinery has nothing to optimise. Revisit only if capital
  grows enough that a single order moves the book (not remotely the case).

What the corpus DOES earn its place for: it explains the MECHANISM behind
findings we already made empirically (why high-turnover strategies die to costs;
why volatility clusters, which is what HAR already exploits). Education and
execution intuition, not a new edge.

Gate-testable candidates extracted (queued BELOW existing agenda items 2-6,
which have higher priors):
| 8 | Deep-RL daily position sizing (Zhang/Zohren/Roberts 1911.10107) | Their daily-frequency formulation is retail-feasible in principle. HIGH overfit risk (ML on ~6k daily obs); must clear matched-null AND walk-forward, and be compared against vol_target_har_live, not buy&hold. | UNTESTED - LOW PRIOR |
