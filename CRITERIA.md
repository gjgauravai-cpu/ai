# Strategy Evaluation Rubric (distilled from the quant canon)

Every scanned idea is scored against these gates IN ORDER. Fail one early gate ->
discard without further work. Sources: Grinold & Kahn (Active Portfolio Mgmt),
Lopez de Prado (Advances in Financial ML), McLean & Pontiff (post-publication
decay), Moskowitz/Moreira-Muir (documented premia), Narang (Inside the Black
Box), plus this repo's own findings (see cleared.json and the journals).

## Gate 0 - Structural "why" (Narang)
Who is on the other side and why do they keep paying? Acceptable answers:
risk premium, forced/constrained flows, mechanical product effects (e.g. daily
LETF reset), liquidity provision. "It worked in a backtest" is NOT a why.

## Gate 1 - Decay realism (McLean-Pontiff)
Published/blogged edges lose ~26-58% of returns out of sample. Anything already
viral on retail channels is presumed mostly arbitraged. Score DOWN by recency
and popularity of the source. A crowded idea needs a structural why to survive.

## Gate 2 - Implementability HERE
Must be tradable in a small US cash account (long-only, T+1, fractional shares,
no shorting, no options for now, ~1 name at a time). Ideas needing shorting,
intraday churn, or institutional infrastructure: log as "blocked", do not build.

## Gate 3 - Cost survival (our stat-arb lesson)
Estimate turnover x spread. Our cross-sectional MR test had REAL signal
(p=0.000 gross) and died to 0.01 net Sharpe at 2bps. High-turnover ideas are
dead on arrival at retail cost levels.

## Gate 4 - Overfitting resistance (Lopez de Prado, Bailey)
Count the knobs. Each tunable parameter is a degree of freedom that must be
pre-committed, not optimized. Prefer <= 2 parameters. Any idea whose appeal
comes from a tuned lookback/threshold combo is presumed overfit.

## Gate 5 - Breadth honesty (Grinold-Kahn)
IR ~ IC x sqrt(breadth). One more correlated Nasdaq-beta sleeve adds no breadth
(all four we tested were 0.83-0.93 correlated and failed). Genuine diversifiers
are rare and precious; correlated variants are clutter.

## Final gate - the PROMOTION GATE (mechanical, in this repo)
Implement as a strategy in strategies.py, then: full-cycle backtest since 1999
-> matched-null validation -> gate.evaluate (net Sharpe >= 0.40, p < 0.05,
net CAGR > 0, maxDD > -85%). Run: python promote.py <name>. Only a PASS may
ever be considered for live capital, and switching the LIVE strategy always
requires explicit owner approval - the pipeline proposes, the owner disposes.
