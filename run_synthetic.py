"""Synthetic long-history stress test: reconstruct leveraged ETFs back to the
underlying's inception (QQQ ~1999, SPY ~1993) so the backtest spans the dot-com
crash and 2008 — the prolonged bears the real LETFs never lived through.

This is the honest counter to the survivorship/period bias in run.py: it shows
what a naked 3x hold would have done through a -99% wipeout, and how much the
regime / vol-target overlays change survivability.

Usage:
    python run_synthetic.py                 # QQQ & SPY, 2x and 3x, since 1999
    python run_synthetic.py --underlyings QQQ,SPY,IWM --start 1999-01-01
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402

import data                              # noqa: E402
import letf as letf_mod                  # noqa: E402
import models                            # noqa: E402
import metrics                           # noqa: E402
from backtest import evaluate, net_return_curves   # noqa: E402
from config import LetfSpec, EngineConfig, DEFAULT  # noqa: E402
from strategies import build_weights     # noqa: E402

OUT = Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)
pd.set_option("display.width", 200)
pd.set_option("display.float_format", lambda v: f"{v:,.4f}")

# Representative expense/spread for synthetic 2x and 3x sleeves.
SYNTH = {2.0: dict(er=0.0092, spread=4.0), 3.0: dict(er=0.0093, spread=5.0)}


def synth_spec(underlying: str, lev: float) -> LetfSpec:
    p = SYNTH[lev]
    return LetfSpec(f"{underlying}x{int(lev)}", lev, underlying, p["er"], p["spread"],
                    f"synthetic {int(lev)}x {underlying}")


def run_one(underlying: str, lev: float, und_df: pd.DataFrame, rf: pd.Series,
            cfg: EngineConfig, strategies: list[str],
            vix_close: pd.Series | None = None) -> dict:
    spec = synth_spec(underlying, lev)
    und_ret = data.daily_returns(und_df).dropna()
    letf_ret = letf_mod.synthetic_letf_returns(und_ret, spec, rf,
                                               cfg.financing_spread, cfg.trading_days)
    letf_ret = letf_ret.dropna()
    decomp = letf_mod.decompose(und_ret, spec, rf, cfg.financing_spread, cfg.trading_days)
    # no LETF OHLC for a synthetic sleeve: HAR off the underlying, scaled by L
    har_l = models.har_vol(und_df, cfg.garch_refit_every,
                           cfg.garch_min_obs, cfg.trading_days) * lev
    iv_blended = (models.iv_floored_vol(har_l, vix_close, lev, cfg.iv_floor_beta,
                                        cfg.trading_days)
                  if vix_close is not None else har_l)
    ctx = {
        "letf_ret": letf_ret,
        "underlying_close": und_df["close"],
        "garch_vol": models.garch_vol(letf_ret, cfg.garch_refit_every,
                                      cfg.garch_min_obs, cfg.trading_days),
        "gjr_vol": models.gjr_vol(letf_ret, cfg.garch_refit_every,
                                  cfg.garch_min_obs, cfg.trading_days),
        "har_vol": har_l,
        "iv_blended_vol": iv_blended,
        "tsmom": models.tsmom_signal(und_df["close"], cfg.sma_window),
        "ar1": models.rolling_ar1(und_ret, cfg.autocorr_window),
        "regime_sma": models.sma_regime(und_df["close"], cfg.sma_window),
        "regime_hmm": pd.Series(1.0, index=letf_ret.index),
    }
    weights = {n: build_weights(n, ctx, cfg) for n in strategies}
    table = evaluate(letf_ret, weights, spec, cfg, rf)
    curves = net_return_curves(letf_ret, weights, spec, cfg, rf)
    trial_sharpes = list(table["Sharpe_net"].values)
    table["DSR"] = pd.Series({
        name: metrics.deflated_sharpe(curves[name].pct_change(), trial_sharpes,
                                      cfg.trading_days)
        for name in table.index if name in curves
    })
    return {"spec": spec, "decomp": decomp, "table": table, "curves": curves,
            "weights": weights, "letf_ret": letf_ret, "und_df": und_df}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--underlyings", default="QQQ,SPY")
    ap.add_argument("--levs", default="2,3")
    ap.add_argument("--start", default="1999-01-01")
    ap.add_argument("--strategies",
                    default="buy_hold,vol_target,vol_target_har,vol_target_gjr,"
                            "vol_target_armod,vol_target_ivhar,vol_target_tsmom")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--validate", action="store_true",
                    help="run the exposure/turnover-matched-null + regime-split harness")
    args = ap.parse_args()

    cfg = EngineConfig(start=args.start)
    unders = [u.strip().upper() for u in args.underlyings.split(",")]
    levs = [float(x) for x in args.levs.split(",")]
    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]

    prices = data.load_prices(unders + [cfg.vix_symbol], cfg.start, None,
                              refresh=args.refresh)
    rf = data.risk_free_series(cfg.start, None, cfg.fallback_rf)
    vix_close = prices[cfg.vix_symbol]["close"] if cfg.vix_symbol in prices else None

    results, decay_rows = {}, []
    for u in unders:
        if u not in prices or u == cfg.vix_symbol:
            continue
        for L in levs:
            r = run_one(u, L, prices[u], rf, cfg, strategies, vix_close)
            key = f"{u} {int(L)}x"
            results[key] = r
            d = r["decomp"]
            decay_rows.append({
                "Synthetic": key, "Years": d["years"],
                "IdxVol%": d["ann_index_vol"] * 100,
                "Naive_Lx_%": d["naive_Lx_total_%"], "Modeled_%": d["modeled_total_%"],
                "DecayDrag%": d["decay+cost_drag_%"],
                "VarDrag/yr%": d["var_drag_log"] / d["years"] * 100,
            })

    print("\n" + "=" * 100)
    print(f"SYNTHETIC DECAY DECOMPOSITION since {args.start} (through dot-com + 2008)")
    print("=" * 100)
    print(pd.DataFrame(decay_rows).set_index("Synthetic").to_string())

    for key, r in results.items():
        print("\n" + "-" * 100)
        print(f"{key}  — synthetic strategies, NET (spans 2000-02 & 2008 bears)")
        print("-" * 100)
        print(r["table"].to_string())

    # plot the marquee case: 3x first underlying, buy&hold vs managed, log scale
    mk = f"{unders[0]} 3x"
    if mk in results:
        c = results[mk]["curves"]
        fig, ax = plt.subplots(figsize=(11, 6))
        for col in ("buy_hold", "regime_sma", "vol_target_regime"):
            if col in c:
                ax.plot(c.index, c[col], label=col, lw=1.4)
        ax.set_yscale("log"); ax.grid(alpha=0.3); ax.legend()
        ax.set_title(f"${int(cfg.starting_capital)} in synthetic {mk} since {args.start} "
                     f"(log) — naked hold vs managed")
        ax.set_ylabel("Account value ($, log)")
        p = OUT / "synthetic_3x_stress.png"; fig.tight_layout(); fig.savefig(p, dpi=130)
        plt.close(fig); print(f"\n[plot] {p}")

    if args.validate:
        import validate
        for key, r in results.items():
            vdf = validate.validate_strategies(
                r["letf_ret"], r["weights"], r["spec"], cfg, rf,
                r["und_df"]["close"], n_rotations=400, split_date="2020-01-01")
            print("\n" + "=" * 100)
            print(f"VALIDATION — {key}: matched-null (rotation) p-value + regime-split sign test")
            print("=" * 100)
            cols = ["real_sharpe", "null_p95", "p_value", "xsSR_pre2020",
                    "xsSR_post2020", "xsSR_hivol", "xsSR_lovol", "sign_consistent", "verdict"]
            print(vdf[cols].to_string())
            vdf.to_csv(OUT / f"validation_{key.replace(' ', '_')}.csv")

    pd.DataFrame(decay_rows).to_csv(OUT / "synthetic_decay.csv", index=False)
    print(f"Artifacts in {OUT}")


if __name__ == "__main__":
    main()
