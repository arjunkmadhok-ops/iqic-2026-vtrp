#!/usr/bin/env python3
"""Replay every historical 3-week IQIC-style contest under the LIVE StockTrak rules.

Each contest: start with $100,000 cash after a Friday close; build on Monday; rebalance on the
next two Mondays if the 2-point band is breached; hold to the 15th trading day. Unlike the
long-run backtest, the circuit breaker uses the account's peak since the contest started,
positions are whole shares, uninvested cash earns nothing, and every trade costs 10 bp.
Trades are modelled at the Monday close (StockTrak limit orders will fill at other prices).

  python contest_sim.py --prices prices.csv
"""
import argparse

import numpy as np
import pandas as pd

import vtrp

CAPITAL = 100_000.0


def contest(px, rets, d0, days, targets, p, use_breaker=True, cost=0.001):
    """Simulate one contest. d0 = signal Friday; days = 15 trading days after it."""
    cols = vtrp.RISKY + [vtrp.CASH]
    shares = pd.Series(0.0, index=cols)
    cash = CAPITAL
    navs, peak, prev = [], CAPITAL, CAPITAL
    week_ends = {d for d in days if d in targets}
    pending = targets[d0]
    tripped = False
    for i, t in enumerate(days):
        price = px.loc[t, cols]
        if pending is not None:                                   # Monday: trade at close
            nav = cash + float((shares * price.fillna(0)).sum())
            cur_w = shares * price.fillna(0) / nav
            if i == 0 or float((pending - cur_w).abs().max()) >= p["band"]:
                tgt = np.floor(nav * pending / price).fillna(0)
                traded = float(((tgt - shares).abs() * price.fillna(0)).sum())
                cash = nav - float((tgt * price.fillna(0)).sum()) - traded * cost
                shares = tgt
            pending = None
        nav = cash + float((shares * price.fillna(0)).sum())
        navs.append(nav)
        peak = max(peak, nav)
        if t in week_ends and i < len(days) - 1:                  # Friday: next week's targets
            w = targets[t]
            if use_breaker and nav / peak - 1 < -p["breaker_dd"]:
                w = vtrp.apply_breaker(w, nav / peak - 1, p)
                tripped = True
            pending = w
    nav = pd.Series(navs, index=days)
    r = nav.pct_change()
    r.iloc[0] = nav.iloc[0] / CAPITAL - 1
    return r, tripped


def metrics(r, m, rf):
    ex, exm = r - rf, m - rf
    beta = np.cov(r, m)[0, 1] / m.var() if m.var() > 0 else np.nan
    return dict(ret=(1 + r).prod() - 1,
                sharpe=ex.mean() / r.std() * np.sqrt(252) if r.std() > 0 else np.nan,
                alpha=(ex.mean() - beta * exm.mean()) * 252,
                treynor=ex.mean() * 252 / beta if abs(beta) > 0.05 else np.nan)


def run(px, start="2005-08-01", target_vol=0.10, use_breaker=True, cost=0.001):
    p = {**vtrp.PARAMS, "target_vol": target_vol}
    rets = px.pct_change(fill_method=None)
    idx = px.index
    week_end = sorted(set(pd.Series(idx, index=idx).groupby(idx.to_period("W-FRI")).max()))
    week_end = [d for d in week_end if d >= pd.Timestamp(start)]
    targets = {}
    for d in week_end:
        loc = idx.get_loc(d)
        if loc >= p["min_history"] + 5:
            targets[d], _ = vtrp.risk_parity_targets(px, loc, p)
    rows = []
    for d0 in week_end:
        loc = idx.get_loc(d0)
        days = idx[loc + 1: loc + 16]
        if len(days) < 15 or d0 not in targets:
            continue
        r, tripped = contest(px, rets, d0, days, targets, p, use_breaker, cost)
        m = rets.loc[days, vtrp.BENCH]
        rf = rets.loc[days, vtrp.CASH].fillna(0)
        s, b = metrics(r, m, rf), metrics(m, m, rf)
        rows.append(dict(start=days[0], breaker=tripped, **{f"s_{k}": v for k, v in s.items()},
                         **{f"spy_{k}": v for k, v in b.items()}))
    return pd.DataFrame(rows).set_index("start")


def summarise(R):
    down = R.spy_ret < 0
    return dict(
        contests=len(R),
        alpha_pos=(R.s_alpha > 0).mean(),
        sharpe_beats_spy=(R.s_sharpe > R.spy_sharpe).mean(),
        treynor_beats_spy=(R.s_treynor > R.spy_treynor).mean(),
        mean_ret=R.s_ret.mean(), spy_mean_ret=R.spy_ret.mean(),
        loss_ratio_when_spy_down=R.s_ret[down].mean() / R.spy_ret[down].mean(),
        p5_ret=R.s_ret.quantile(0.05), spy_p5_ret=R.spy_ret.quantile(0.05),
        breaker_share=R.breaker.mean(),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prices", required=True)
    args = ap.parse_args()
    px = vtrp.load_prices(args.prices)
    out = {}
    R = run(px)
    for name, (a, b) in {"2005-2015": ("2005", "2015"), "2016-2026": ("2016", "2026")}.items():
        out[("declared rules", name)] = summarise(R.loc[a:b])
    R.to_csv("contests.csv")
    for label, kw in {"no breaker": dict(use_breaker=False),
                      "20 bp costs": dict(cost=0.002),
                      "5 bp costs": dict(cost=0.0005),
                      "2 bp costs": dict(cost=0.0002),
                      "0 bp costs": dict(cost=0.0),
                      "vol target 8%": dict(target_vol=0.08),
                      "vol target 12%": dict(target_vol=0.12),
                      "vol target 14%": dict(target_vol=0.14)}.items():
        X = run(px, **kw)
        for name, (a, b) in {"2005-2015": ("2005", "2015"), "2016-2026": ("2016", "2026")}.items():
            out[(label, name)] = summarise(X.loc[a:b])
    pd.set_option("display.width", 200)
    print(pd.DataFrame(out).T.round(3).to_string())


if __name__ == "__main__":
    main()
