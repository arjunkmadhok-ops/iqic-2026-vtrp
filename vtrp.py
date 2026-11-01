#!/usr/bin/env python3
"""IQIC 2026 - Volatility-Targeted Risk Parity (VTRP).

The strategy declared in our IQIC Strategy Declaration:

  1. Volatility:   sigma_i = std of the last 63 daily returns x sqrt(252)
  2. Risk parity:  w_i = (1/sigma_i) / sum_j(1/sigma_j), capped at 25% per ETF
                   (excess spread pro rata over the uncapped ETFs)
  3. Vol target:   sigma_p = sqrt(w' Sigma w) from the 63-day covariance matrix;
                   k = min(1, 10% / sigma_p)  (never leverage)
  4. Breaker:      c = 0.5 if the portfolio is more than 8% below its peak at a
                   Friday close, otherwise c = 1. Final targets = c * k * w_i;
                   SHY holds the rest.
  5. Band:         trade only if some weight is >= 2 percentage points from target;
                   then rebalance every position.

Signals use the last close of each week and are executed at the NEXT trading
day's close in the backtest. Costs are charged per unit of value traded.

Usage:
  python vtrp.py --prices prices.csv                  # full backtest + statistics
  python vtrp.py --prices prices.csv --weights-only   # target weights at last close

prices.csv: Date column + one column per ticker of split/dividend-adjusted closes
(build it with build_prices.py).
"""
import argparse

import numpy as np
import pandas as pd

RISKY = ["XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY", "IEF", "GLD"]
CASH = "SHY"
BENCH = "SPY"

PARAMS = dict(
    vol_window=63,      # days of returns for volatility and covariance
    min_history=131,    # an ETF joins once it has ~6 months of prices
    cap=0.25,           # max weight per ETF
    target_vol=0.10,    # annualised volatility target
    breaker_dd=0.08,    # drawdown that triggers the circuit breaker
    breaker_cut=0.5,    # c when the breaker is on
    band=0.02,          # rebalance band (2 percentage points)
    cost=0.0005,        # cost per unit of value traded (5 bp)
)


def load_prices(path):
    px = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    return px.apply(pd.to_numeric, errors="coerce")


def cap_weights(ws, cap):
    """Cap each weight at `cap`, spreading the excess pro rata over uncapped names."""
    ws = ws.copy()
    for _ in range(10):
        over = ws > cap
        if not over.any():
            break
        excess = (ws[over] - cap).sum()
        ws[over] = cap
        under = ~over
        ws[under] += excess * ws[under] / ws[under].sum()
    return ws.clip(upper=cap)


def risk_parity_targets(px, loc, p=PARAMS, cap=None):
    """Rules 1-3 using data up to and including row `loc` of px. Returns weights incl. SHY."""
    cap = p["cap"] if cap is None else cap
    rets = px.iloc[loc - p["vol_window"]: loc + 1].pct_change(fill_method=None)
    window = rets.iloc[1:]                       # the last vol_window daily returns
    vol = window.std() * np.sqrt(252)
    eligible = [a for a in RISKY
                if a in px and not np.isnan(px[a].iloc[loc - p["min_history"] + 5])
                and not window[a].isna().any()]
    w = pd.Series(0.0, index=RISKY + [CASH])
    if not eligible:
        w[CASH] = 1.0
        return w, dict(pred_vol=np.nan, k=0.0)
    iv = 1 / vol[eligible]
    ws = cap_weights(iv / iv.sum(), cap)
    cov = window[eligible].cov() * 252
    pvol = float(np.sqrt(ws.values @ cov.values @ ws.values))
    k = min(1.0, p["target_vol"] / pvol, 1.0 / ws.sum())
    w[eligible] = ws * k
    w[CASH] = 1 - w[RISKY].sum()
    return w, dict(pred_vol=pvol, k=k, vol=vol[eligible], cov=cov)


def cap_cash_sleeve(w, class_cap):
    """StockTrak position limits apply to SHY too. Cap SHY at the class limit; the
    remainder is left uninvested as cash. Returns (weights, cash weight)."""
    w = w.copy()
    cash = max(0.0, float(w[CASH]) - class_cap)
    w[CASH] = min(float(w[CASH]), class_cap)
    return w, cash


def risk_contributions(w, cov):
    """Share of portfolio volatility from each asset: RC_i = w_i (Sigma w)_i / (w' Sigma w).

    Shares sum to 1. Inverse-volatility weights equalise RC_i only if all correlations
    are equal, so the spread of these shares measures how far the portfolio is from
    exact equal-risk-contribution risk parity.
    """
    names = list(cov.index)
    wv = w[names].values
    sw = cov.values @ wv
    total = float(wv @ sw)
    return pd.Series(wv * sw / total, index=names)


def apply_breaker(w, drawdown, p=PARAMS):
    """Rule 4: halve risky weights if drawdown is worse than -breaker_dd."""
    if drawdown < -p["breaker_dd"]:
        w = w.copy()
        w[RISKY] *= p["breaker_cut"]
        w[CASH] = 1 - w[RISKY].sum()
    return w


def backtest(px, start="2005-01-03", p=PARAMS):
    """Daily simulation. Returns (daily strategy returns, daily weights, turnover per trade)."""
    rets = px.pct_change(fill_method=None)
    cols = RISKY + [CASH]
    idx = px.index[px.index >= pd.Timestamp(start)]
    week_end = set(pd.Series(px.index, index=px.index).groupby(px.index.to_period("W-FRI")).max())
    cur = pd.Series(0.0, index=cols)
    cur[CASH] = 1.0
    pending = None
    out_r, out_w, turns = [], [], []
    for t in idx:
        r = rets.loc[t, cols].fillna(0.0)
        day_r = float((cur * r).sum())
        grown = cur * (1 + r)
        cur = grown / grown.sum()
        cost = 0.0
        if pending is not None:                      # execute at today's close
            tw = pending
            if float((tw - cur).abs().max()) < p["band"]:
                tw = cur                             # rule 5: inside the band, no trade
            to = float((tw - cur).abs().sum())
            cost = to * p["cost"]
            turns.append((t, to))
            cur = tw
            pending = None
        out_r.append(day_r - cost)
        out_w.append(cur.copy())
        if t in week_end:                            # signal at the week's last close
            loc = px.index.get_loc(t)
            if loc >= p["min_history"] + 5:
                w, _ = risk_parity_targets(px, loc, p)
                eq = np.cumprod(1 + np.array(out_r[-63:]))
                dd = eq[-1] / max(1.0, eq.max()) - 1  # backtest peak: past 63 trading days
                pending = apply_breaker(w, dd, p)
    return (pd.Series(out_r, index=idx, name="vtrp"), pd.DataFrame(out_w, index=idx),
            pd.Series(dict(turns), dtype=float))


def stats(r, b, rf):
    """Annualised statistics of daily returns r vs benchmark b with daily risk-free rf."""
    ex, exb = r - rf, b - rf
    beta = np.cov(r, b)[0, 1] / b.var()
    eq = (1 + r).cumprod()
    return dict(
        CAGR=(1 + r).prod() ** (252 / len(r)) - 1,
        Vol=r.std() * np.sqrt(252),
        Sharpe=ex.mean() / r.std() * np.sqrt(252),
        Beta=beta,
        Alpha=(ex.mean() - beta * exb.mean()) * 252,
        Treynor=ex.mean() * 252 / beta,
        MaxDD=(eq / eq.cummax() - 1).min(),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prices", required=True)
    ap.add_argument("--weights-only", action="store_true")
    ap.add_argument("--cost-bp", type=float, default=5.0)
    args = ap.parse_args()
    px = load_prices(args.prices)
    p = {**PARAMS, "cost": args.cost_bp / 1e4}
    if args.weights_only:
        w, info = risk_parity_targets(px, len(px) - 1, p)
        rc = risk_contributions(w, info["cov"])
        print(f"Signal date {px.index[-1].date()}  predicted risky-sleeve vol {info['pred_vol']:.2%}  k={info['k']:.3f}")
        table = pd.DataFrame({"weight_%": w * 100, "vol_63d_%": info["vol"] * 100,
                              "risk_share_%": rc * 100}).loc[w[w > 0].index]
        print(table.sort_values("weight_%", ascending=False).round(2).to_string())
        return
    r, W, T = backtest(px, p=p)
    rets = px.pct_change(fill_method=None)
    d = pd.DataFrame({"VTRP": r, "SPY": rets[BENCH], "rf": rets[CASH]}).loc["2005-08-01":].fillna(0)
    rows = {}
    for name, (a, b) in {"2005-2015 (in-sample)": ("2005-08-01", "2015-12-31"),
                         "2016-2026 (out-of-sample)": ("2016-01-01", "2026-12-31"),
                         "Full": ("2005-08-01", "2026-12-31")}.items():
        x = d.loc[a:b]
        for k in ["VTRP", "SPY"]:
            rows[(name, k)] = stats(x[k], x["SPY"], x["rf"])
    print(pd.DataFrame(rows).T.round(3).to_string())
    print(f"Turnover per year: {T.loc['2005-08-01':].sum() / (len(d) / 252):.2f}")


if __name__ == "__main__":
    main()
