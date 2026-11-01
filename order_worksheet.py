#!/usr/bin/env python3
"""IQIC 2026 - StockTrak order worksheet for the VTRP strategy.

Turns the declared rules into whole-share limit orders for the next Monday.
It does NOT connect to StockTrak or place orders: you enter them by hand,
then update account.json from StockTrak's actual fills.

Usage (after the Friday close):
  python order_worksheet.py --prices prices.csv --account account.json \
      --quotes quotes.csv --signal-date 2026-10-30

Inputs
  prices.csv   adjusted closes through the signal date (build_prices.py)
  account.json read from StockTrak (My Portfolio -> Portfolio Summary / Open Positions):
                 equity               current account value
                 cash                 current cash
                 holdings             {"IEF": 120, ...} shares held now
                 max_position_weight  class position limit (our cap is min(25%, this))
                 commission_per_trade fixed fee per order, if any
                 nav_history          account closing values since Nov 2 (for the breaker)
                 first_build          true for the Nov 2 build (ignores the 2-point band)
  quotes.csv   ticker,bid,ask  (Monday quotes, after 10:00 ET)

Outputs
  orders_<date>.csv  proposed orders: sells first, then buys, SHY last
  signal_<date>.json audit record: weights, volatilities, k, drawdown, breaker state
"""
import argparse
import csv
import json
import math
from pathlib import Path

import pandas as pd

import vtrp


def mid(q):
    return (q[0] + q[1]) / 2


def plan_orders(w, holdings, cash, equity, quotes, fee=0.0, first=False, band=vtrp.PARAMS["band"]):
    """Whole-share orders that move the account to target weights w.

    Rule 5: trade only on the first build or if some weight is >= band from target.
    Sells first (valued at the bid), then risky buys largest first, then SHY with the cash
    left (buys valued at the ask). Cash never goes below zero, so the account never borrows.
    Returns (orders, estimated cash after fills, largest weight gap, whether to trade).
    """
    cur_w = {a: holdings.get(a, 0) * mid(quotes[a]) / equity for a in w.index}
    max_dev = max(abs(w[a] - cur_w[a]) for a in w.index)
    trade = first or max_dev >= band
    orders = []
    if not trade:
        return orders, cash, max_dev, trade
    target_sh = {a: math.floor(equity * w[a] / mid(quotes[a])) for a in w.index}
    for a, held in sorted(holdings.items()):              # sells, incl. tickers no longer targeted
        diff = held - target_sh.get(a, 0)
        if diff > 0:
            bid, _ = quotes[a]
            proceeds = diff * bid - fee
            cash += proceeds
            orders.append(dict(ticker=a, action="SELL", shares=diff, limit=round(mid(quotes[a]), 2),
                               reprice_1530=round(bid, 2), est_cash=round(proceeds, 2)))
    buys = sorted((a for a in w.index if a != vtrp.CASH), key=lambda a: -w[a]) + [vtrp.CASH]
    for a in buys:
        diff = target_sh[a] - holdings.get(a, 0)
        if diff <= 0:
            continue
        _, ask = quotes[a]
        diff = min(diff, math.floor(max(0.0, cash - fee) / ask))    # never borrow
        if diff > 0:
            spend = diff * ask + fee
            cash -= spend
            orders.append(dict(ticker=a, action="BUY", shares=diff, limit=round(mid(quotes[a]), 2),
                               reprice_1530=round(ask, 2), est_cash=round(-spend, 2)))
    return orders, cash, max_dev, trade


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prices", required=True)
    ap.add_argument("--account", required=True)
    ap.add_argument("--quotes", required=True)
    ap.add_argument("--signal-date", required=True, help="the Friday close the signal uses, YYYY-MM-DD")
    ap.add_argument("--outdir", default=".")
    args = ap.parse_args()

    acct = json.loads(Path(args.account).read_text())
    equity, cash = float(acct["equity"]), float(acct["cash"])
    holdings = {k: int(v) for k, v in acct.get("holdings", {}).items() if int(v) != 0}
    fee = float(acct.get("commission_per_trade", 0.0))
    cap = min(vtrp.PARAMS["cap"], float(acct.get("max_position_weight", 1.0)))
    first = bool(acct.get("first_build", False))
    if any(v < 0 for v in holdings.values()):
        raise SystemExit("Short positions found: the strategy is long-only. Reconcile with StockTrak first.")

    quotes = {}
    with open(args.quotes, newline="") as f:
        for row in csv.DictReader(f):
            bid, ask = float(row["bid"]), float(row["ask"])
            if not 0 < bid <= ask:
                raise SystemExit(f"Bad quote for {row['ticker']}: bid {bid}, ask {ask}")
            quotes[row["ticker"].strip().upper()] = (bid, ask)

    # --- Rules 1-3 at the signal close -------------------------------------------------
    px = vtrp.load_prices(args.prices)
    sig = pd.Timestamp(args.signal_date)
    px = px.loc[:sig]
    if px.index[-1] != sig:
        raise SystemExit(f"prices.csv does not contain {sig.date()}; last row is {px.index[-1].date()}")
    w, info = vtrp.risk_parity_targets(px, len(px) - 1, cap=cap)

    # --- Rule 4: breaker on the account's peak since Nov 2 -------------------------------
    navs = [float(x) for x in acct.get("nav_history", [])] + [equity]
    dd = equity / max(navs) - 1
    w = vtrp.apply_breaker(w, dd)
    breaker_on = dd < -vtrp.PARAMS["breaker_dd"]
    # The class position limit applies to SHY as well; anything above it stays as cash.
    w, cash_target = vtrp.cap_cash_sleeve(w, float(acct.get("max_position_weight", 1.0)))

    missing = [a for a in w.index if (w[a] > 0 or a in holdings) and a not in quotes]
    if missing:
        raise SystemExit(f"Missing quotes for {missing}")

    # --- Rule 5 and order arithmetic ---------------------------------------------------------
    orders, cash, max_dev, trade = plan_orders(w, holdings, cash, equity, quotes, fee, first)
    rc = vtrp.risk_contributions(w, info["cov"])

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    tag = sig.strftime("%Y-%m-%d")
    with open(out / f"orders_{tag}.csv", "w", newline="") as f:
        cols = ["ticker", "action", "shares", "limit", "reprice_1530", "est_cash"]
        wr = csv.DictWriter(f, fieldnames=cols)
        wr.writeheader()
        wr.writerows(orders)
    record = dict(signal_date=tag, cap=cap, predicted_sleeve_vol=round(info["pred_vol"], 4),
                  k=round(info["k"], 4), drawdown=round(dd, 4), breaker_on=bool(breaker_on),
                  max_weight_gap=round(max_dev, 4), trade=bool(trade),
                  target_weights={a: round(float(w[a]), 4) for a in w.index if w[a] > 0},
                  volatility_63d={a: round(float(v), 4) for a, v in info.get("vol", pd.Series()).items()},
                  risk_share={a: round(float(v), 4) for a, v in rc.items()},
                  cash_target_weight=round(cash_target, 4),
                  est_cash_after_fills=round(cash, 2))
    (out / f"signal_{tag}.json").write_text(json.dumps(record, indent=2))

    print(f"Signal {tag}: predicted sleeve vol {info['pred_vol']:.2%}, k={info['k']:.3f}, "
          f"drawdown {dd:.2%}, breaker {'ON' if breaker_on else 'off'}, cap {cap:.0%}")
    if not trade:
        print(f"No trade: largest gap {max_dev:.2%} is inside the 2-point band.")
    else:
        print(pd.DataFrame(orders).to_string(index=False))
        print(f"Estimated cash after all fills: ${cash:,.2f}")
        print("Enter orders after 10:00 ET at the limit shown; re-price unfilled orders at 15:30 ET "
              "to the reprice_1530 price. Log every fill.")


if __name__ == "__main__":
    main()
