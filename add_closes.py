#!/usr/bin/env python3
"""Append newly published daily closes to a prices file (used when the archive lags).

  python add_closes.py --prices prices_latest.csv --closes closes.csv

closes.csv (one column per new date):
  ticker,2026-10-30
  SPY,768.10
  XLB,52.31
  ...

All 15 tickers (SPY, 11 sector ETFs, IEF, GLD, SHY) must be present for each date.
Raw closes equal adjusted closes only when no distribution goes ex between the last
date in the file and the new date; check ex-dividend dates (IEF and SHY usually go ex on
the first business day of each month) and rebuild from the archive later to confirm.
"""
import argparse

import pandas as pd

import vtrp

TICKERS = [vtrp.BENCH] + vtrp.RISKY + [vtrp.CASH]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prices", required=True)
    ap.add_argument("--closes", required=True)
    args = ap.parse_args()
    px = pd.read_csv(args.prices, index_col=0)
    new = pd.read_csv(args.closes, index_col=0)
    new.index = new.index.str.strip().str.upper()
    for date in new.columns:
        col = pd.to_numeric(new[date], errors="coerce")
        missing = [t for t in TICKERS if t not in col.index or pd.isna(col.get(t))]
        if missing:
            raise SystemExit(f"{date}: missing closes for {missing}")
        d = pd.Timestamp(date).strftime("%Y-%m-%d")
        if d <= px.index[-1]:
            raise SystemExit(f"{d} is not after the last date in {args.prices} ({px.index[-1]})")
        moves = (col[TICKERS] / px.iloc[-1][TICKERS] - 1).abs()
        big = moves[moves > 0.08]
        if len(big):
            print(f"Check these {d} moves over 8% for typos:", big.round(3).to_dict())
        px.loc[d] = col[px.columns]
    px.to_csv(args.prices)
    print(f"{args.prices} now ends on {px.index[-1]} ({len(px)} rows)")


if __name__ == "__main__":
    main()
