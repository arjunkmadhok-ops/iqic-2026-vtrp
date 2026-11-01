#!/usr/bin/env python3
"""Build prices.csv (adjusted closes, one column per ETF) from the open sp500-data archive.

  pip install pandas pyarrow
  python build_prices.py                       # downloads prices.parquet (~235 MB)
  python build_prices.py --parquet prices.parquet   # use a file you already have

Source: Yahoo Finance prices published by the sp500-data project as a GitHub release
asset. The declaration's results used the file with SHA-256
974f5745443a01f8bc965cc6213edf3b1f5c41fd08e02c77b6e2339ecae3531a (prices through 2026-10-28).
The release is refreshed nightly, so a newer download adds later days; earlier
adjusted prices can shift slightly when new dividends are paid.
"""
import argparse
import hashlib
import urllib.request
from pathlib import Path

import pandas as pd

URL = "https://github.com/Johnbrick123/sp500-data/releases/download/data/prices.parquet"
TICKERS = ["SPY", "XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY",
           "IEF", "GLD", "SHY"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", help="existing prices.parquet; omit to download")
    ap.add_argument("--out", default="prices.csv")
    args = ap.parse_args()
    path = Path(args.parquet or "prices.parquet")
    if not args.parquet:
        print(f"Downloading {URL} ...")
        urllib.request.urlretrieve(URL, path)
    print(f"{path}: SHA-256 {sha256(path)}")
    df = pd.read_parquet(path, columns=["date", "ticker", "adj_close"],
                         filters=[("ticker", "in", TICKERS)])
    px = df.pivot(index="date", columns="ticker", values="adj_close")[TICKERS]
    px.index = pd.to_datetime(px.index).strftime("%Y-%m-%d")
    px.index.name = "Date"
    px.to_csv(args.out)
    print(f"Wrote {args.out}: {len(px)} days, {px.index[0]} to {px.index[-1]}")


if __name__ == "__main__":
    main()
