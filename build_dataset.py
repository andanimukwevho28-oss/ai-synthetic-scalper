"""
build_dataset.py

Reads raw tick CSVs (time,bid,ask) exported from MT5 per symbol, computes
features + triple-barrier labels, and writes a clean parquet dataset ready
for train.py.

Usage:
    python build_dataset.py --symbol "Volatility 75 Index"
    python build_dataset.py --all

--- How to export tick data from MT5 for a Deriv synthetic symbol ---
MT5's Strategy Tester / History Center doesn't cleanly bulk-export raw ticks
via the GUI for long ranges. The reliable way is a small MQL5 script run
once in the terminal:

    // ExportTicks.mq5 (script, not EA) - copy into MQL5/Scripts, compile, run
    void OnStart() {
        MqlTick ticks[];
        datetime from = D'2025.01.01 00:00';
        datetime to   = D'2025.06.01 00:00';
        int n = CopyTicksRange(_Symbol, ticks, COPY_TICKS_ALL,
                                (long)from * 1000, (long)to * 1000);
        int h = FileOpen(_Symbol + "_ticks.csv", FILE_WRITE|FILE_CSV|FILE_ANSI);
        FileWrite(h, "time", "bid", "ask");
        for (int i = 0; i < n; i++)
            FileWrite(h, TimeToString(ticks[i].time, TIME_DATE|TIME_SECONDS),
                       ticks[i].bid, ticks[i].ask);
        FileClose(h);
    }

Run it with the chart symbol set to the Deriv synthetic you want, find the
output under MQL5/Files/, then move it into data/raw_ticks/<symbol>.csv.
"""

import argparse
import os

import pandas as pd

from config import SYMBOLS, RAW_TICKS_DIR, PROCESSED_DIR
from features import build_features, build_labels, TARGET_POINTS, STOP_POINTS
import features as feat_module


def process_symbol(symbol: str) -> None:
    point = SYMBOLS[symbol]["point"]
    raw_path = os.path.join(RAW_TICKS_DIR, f"{symbol}.csv")
    if not os.path.exists(raw_path):
        print(f"[skip] no raw tick file for {symbol} at {raw_path}")
        return

    print(f"[{symbol}] loading raw ticks...")
    ticks = pd.read_csv(raw_path)
    ticks["time"] = pd.to_datetime(ticks["time"])
    ticks = ticks.sort_values("time").reset_index(drop=True)

    print(f"[{symbol}] {len(ticks)} ticks loaded, building features...")
    X = build_features(ticks)

    # Convert point-based thresholds to this symbol's price units for labeling
    feat_module.TARGET_POINTS = TARGET_POINTS * point
    feat_module.STOP_POINTS = STOP_POINTS * point
    y = build_labels(ticks)

    data = X.copy()
    data["label"] = y
    before = len(data)
    data = data.dropna()
    print(f"[{symbol}] dropped {before - len(data)} warm-up/timeout rows, "
          f"{len(data)} usable rows remain")

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    out_path = os.path.join(PROCESSED_DIR, f"{symbol}.parquet")
    data.to_parquet(out_path)
    print(f"[{symbol}] wrote {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, default=None)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    if args.all:
        for s in SYMBOLS:
            process_symbol(s)
    elif args.symbol:
        if args.symbol not in SYMBOLS:
            raise SystemExit(f"Unknown symbol '{args.symbol}'. "
                              f"Add it to config.py SYMBOLS first.")
        process_symbol(args.symbol)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
