"""
config.py — central config for the training pipeline.

Point size matters: Deriv synthetics differ per symbol. Check the symbol's
SYMBOL_POINT in MT5 (right-click chart -> Symbols -> Properties) and set it
here to match, or the label thresholds (in "points") won't line up with the
EA's TP/SL distances, which are also expressed in points.
"""

# Deriv synthetic symbols this pipeline targets (train one model per symbol,
# or pool data across them if you want a general model - start per-symbol).
SYMBOLS = {
    "Volatility 75 Index": {"point": 0.01},
    "Volatility 75 (1s) Index": {"point": 0.01},
    "Jump 100 Index": {"point": 0.01},
    "Jump 75 Index": {"point": 0.01},
}

# Raw tick CSV expected columns: time,bid,ask
# time format: "YYYY-MM-DD HH:MM:SS.fff" (UTC), exported from MT5
# (Tools -> History Center -> export ticks, or via a script using
# CopyTicksRange written to CSV — see build_dataset.py for the MT5 export
# helper script reference).
RAW_TICKS_DIR = "data/raw_ticks"        # one CSV per symbol, named "<symbol>.csv"
PROCESSED_DIR = "data/processed"
MODEL_OUT_DIR = "models"

TRAIN_TEST_SPLIT = 0.8   # chronological split, NOT random (avoid lookahead leakage)
RANDOM_SEED = 42
