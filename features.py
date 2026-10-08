"""
features.py

Feature engineering for the tick-scalping model.

CRITICAL: Every feature computed here MUST be reproduced EXACTLY (same
formula, same lookback, same rounding behaviour) in the MQL5 EA's
BuildFeatureVector() function (see mt5_ea/FeatureEngine.mqh). If the two
diverge, the model scores live data it was never trained on.

Label target: "will price move >= TARGET_POINTS before it moves
<= -STOP_POINTS, within HORIZON_TICKS ticks" (a triple-barrier label).
This matches the EA's actual TP/SL rule instead of raw next-tick direction.
"""

import numpy as np
import pandas as pd

# ---- Config: MUST match Inputs in the MT5 EA (Inputs.mqh) ------------------
HORIZON_TICKS = 15
TARGET_POINTS = 25
STOP_POINTS = 25
FAST_EMA = 5
SLOW_EMA = 20
VOL_WINDOW = 30
RUN_WINDOW = 10

FEATURE_COLUMNS = [
    "spread", "ema_diff", "ret_1", "ret_5", "ret_15",
    "run_len", "tick_speed", "volatility", "spread_ratio",
]


def build_features(ticks: pd.DataFrame) -> pd.DataFrame:
    """
    ticks: DataFrame with columns ['time', 'bid', 'ask'], one row per tick,
    sorted ascending by time, single symbol only.
    """
    df = ticks.copy()
    df["mid"] = (df["bid"] + df["ask"]) / 2.0
    df["spread"] = df["ask"] - df["bid"]

    df["ema_fast"] = df["mid"].ewm(span=FAST_EMA, adjust=False).mean()
    df["ema_slow"] = df["mid"].ewm(span=SLOW_EMA, adjust=False).mean()
    df["ema_diff"] = df["ema_fast"] - df["ema_slow"]

    df["ret_1"] = df["mid"].diff(1)
    df["ret_5"] = df["mid"].diff(5)
    df["ret_15"] = df["mid"].diff(15)

    tick_dir = np.sign(df["mid"].diff()).fillna(0).values

    def run_length(arr):
        out = np.zeros(len(arr))
        streak = 0
        last = 0
        for i, v in enumerate(arr):
            if v == 0:
                streak = 0
            elif v == last:
                streak += 1
            else:
                streak = 1
            out[i] = streak
            last = v if v != 0 else last
        return out

    df["run_len"] = run_length(tick_dir)

    df["t_epoch"] = pd.to_datetime(df["time"]).astype(np.int64) / 1e9
    dt = df["t_epoch"].diff().rolling(RUN_WINDOW).mean()
    df["tick_speed"] = 1.0 / dt.replace(0, np.nan)

    df["volatility"] = df["mid"].diff().rolling(VOL_WINDOW).std()
    df["spread_ratio"] = df["spread"] / df["spread"].rolling(VOL_WINDOW).mean()

    return df[FEATURE_COLUMNS]


def build_labels(ticks: pd.DataFrame) -> pd.Series:
    """
    Triple-barrier label per tick i:
        1  -> mid hits +TARGET_POINTS before -STOP_POINTS within HORIZON_TICKS
        0  -> hits -STOP_POINTS first, or neither barrier hit (timeout)

    TARGET_POINTS / STOP_POINTS must already be in the SAME price units as
    `mid` (i.e. pre-multiplied by the symbol's point size). See train.py.
    """
    mid = ((ticks["bid"] + ticks["ask"]) / 2.0).values
    n = len(mid)
    labels = np.full(n, np.nan)

    for i in range(n - HORIZON_TICKS):
        entry = mid[i]
        window = mid[i + 1: i + 1 + HORIZON_TICKS]
        up_hit = np.where(window - entry >= TARGET_POINTS)[0]
        down_hit = np.where(entry - window >= STOP_POINTS)[0]

        up_first = up_hit[0] if len(up_hit) else np.inf
        down_first = down_hit[0] if len(down_hit) else np.inf

        if up_first == np.inf and down_first == np.inf:
            continue  # timeout, leave NaN -> dropped from training set
        labels[i] = 1.0 if up_first < down_first else 0.0

    return pd.Series(labels, index=ticks.index, name="label")
