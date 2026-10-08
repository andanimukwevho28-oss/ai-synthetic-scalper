# AI Synthetic Scalper — Deriv Volatility/Jump Indices

Two linked pieces:

1. **`python_train/`** — Python (Visual Studio / VS Code) project that pulls
   tick history, engineers features, labels each tick with a triple-barrier
   outcome, trains a LightGBM classifier, and exports it to **ONNX**.
2. **`mt5_ea/`** — MT5 Expert Advisor (`AI_Synthetic_Scalper.mq5`) that loads
   that `.onnx` file with `OnnxCreate()`/`OnnxRun()` and trades live off it.

## How they link

```
python_train/train.py  --symbol "Volatility 75 Index"
        │
        ▼
models/Volatility 75 Index.onnx
        │
   (copy this file into your MT5 terminal's  MQL5/Files/  folder)
        │
        ▼
mt5_ea/AI_Synthetic_Scalper.mq5   (input: InpModelFileName = "Volatility 75 Index.onnx")
```

Train **one model per symbol** (Volatility 75, Volatility 75 (1s), Jump 100,
Jump 75 each get their own `.onnx` — their tick behaviour differs enough that
a single pooled model will underperform). Attach one EA instance per chart/symbol,
each pointed at its own `.onnx` file.

## What this actually does (read this before trading real money)

- The model predicts: *"will price move +25 points before -25 points, within
  the next 15 ticks?"* — not "the next tick's direction." This is deliberately
  matched to the EA's actual TP/SL so the label and the trading rule agree.
- **No martingale, no stacking.** One position at a time. This is the opposite
  of the "many trades at the same price" pattern in grid/martingale EAs —
  those fake a high win rate by volume, not accuracy. This EA lives or dies on
  the model's real edge.
- The features in `features.py` and `FeatureEngine.mqh` **must stay identical**
  — same lookback windows, same formulas. If you change one, change the other,
  retrain, and redeploy the new `.onnx`. A mismatch here is the single most
  common way people silently break these systems.
- **Realistic expectations**: a well-built model on retail tick data typically
  lands somewhere around 53–58% directional accuracy on this kind of task —
  genuinely useful with correct position sizing and risk control, nowhere
  near the "always right, instant profit" impression a marketing video gives.
  Test on out-of-sample data and on a demo account before going live. Nothing
  here guarantees profitability; markets can and do invalidate any trained
  model when regime shifts.

## Step-by-step

1. Export tick history for each symbol from MT5 (see the MQL5 export script
   documented in `python_train/build_dataset.py`'s docstring) into
   `python_train/data/raw_ticks/<Symbol Name>.csv`.
2. `pip install -r python_train/requirements.txt`
3. `python build_dataset.py --symbol "Volatility 75 Index"`
4. `python train.py --symbol "Volatility 75 Index"` → produces
   `models/Volatility 75 Index.onnx` + a report of test accuracy/AUC/feature
   importance. **Check the AUC before deploying** — anything near 0.50 means
   no real edge on that symbol/timeframe; don't trade that model live.
5. Copy the `.onnx` file into your MT5 terminal's `MQL5/Files/` folder.
6. Copy all files from `mt5_ea/` into your MT5 terminal's `MQL5/Experts/`
   folder (keep them together — the `.mq5` `#include`s the `.mqh` files).
7. Open `AI_Synthetic_Scalper.mq5` in MetaEditor, compile (F7).
8. Attach to the matching chart (e.g. Volatility 75 Index chart), set
   `InpModelFileName` to match, enable Algo Trading, and enable
   **"Allow DLL imports" is NOT needed** but **"Allow WebRequest"/ONNX
   support requires MT5 build 3980+** — update your terminal if OnnxCreate
   fails on load.
9. **Test on a demo account first.** Watch the Experts log — every open,
   close, retry, and rejection is logged with the reason.

## Broker-interference hardening included in the EA

- Retries on requotes, price-changed, timeout, connection errors, and
  trade-context-busy, with widening slippage tolerance on repeated requotes.
- Clamps SL/TP to the broker's actual freeze level + stop level (plus a
  configurable safety buffer) so orders aren't silently rejected or clipped.
- Refuses to open when spread spikes beyond `InpMaxSpreadPoints` (protects
  against broker-side spread widening around news/rollover).
- Refuses to trade if the symbol is reported disabled/close-only.
- Refreshes the live tick immediately before sending each order (avoids
  acting on a stale quote).
- Daily loss guard pauses new entries if the account is down more than
  `InpMaxDailyLossPct` for the day.

None of this can force a broker to fill you — no EA can — but it stops the
common failure modes (silent stop-level rejection, stale-quote requotes,
context-busy errors) from either crashing the EA or leaving positions
unmanaged.
