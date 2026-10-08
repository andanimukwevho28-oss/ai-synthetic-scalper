//+------------------------------------------------------------------+
//| Inputs.mqh - all tunable EA parameters                            |
//| Keep HORIZON/TARGET/STOP/EMA/window values identical to           |
//| python_train/features.py and config.py, or the model sees         |
//| features it was never trained on.                                 |
//+------------------------------------------------------------------+
#property strict

//--- Model
input string  InpModelFileName   = "Volatility 75 Index.onnx"; // Must be in MQL5/Files/
input double  InpEntryThreshold  = 0.62;   // min model probability to open a trade
input double  InpExitFlipThreshold = 0.45; // if live prob drops below this while in a BUY, close early (mirror for SELL)

//--- Feature engineering (MUST match python_train/features.py)
input int     InpFastEMA         = 5;
input int     InpSlowEMA         = 20;
input int     InpVolWindow       = 30;
input int     InpRunWindow       = 10;

//--- Trade management
input double  InpLots            = 0.01;
input int     InpTargetPoints    = 25;     // TP distance, in points
input int     InpStopPoints      = 25;     // SL distance, in points
input int     InpMaxSpreadPoints = 40;     // refuse to enter if spread exceeds this
input int     InpMagicNumber     = 990177;
input int     InpSlippagePoints  = 10;
input bool    InpOneTradeAtATime = true;   // no stacking/martingale - one position max

//--- Broker-interference hardening
input int     InpMaxRetries        = 5;    // retry OrderSend on transient errors
input int     InpRetryDelayMs      = 250;  // wait between retries
input int     InpMaxSlippageRetries= 3;    // widen slippage tolerance after N requote failures
input int     InpFreezeCheckBuffer = 5;    // extra points beyond broker's freeze/stop level, safety margin
input int     InpMinBarsHistory    = 100;  // don't trade until this many bars of context exist

//--- Session / risk guards
input bool    InpUseTradingHoursFilter = false;
input int     InpStartHour       = 0;
input int     InpEndHour         = 23;
input double  InpMaxDailyLossPct = 5.0;    // pause trading if daily loss exceeds this % of starting balance
