//+------------------------------------------------------------------+
//| AI_Synthetic_Scalper.mq5                                           |
//| ONNX-model-driven tick scalper for Deriv synthetic indices          |
//| (Volatility 75, Volatility 75 (1s), Jump 100, Jump 75, etc).       |
//|                                                                      |
//| Architecture:                                                       |
//|  - On every new tick, build a feature vector (FeatureEngine.mqh)   |
//|  - Run it through the ONNX model (OnnxModel.mqh) -> P(target hit    |
//|    before stop, within HORIZON_TICKS)                               |
//|  - If P > InpEntryThreshold and flat -> open BUY                    |
//|  - If (1-P) > InpEntryThreshold and flat -> open SELL                |
//|  - Fixed TP/SL bracket per trade (no martingale, no stacking)       |
//|  - Optional early exit if live probability flips against position  |
//|  - TradeExecutor.mqh hardens every order against broker requotes,   |
//|    trade-context-busy, freeze/stop-level rejections, and dropped    |
//|    connections                                                       |
//|                                                                      |
//| IMPORTANT: Attach this EA to a Deriv MT5 account (Deriv (SVG) LLC   |
//| MT5 or similar server exposing synthetic indices). This is NOT the  |
//| Deriv WebSocket/DTrader API - if you're routing through that        |
//| instead, this EA doesn't apply; use a Python client against the WS  |
//| API and run ONNX inference there instead (onnxruntime, same .onnx   |
//| file works).                                                         |
//+------------------------------------------------------------------+
#property copyright "Take"
#property version   "1.00"
#property strict

#include "Inputs.mqh"
#include "FeatureEngine.mqh"
#include "OnnxModel.mqh"
#include "TradeExecutor.mqh"

CFeatureEngine *g_features = NULL;
COnnxModel      g_model;
CTradeExecutor *g_executor = NULL;

double g_dayStartBalance = 0.0;
datetime g_currentDay = 0;
bool g_tradingPaused = false;

//+------------------------------------------------------------------+
int OnInit()
  {
   g_features = new CFeatureEngine(InpFastEMA, InpSlowEMA, InpVolWindow, InpRunWindow);
   g_executor = new CTradeExecutor(InpMagicNumber, InpMaxRetries, InpRetryDelayMs, InpFreezeCheckBuffer);

   if(!g_model.Load(InpModelFileName))
     {
      Print("AI_Synthetic_Scalper: ONNX model failed to load. EA will not trade until fixed.");
      // Do not return INIT_FAILED - allow chart attachment so the user can see
      // the log/error, but block trading via g_tradingPaused.
      g_tradingPaused = true;
     }

   ResetDailyTracking();

   PrintFormat("AI_Synthetic_Scalper initialized on %s. Model loaded=%s",
               _Symbol, g_model.IsLoaded() ? "true" : "false");
   return(INIT_SUCCEEDED);
  }

//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   if(g_features != NULL) { delete g_features; g_features = NULL; }
   if(g_executor != NULL) { delete g_executor; g_executor = NULL; }
  }

//+------------------------------------------------------------------+
void ResetDailyTracking()
  {
   MqlDateTime dt;
   TimeToStruct(TimeCurrent(), dt);
   dt.hour = 0; dt.min = 0; dt.sec = 0;
   g_currentDay = StructToTime(dt);
   g_dayStartBalance = AccountInfoDouble(ACCOUNT_BALANCE);
   g_tradingPaused = !g_model.IsLoaded(); // stay paused if model never loaded
  }

//+------------------------------------------------------------------+
bool WithinTradingHours()
  {
   if(!InpUseTradingHoursFilter) return true;
   MqlDateTime dt;
   TimeToStruct(TimeCurrent(), dt);
   if(InpStartHour <= InpEndHour)
      return (dt.hour >= InpStartHour && dt.hour <= InpEndHour);
   return (dt.hour >= InpStartHour || dt.hour <= InpEndHour); // wraps midnight
  }

//+------------------------------------------------------------------+
void CheckDailyLossGuard()
  {
   MqlDateTime dt;
   TimeToStruct(TimeCurrent(), dt);
   dt.hour = 0; dt.min = 0; dt.sec = 0;
   datetime today = StructToTime(dt);
   if(today != g_currentDay)
      ResetDailyTracking();

   double balance = AccountInfoDouble(ACCOUNT_BALANCE);
   double lossPct = (g_dayStartBalance > 0)
      ? (g_dayStartBalance - balance) / g_dayStartBalance * 100.0
      : 0.0;

   if(lossPct >= InpMaxDailyLossPct)
     {
      if(!g_tradingPaused)
         PrintFormat("AI_Synthetic_Scalper: daily loss guard triggered (%.2f%% >= %.2f%%), pausing new entries.",
                     lossPct, InpMaxDailyLossPct);
      g_tradingPaused = true;
     }
  }

//+------------------------------------------------------------------+
//| Manage an existing open position: early-exit on probability flip |
//+------------------------------------------------------------------+
void ManageOpenPosition(double liveProbUp)
  {
   for(int i = 0; i < PositionsTotal(); i++)
     {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0) continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol) continue;
      if((int)PositionGetInteger(POSITION_MAGIC) != InpMagicNumber) continue;

      long type = PositionGetInteger(POSITION_TYPE);
      bool isBuy = (type == POSITION_TYPE_BUY);

      // For a BUY, we entered because P(up) was high. If it has flipped low,
      // the model no longer supports the position -> close early rather than
      // waiting for the fixed SL, since fixed SL by definition means "wrong".
      bool shouldFlipClose = isBuy
         ? (liveProbUp < InpExitFlipThreshold)
         : (liveProbUp > (1.0 - InpExitFlipThreshold));

      if(shouldFlipClose)
        {
         PrintFormat("AI_Synthetic_Scalper: model flipped (p=%.3f), closing ticket=%I64u early",
                     liveProbUp, ticket);
         g_executor.ClosePosition(ticket);
        }
     }
  }

//+------------------------------------------------------------------+
void OnTick()
  {
   MqlTick tick;
   if(!SymbolInfoTick(_Symbol, tick)) return;

   if(g_features == NULL || g_executor == NULL) return;

   g_features.OnTick(tick.bid, tick.ask, tick.time);

   CheckDailyLossGuard();

   if(g_features.Count() < InpMinBarsHistory) return; // not enough warm-up yet

   double featureVec[];
   if(!g_features.ExtractFeatures(tick.bid, tick.ask, featureVec))
      return;

   if(!g_model.IsLoaded())
      return; // nothing we can do without the model

   double probUp = g_model.PredictUpProbability(featureVec);
   if(probUp < 0.0) return; // inference failed this tick, skip safely

   //--- Manage any open position first (early exit on flip)
   int openCount = g_executor.CountOpenPositions(_Symbol);
   if(openCount > 0)
     {
      ManageOpenPosition(probUp);
      return; // don't also try to open a new trade this tick
     }

   if(g_tradingPaused) return;
   if(!WithinTradingHours()) return;
   if(InpOneTradeAtATime && openCount > 0) return;

   //--- Broker interference guard: refuse to trade on abnormal spread
   double point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
   double spreadPoints = (tick.ask - tick.bid) / point;
   if(spreadPoints > InpMaxSpreadPoints)
     {
      return; // spread spike - likely broker-side widening, sit out
     }

   //--- Broker interference guard: confirm symbol is actually tradeable now
   long tradeMode = SymbolInfoInteger(_Symbol, SYMBOL_TRADE_MODE);
   if(tradeMode == SYMBOL_TRADE_MODE_DISABLED || tradeMode == SYMBOL_TRADE_MODE_CLOSEONLY)
      return;

   bool wantBuy  = (probUp >= InpEntryThreshold);
   bool wantSell = ((1.0 - probUp) >= InpEntryThreshold);

   if(!wantBuy && !wantSell) return; // model isn't confident either way

   if(wantBuy)
      g_executor.OpenPosition(_Symbol, true, InpLots, InpTargetPoints, InpStopPoints);
   else if(wantSell)
      g_executor.OpenPosition(_Symbol, false, InpLots, InpTargetPoints, InpStopPoints);
  }
//+------------------------------------------------------------------+
