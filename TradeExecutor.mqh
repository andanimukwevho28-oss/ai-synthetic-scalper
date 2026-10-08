//+------------------------------------------------------------------+
//| TradeExecutor.mqh                                                  |
//| Handles order placement/closing with hardening against broker-side |
//| interference: requotes, trade-context-busy, invalid stops from     |
//| freeze/stop level violations, and transient connection errors.     |
//| Uses CTrade for the actual send, wrapped in a retry loop.          |
//+------------------------------------------------------------------+
#property strict
#include <Trade\Trade.mqh>
#include "Inputs.mqh"

class CTradeExecutor
  {
private:
   CTrade m_trade;
   int    m_magic;
   int    m_maxRetries;
   int    m_retryDelayMs;
   int    m_freezeBuffer;

   //--- Retryable error codes: transient / broker-side, worth retrying.
   //--- Non-retryable errors (bad volume, market closed, no money) are NOT retried.
   bool IsRetryable(int err)
     {
      switch(err)
        {
         case TRADE_RETCODE_REQUOTE:
         case TRADE_RETCODE_PRICE_CHANGED:
         case TRADE_RETCODE_PRICE_OFF:
         case TRADE_RETCODE_TIMEOUT:
         case TRADE_RETCODE_CONNECTION:
         case TRADE_RETCODE_TRADE_DISABLED: // sometimes transient during broker maintenance windows
         case TRADE_RETCODE_MARKET_CLOSED:  // synthetic indices trade ~24/7 but guard anyway
            return true;
         default:
            return false;
        }
     }

   //--- Validates SL/TP against the broker's freeze & stop level, and widens
   //--- the safety buffer so orders aren't silently rejected/clipped.
   void ClampStops(const string symbol, bool isBuy, double entry,
                    double &sl, double &tp)
     {
      double point = SymbolInfoDouble(symbol, SYMBOL_POINT);
      int stopLevel = (int)SymbolInfoInteger(symbol, SYMBOL_TRADE_STOPS_LEVEL);
      int freezeLevel = (int)SymbolInfoInteger(symbol, SYMBOL_TRADE_FREEZE_LEVEL);
      int minDistPoints = MathMax(stopLevel, freezeLevel) + InpFreezeCheckBuffer;
      double minDist = minDistPoints * point;

      if(isBuy)
        {
         if(entry - sl < minDist) sl = entry - minDist;
         if(tp - entry < minDist) tp = entry + minDist;
        }
      else
        {
         if(sl - entry < minDist) sl = entry + minDist;
         if(entry - tp < minDist) tp = entry - minDist;
        }
     }

public:
   CTradeExecutor(int magic, int maxRetries, int retryDelayMs, int freezeBuffer)
     {
      m_magic = magic;
      m_maxRetries = maxRetries;
      m_retryDelayMs = retryDelayMs;
      m_freezeBuffer = freezeBuffer;
      m_trade.SetExpertMagicNumber(magic);
      m_trade.SetDeviationInPoints(InpSlippagePoints);
      m_trade.SetTypeFillingBySymbol(_Symbol);
     }

   //--- Opens a position with retry + requote handling. Returns true on success.
   bool OpenPosition(const string symbol, bool isBuy, double lots,
                      int tpPoints, int slPoints)
     {
      if(!SymbolInfoInteger(symbol, SYMBOL_TRADE_MODE) ||
         SymbolInfoInteger(symbol, SYMBOL_TRADE_MODE) == SYMBOL_TRADE_MODE_DISABLED)
        {
         Print("TradeExecutor: symbol trading disabled by broker, aborting entry");
         return false;
        }

      double point = SymbolInfoDouble(symbol, SYMBOL_POINT);
      int digits = (int)SymbolInfoInteger(symbol, SYMBOL_DIGITS);
      int slippage = InpSlippagePoints;

      for(int attempt = 0; attempt < m_maxRetries; attempt++)
        {
         //--- Refresh prices right before send - stale quotes are a common
         //--- source of rejected orders when broker feed is momentarily slow.
         MqlTick tick;
         if(!SymbolInfoTick(symbol, tick))
           {
            Sleep(m_retryDelayMs);
            continue;
           }

         double entry = isBuy ? tick.ask : tick.bid;
         double sl = isBuy ? entry - slPoints * point : entry + slPoints * point;
         double tp = isBuy ? entry + tpPoints * point : entry - tpPoints * point;
         ClampStops(symbol, isBuy, entry, sl, tp);
         sl = NormalizeDouble(sl, digits);
         tp = NormalizeDouble(tp, digits);

         m_trade.SetDeviationInPoints(slippage);

         bool sent = isBuy
            ? m_trade.Buy(lots, symbol, entry, sl, tp, "AI_Scalper")
            : m_trade.Sell(lots, symbol, entry, sl, tp, "AI_Scalper");

         if(sent)
           {
            PrintFormat("TradeExecutor: %s opened lots=%.2f entry=%.5f sl=%.5f tp=%.5f (attempt %d)",
                        isBuy ? "BUY" : "SELL", lots, entry, sl, tp, attempt + 1);
            return true;
           }

         int err = (int)m_trade.ResultRetcode();
         PrintFormat("TradeExecutor: order attempt %d failed, retcode=%d (%s)",
                     attempt + 1, err, m_trade.ResultRetcodeDescription());

         if(err == TRADE_RETCODE_TRADE_CONTEXT_BUSY)
           {
            Sleep(m_retryDelayMs);
            continue;
           }
         if(IsRetryable(err))
           {
            slippage += InpSlippagePoints; // widen tolerance on repeated requotes
            Sleep(m_retryDelayMs);
            continue;
           }

         //--- Non-retryable error (invalid volume, no money, etc.) - stop trying.
         PrintFormat("TradeExecutor: non-retryable error %d, giving up on this signal", err);
         return false;
        }

      Print("TradeExecutor: exhausted retries, entry failed");
      return false;
     }

   //--- Closes a position by ticket with retry handling.
   bool ClosePosition(ulong ticket)
     {
      for(int attempt = 0; attempt < m_maxRetries; attempt++)
        {
         if(!PositionSelectByTicket(ticket))
            return true; // already closed (e.g. hit TP/SL between our checks)

         bool ok = m_trade.PositionClose(ticket, InpSlippagePoints);
         if(ok)
           {
            PrintFormat("TradeExecutor: closed ticket=%I64u (attempt %d)", ticket, attempt + 1);
            return true;
           }

         int err = (int)m_trade.ResultRetcode();
         PrintFormat("TradeExecutor: close attempt %d failed, retcode=%d (%s)",
                     attempt + 1, err, m_trade.ResultRetcodeDescription());

         if(err == TRADE_RETCODE_TRADE_CONTEXT_BUSY || IsRetryable(err))
           {
            Sleep(m_retryDelayMs);
            continue;
           }
         return false;
        }
      Print("TradeExecutor: exhausted retries, close failed");
      return false;
     }

   //--- Count open positions for this symbol+magic (enforces one-trade-at-a-time)
   int CountOpenPositions(const string symbol)
     {
      int count = 0;
      for(int i = 0; i < PositionsTotal(); i++)
        {
         ulong ticket = PositionGetTicket(i);
         if(ticket == 0) continue;
         if(PositionGetString(POSITION_SYMBOL) != symbol) continue;
         if((int)PositionGetInteger(POSITION_MAGIC) != m_magic) continue;
         count++;
        }
      return count;
     }
  };
