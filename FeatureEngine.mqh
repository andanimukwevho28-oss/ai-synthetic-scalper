//+------------------------------------------------------------------+
//| FeatureEngine.mqh                                                  |
//| Reproduces python_train/features.py build_features() tick-by-tick. |
//| Feature order MUST match FEATURE_COLUMNS in features.py exactly:   |
//|  [spread, ema_diff, ret_1, ret_5, ret_15, run_len, tick_speed,     |
//|   volatility, spread_ratio]                                        |
//+------------------------------------------------------------------+
#property strict

#define FEATURE_COUNT 9

class CFeatureEngine
  {
private:
   double m_mid[];       // ring buffer of recent mid prices
   datetime m_time[];    // matching timestamps
   int    m_capacity;
   int    m_count;       // number of valid entries currently stored
   int    m_head;        // index of most recent entry

   double m_ema_fast;
   double m_ema_slow;
   bool   m_ema_init;

   int    m_run_len;
   double m_last_dir;    // -1, 0, +1

   int    m_fast_period, m_slow_period, m_vol_window, m_run_window;

   double At(int back) // back=0 -> most recent
     {
      int idx = m_head - back;
      if(idx < 0) idx += m_capacity;
      return m_mid[idx];
     }

   datetime TimeAt(int back)
     {
      int idx = m_head - back;
      if(idx < 0) idx += m_capacity;
      return m_time[idx];
     }

public:
   CFeatureEngine(int fast_p, int slow_p, int vol_w, int run_w)
     {
      m_fast_period = fast_p;
      m_slow_period = slow_p;
      m_vol_window  = vol_w;
      m_run_window  = run_w;
      m_capacity    = MathMax(vol_w, run_w) + slow_p + 20;
      ArrayResize(m_mid, m_capacity);
      ArrayResize(m_time, m_capacity);
      m_count = 0;
      m_head  = -1;
      m_ema_init = false;
      m_ema_fast = 0.0;
      m_ema_slow = 0.0;
      m_run_len  = 0;
      m_last_dir = 0.0;
     }

   int Count() const { return m_count; }

   //--- feed a new tick (mid price). Call once per new tick BEFORE ExtractFeatures.
   void OnTick(double bid, double ask, datetime t)
     {
      double mid = (bid + ask) / 2.0;

      m_head = (m_head + 1) % m_capacity;
      m_mid[m_head] = mid;
      m_time[m_head] = t;
      if(m_count < m_capacity) m_count++;

      //--- EMA update (matches pandas ewm(span=N, adjust=False))
      double alpha_fast = 2.0 / (m_fast_period + 1.0);
      double alpha_slow = 2.0 / (m_slow_period + 1.0);
      if(!m_ema_init)
        {
         m_ema_fast = mid;
         m_ema_slow = mid;
         m_ema_init = true;
        }
      else
        {
         m_ema_fast = alpha_fast * mid + (1.0 - alpha_fast) * m_ema_fast;
         m_ema_slow = alpha_slow * mid + (1.0 - alpha_slow) * m_ema_slow;
        }

      //--- run-length update (matches features.py run_length())
      if(m_count >= 2)
        {
         double prev = At(1);
         double diff = mid - prev;
         double dir = (diff > 0) ? 1.0 : (diff < 0 ? -1.0 : 0.0);
         if(dir == 0.0)
            m_run_len = 0;
         else if(dir == m_last_dir)
            m_run_len++;
         else
            m_run_len = 1;
         if(dir != 0.0) m_last_dir = dir;
        }
     }

   //--- Returns true and fills out[FEATURE_COUNT] if enough history exists.
   bool ExtractFeatures(double &bid, double &ask, double &out[])
     {
      int warmup = MathMax(m_slow_period, MathMax(m_vol_window, m_run_window)) + 1;
      if(m_count < warmup) return false;
      if(ArraySize(out) < FEATURE_COUNT) ArrayResize(out, FEATURE_COUNT);

      double mid = At(0);
      double spread = ask - bid;
      double ema_diff = m_ema_fast - m_ema_slow;

      double ret_1  = (m_count >= 2)  ? (mid - At(1))  : 0.0;
      double ret_5  = (m_count >= 6)  ? (mid - At(5))  : 0.0;
      double ret_15 = (m_count >= 16) ? (mid - At(15)) : 0.0;

      //--- volatility: rolling stdev of first differences over m_vol_window
      double vol = 0.0;
      {
         int n = MathMin(m_vol_window, m_count - 1);
         if(n > 1)
           {
            double diffs[];
            ArrayResize(diffs, n);
            double sum = 0.0;
            for(int i = 0; i < n; i++)
              {
               diffs[i] = At(i) - At(i + 1);
               sum += diffs[i];
              }
            double mean = sum / n;
            double sq = 0.0;
            for(int i = 0; i < n; i++)
               sq += (diffs[i] - mean) * (diffs[i] - mean);
            vol = MathSqrt(sq / (n - 1)); // sample stdev, matches pandas default ddof=1
           }
      }

      //--- tick speed: 1 / mean(dt) over run_window, using stored timestamps
      double tick_speed = 0.0;
      {
         int n = MathMin(m_run_window, m_count - 1);
         if(n >= 1)
           {
            long total_sec = 0;
            for(int i = 0; i < n; i++)
               total_sec += (long)(TimeAt(i) - TimeAt(i + 1));
            double mean_dt = (n > 0) ? (double)total_sec / n : 0.0;
            if(mean_dt > 0.0) tick_speed = 1.0 / mean_dt;
           }
      }

      //--- spread_ratio: current spread / rolling mean spread over vol_window
      //--- NOTE: we don't store historical spreads separately; approximate
      //--- using current spread vs itself is insufficient, so we track a
      //--- simple running mean of spread as a lightweight proxy.
      double spread_ratio = 1.0; // safe neutral default; refined below
      static double s_spread_sum = 0.0;
      static int    s_spread_n = 0;
      static double s_spread_buf[]; // simple ring for mean spread
      if(ArraySize(s_spread_buf) == 0) ArrayResize(s_spread_buf, m_vol_window);
      // shift buffer
      for(int i = m_vol_window - 1; i > 0; i--) s_spread_buf[i] = s_spread_buf[i-1];
      s_spread_buf[0] = spread;
      if(s_spread_n < m_vol_window) s_spread_n++;
      double mean_spread = 0.0;
      for(int i = 0; i < s_spread_n; i++) mean_spread += s_spread_buf[i];
      mean_spread /= s_spread_n;
      if(mean_spread > 0.0) spread_ratio = spread / mean_spread;

      out[0] = spread;
      out[1] = ema_diff;
      out[2] = ret_1;
      out[3] = ret_5;
      out[4] = ret_15;
      out[5] = (double)m_run_len;
      out[6] = tick_speed;
      out[7] = vol;
      out[8] = spread_ratio;

      return true;
     }
  };
