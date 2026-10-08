//+------------------------------------------------------------------+
//| OnnxModel.mqh                                                      |
//| Wraps OnnxCreate/OnnxRun for the LightGBM->ONNX classifier.        |
//|                                                                      |
//| Model I/O contract (verified against onnxmltools convert_lightgbm  |
//| output in python_train/train.py):                                  |
//|   input:  "input"          float32[N, FEATURE_COUNT]               |
//|   output: "label"          int64[N]                                 |
//|   output: "probabilities"  float32[N, 2]   (prob class0, prob class1)|
//| We only need probabilities[1] = P(price hits target before stop).  |
//+------------------------------------------------------------------+
#property strict
#include "FeatureEngine.mqh"

class COnnxModel
  {
private:
   long   m_handle;
   bool   m_loaded;

public:
   COnnxModel() { m_handle = INVALID_HANDLE; m_loaded = false; }

   ~COnnxModel()
     {
      if(m_handle != INVALID_HANDLE)
         OnnxRelease(m_handle);
     }

   bool Load(const string filename)
     {
      // ONNX_DEFAULT loads from MQL5/Files/ (or common Files/ with the right flag)
      m_handle = OnnxCreate(filename, ONNX_DEFAULT);
      if(m_handle == INVALID_HANDLE)
        {
         PrintFormat("OnnxModel: failed to load '%s', error=%d", filename, GetLastError());
         m_loaded = false;
         return false;
        }

      // Fix the input shape: [1, FEATURE_COUNT] (batch size 1 for live inference)
      long input_shape[] = {1, FEATURE_COUNT};
      if(!OnnxSetInputShape(m_handle, 0, input_shape))
        {
         PrintFormat("OnnxModel: failed to set input shape, error=%d", GetLastError());
         OnnxRelease(m_handle);
         m_handle = INVALID_HANDLE;
         m_loaded = false;
         return false;
        }

      // Output 0 = label [1], output 1 = probabilities [1,2]
      long label_shape[] = {1};
      long proba_shape[] = {1, 2};
      OnnxSetOutputShape(m_handle, 0, label_shape);
      OnnxSetOutputShape(m_handle, 1, proba_shape);

      m_loaded = true;
      PrintFormat("OnnxModel: loaded '%s' successfully", filename);
      return true;
     }

   bool IsLoaded() const { return m_loaded; }

   //--- Runs inference, returns P(class==1) i.e. "target hit before stop".
   //--- Returns -1.0 on failure (caller must treat as "no trade").
   double PredictUpProbability(const double &features[])
     {
      if(!m_loaded) return -1.0;
      if(ArraySize(features) != FEATURE_COUNT) return -1.0;

      float input_data[];
      ArrayResize(input_data, FEATURE_COUNT);
      for(int i = 0; i < FEATURE_COUNT; i++)
         input_data[i] = (float)features[i];

      long label_out[];
      float proba_out[];

      bool ok = OnnxRun(m_handle, ONNX_DEFAULT, input_data, label_out, proba_out);
      if(!ok)
        {
         PrintFormat("OnnxModel: OnnxRun failed, error=%d", GetLastError());
         return -1.0;
        }
      if(ArraySize(proba_out) < 2) return -1.0;

      return (double)proba_out[1]; // P(class 1)
     }
  };
