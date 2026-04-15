# Real-Time Stock Price Predictor

A full-stack ML project that trains a stock direction classifier in Python and runs low-latency inference in C++ — the same architecture used in production quant systems.

```
BUY  2024-12-16  O=249.68  H=250.01  L=246.30  C=246.64  V=51694800  ✅
HOLD 2024-12-17  O=252.10  H=252.45  L=248.42  C=248.72  V=51356400  ✅
BUY  2024-12-20  O=253.11  H=253.61  L=244.36  C=246.69  V=147495300 ✅
SELL 2024-12-23  O=253.88  H=254.26  L=252.07  C=253.39  V=40858800  ✅
```

---

## Architecture

```
┌─────────────────────────────────┐     ┌──────────────────────────────────┐
│         Python (Training)        │     │        C++ (Inference)            │
│                                 │     │                                  │
│  yfinance → Feature Engineering │     │  CSVLoader → PriceWindow         │
│  RSI · MACD · Bollinger · SMA   │────▶│  FeatureCalculator → ModelRunner │
│  XGBoost → ONNX Export          │     │  ONNX Runtime → BUY/SELL/HOLD    │
│  StandardScaler → scaler_params │     │  Benchmark (µs timing)           │
└─────────────────────────────────┘     └──────────────────────────────────┘
         model.onnx + scaler_params.csv (bridge between both sides)
```

---

## Features

**Python pipeline (`python/train.py`)**
- Downloads historical OHLCV data via `yfinance`
- Engineers 8 technical indicators: SMA14, SMA50, RSI14, MACD, Bollinger Band Width, Price Deviation, Volume Change, HL Range
- Trains and compares 3 models: Logistic Regression, SVM, XGBoost
- Backtests strategy vs buy-and-hold
- Exports best model to ONNX format for C++ consumption
- Saves StandardScaler params for C++ feature normalization

**C++ inference engine (`cpp/stock_predictor.cpp`)**
- `PriceBar` — OHLCV data struct with operator overloading
- `PriceWindow` — fixed-size rolling buffer using `std::deque`
- `FeatureCalculator` — RSI (Wilder smoothing), MACD, Bollinger Width, SMA
- `ModelRunner` — ONNX Runtime inference with StandardScaler normalization
- `CSVLoader` — CSV file parser with error handling
- `Benchmark` — RAII microsecond timer using `std::chrono`
- `StockPredictor` — top-level orchestrator with BUY/SELL/HOLD summary

**Performance**
- Inference latency: ~6–12 µs per bar
- Backtest accuracy: ~60% on Dec 2024 AAPL data (vs 50% random baseline)
- Signal distribution on 477 bars: BUY 179 · SELL 181 · HOLD 117

---

## Project Structure

```
stock-predictor/
├── cpp/
│   └── stock_predictor.cpp     # C++ inference engine
├── python/
│   └── train.py                # ML training pipeline
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Quickstart

### Step 1 — Train the model (Python)

```bash
cd python
pip install -r ../requirements.txt
python train.py
```

This downloads AAPL data, trains the model, and outputs:
- `model.onnx` — the trained XGBoost model
- `scaler_params.csv` — feature normalization parameters
- `eda.png`, `confusion_matrices.png`, `feature_importance.png`, `backtest.png`

### Step 2 — Build the C++ engine

**macOS (Homebrew)**
```bash
brew install onnxruntime
cd cpp
g++ -std=c++17 -O2 -Wall stock_predictor.cpp \
    -I/opt/homebrew/include \
    -L/opt/homebrew/lib \
    -lonnxruntime \
    -o stock_predictor
```

**Linux**
```bash
sudo apt install libonnxruntime-dev
g++ -std=c++17 -O2 -Wall stock_predictor.cpp \
    -I/usr/include/onnxruntime \
    -lonnxruntime \
    -o stock_predictor
```

### Step 3 — Run inference

```bash
# Copy model files next to the binary
cp ../python/model.onnx .
cp ../python/scaler_params.csv .
cp ../python/AAPL.csv .

./stock_predictor AAPL.csv model.onnx
```

**Sample output:**
```
[Scaler] Loaded 8 features
[ModelRunner] Loaded model: model.onnx
[CSVLoader] Loaded 502 bars from AAPL.csv

  [BUY ]  2024-12-16  O=249.68  H=250.01  L=246.30  C=246.64  V=51694800
    SMA14        : 233.7110
    RSI14        : 77.7028
    MACD         : 5.2719
    Bollinger W  : 0.1470
    [Benchmark] inference took 9 µs

------------------------------------------------------------
  SUMMARY  BUY: 179   SELL: 181   HOLD: 117
------------------------------------------------------------
```

---

## C++ Concepts Covered

| Concept | Where |
|---|---|
| Structs + operator overloading | `PriceBar` + `operator<<` |
| OOP class design | All 7 classes |
| `std::deque` rolling buffer | `PriceWindow` |
| Static methods + STL algorithms | `FeatureCalculator` |
| File I/O + stream parsing | `CSVLoader` |
| RAII pattern | `Benchmark` destructor |
| `std::chrono` timing | `Benchmark` |
| Third-party library integration | ONNX Runtime in `ModelRunner` |
| `enum class` | `Signal` |

---

## Requirements

**Python**
```
yfinance
pandas
numpy
scikit-learn
xgboost
onnxmltools
onnxruntime
seaborn
matplotlib
```

**C++**
- C++17 or later
- ONNX Runtime (via Homebrew or apt)

---

## Roadmap

- [ ] `runLive()` — WebSocket real-time feed instead of CSV
- [ ] Periodic retraining script (retrain monthly on fresh data)
- [ ] Win rate tracker — log signal vs actual next-day outcome
- [ ] Raspberry Pi deployment — run inference on edge hardware

---

## Disclaimer

This project is for **educational purposes only**. The signals generated are not financial advice and should not be used for real trading decisions.

---

## Author

Built as a learning project covering machine learning, Python data pipelines, and C++ systems programming.
