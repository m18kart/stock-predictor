// ============================================================
//  Real-Time Stock Price Predictor — Complete Implementation
//  Concepts covered: OOP, STL containers, file I/O,
//  operator overloading, enums, chrono timing
// ============================================================

#include <iostream>
#include <deque>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <numeric>
#include <stdexcept>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <array>
#include </opt/homebrew/include/onnxruntime/onnxruntime_cxx_api.h>


// ============================================================
//  1. PriceBar — one OHLCV candle
// ============================================================

struct PriceBar {
    std::string date;
    double open;
    double high;
    double low;
    double close;
    long   volume;

    PriceBar(std::string da, double op, double hg,
             double lw, double clo, long vol)
        : date(da), open(op), high(hg),
          low(lw), close(clo), volume(vol) {}

    // Default constructor (needed by CSVLoader before fields are set)
    PriceBar() : open(0), high(0), low(0), close(0), volume(0) {}
};

// Free function — must be outside the struct
std::ostream& operator<<(std::ostream& os, const PriceBar& bar) {
    os << bar.date
       << "  O=" << std::fixed << std::setprecision(2) << bar.open
       << "  H=" << bar.high
       << "  L=" << bar.low
       << "  C=" << bar.close
       << "  V=" << bar.volume;
    return os;
}


// ============================================================
//  2. PriceWindow — fixed-size rolling buffer
// ============================================================

class PriceWindow {
public:
    explicit PriceWindow(size_t capacity) : capacity_(capacity) {}

    void push(double price) {
        if (buffer_.size() == capacity_) buffer_.pop_front();
        buffer_.push_back(price);
    }

    bool isFull() const { return buffer_.size() == capacity_; }

    double newest() const {
        if (buffer_.empty()) throw std::runtime_error("Window is empty");
        return buffer_.back();
    }

    double oldest() const {
        if (buffer_.empty()) throw std::runtime_error("Window is empty");
        return buffer_.front();
    }

    std::vector<double> toVector() const {
        return std::vector<double>(buffer_.begin(), buffer_.end());
    }

private:
    size_t             capacity_;
    std::deque<double> buffer_;
};


// ============================================================
//  3. FeatureCalculator — technical indicators
// ============================================================

class FeatureCalculator {
public:
    // --- Simple Moving Average ---
    static double sma(const std::vector<double>& prices) {
        if (prices.empty()) return 0.0;
        double sum = std::accumulate(prices.begin(), prices.end(), 0.0);
        return sum / static_cast<double>(prices.size());
    }

    // --- RSI (Wilder smoothing) ---
    // 1. Compute daily changes: change[i] = prices[i] - prices[i-1]
    // 2. Separate into gains (positive) and losses (absolute negative)
    // 3. Seed with simple average of first 'period' gains/losses
    // 4. Wilder smoothing: avg = (prev_avg*(period-1) + current) / period
    // 5. RS = avg_gain / avg_loss,  RSI = 100 - 100/(1+RS)
    static double rsi(const std::vector<double>& prices, int period = 14) {
        if (static_cast<int>(prices.size()) < period + 1) return 50.0;

        std::vector<double> gains, losses;
        for (size_t i = 1; i < prices.size(); ++i) {
            double change = prices[i] - prices[i - 1];
            gains.push_back(change > 0 ?  change : 0.0);
            losses.push_back(change < 0 ? -change : 0.0);
        }

        double avgGain = std::accumulate(gains.begin(),
                                         gains.begin() + period, 0.0) / period;
        double avgLoss = std::accumulate(losses.begin(),
                                         losses.begin() + period, 0.0) / period;

        for (size_t i = period; i < gains.size(); ++i) {
            avgGain = (avgGain * (period - 1) + gains[i])  / period;
            avgLoss = (avgLoss * (period - 1) + losses[i]) / period;
        }

        if (avgLoss < 1e-10) return 100.0;
        double rs = avgGain / avgLoss;
        return 100.0 - (100.0 / (1.0 + rs));
    }

    // --- MACD (12-period EMA minus 26-period EMA) ---
    // EMA[0] = prices[0]
    // EMA[i] = prices[i] * k + EMA[i-1] * (1-k),  k = 2/(period+1)
    static double macd(const std::vector<double>& prices) {
        if (prices.size() < 26) return 0.0;
        return ema(prices, 12) - ema(prices, 26);
    }

    // --- Bollinger Band Width = (upper - lower) / middle ---
    // upper/lower = sma +/- 2 * stddev
    static double bollingerWidth(const std::vector<double>& prices) {
        if (prices.size() < 2) return 0.0;
        double avg = sma(prices);
        double sq_sum = 0.0;
        for (double p : prices) sq_sum += (p - avg) * (p - avg);
        double stddev = std::sqrt(sq_sum / prices.size());
        return (avg > 0) ? (4.0 * stddev) / avg : 0.0;
    }

    // --- Price deviation from SMA ---
    static double priceDeviation(const std::vector<double>& prices) {
        double avg = sma(prices);
        if (avg == 0.0) return 0.0;
        return (prices.back() - avg) / avg;
    }

private:
    static double ema(const std::vector<double>& prices, int period) {
        double k   = 2.0 / (period + 1);
        double val = prices[0];
        for (size_t i = 1; i < prices.size(); ++i)
            val = prices[i] * k + val * (1.0 - k);
        return val;
    }
};


// ============================================================
//  4. FeatureVector — model input snapshot
// ============================================================

struct FeatureVector {
    double sma14        = 0.0;
    double sma50        = 0.0;
    double rsi14        = 0.0;
    double macd         = 0.0;
    double bollingerW   = 0.0;
    double deviation    = 0.0;
    double volumeChange = 0.0;
    double hlRange      = 0.0;

    void print() const {
        std::cout << std::fixed << std::setprecision(4)
                  << "    SMA14        : " << sma14        << "\n"
                  << "    SMA50        : " << sma50        << "\n"
                  << "    RSI14        : " << rsi14        << "\n"
                  << "    MACD         : " << macd         << "\n"
                  << "    Bollinger W  : " << bollingerW   << "\n"
                  << "    Deviation    : " << deviation    << "\n"
                  << "    Volume Chg   : " << volumeChange << "\n"
                  << "    HL Range     : " << hlRange      << "\n";
    }
};


// ============================================================
//  5. Signal
// ============================================================

enum class Signal { BUY, SELL, HOLD };

std::string signalToString(Signal s) {
    switch (s) {
        case Signal::BUY:  return "BUY ";
        case Signal::SELL: return "SELL";
        case Signal::HOLD: return "HOLD";
    }
    return "UNKNOWN";
}


// ============================================================
//  6. ModelRunner — stub (swap for ONNX when model is ready)
// ============================================================

class ModelRunner {
public:
    explicit ModelRunner(const std::string& modelPath,
                         const std::string& scalerPath = "scaler_params.csv")
        : env_(ORT_LOGGING_LEVEL_WARNING, "stock"),
          session_(env_, modelPath.c_str(), Ort::SessionOptions{})
    {
        loadScaler(scalerPath);
        std::cout << "[ModelRunner] Loaded model: " << modelPath << "\n";
        std::cout << "[ModelRunner] Loaded scaler: " << scalerPath << "\n\n";
    }

    Signal classify(const FeatureVector& fv) {
        // Step 1: scale features the same way Python's StandardScaler did
        std::vector<float> input = scaleFeatures(fv);

        // Step 2: create input tensor
        std::array<int64_t, 2> shape = {1, (int64_t)input.size()};
        Ort::MemoryInfo memInfo = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        memInfo, input.data(), input.size(),
        shape.data(), shape.size());

        const char* inputNames[]  = {"float_input"};
        const char* outputNames[] = {"label", "probabilities"};

        Ort::RunOptions runOptions;
        auto outputs = session_.Run(
            runOptions,
            inputNames,  &inputTensor, 1,
            outputNames, 2);   // fetch both outputs

        // Get probability of UP (class 1)
        float* probs = outputs[1].GetTensorMutableData<float>();
        float probUp = probs[1];   // index 0 = DOWN, index 1 = UP

        // HOLD zone: model is not confident enough
        if (probUp > 0.60f) return Signal::BUY;
        if (probUp < 0.40f) return Signal::SELL;
        return Signal::HOLD;
        }

private:
    Ort::Env     env_;
    mutable Ort::Session session_;

    // Scaler params loaded from scaler_params.csv
    std::vector<double> means_;
    std::vector<double> scales_;

    void loadScaler(const std::string& path) {
        std::ifstream file(path);
        if (!file.is_open())
            throw std::runtime_error("Cannot open scaler: " + path);

        std::string line;
        std::getline(file, line); // skip header: feature,mean,scale

        while (std::getline(file, line)) {
            std::istringstream ss(line);
            std::string feature, mean, scale;
            std::getline(ss, feature, ',');
            std::getline(ss, mean,    ',');
            std::getline(ss, scale,   ',');
            means_.push_back(std::stod(mean));
            scales_.push_back(std::stod(scale));
        }
        std::cout << "[Scaler] Loaded " << means_.size() << " features\n";
    }

    // Apply StandardScaler: z = (x - mean) / scale
    std::vector<float> scaleFeatures(const FeatureVector& fv) const {
        std::vector<double> raw = {
            fv.sma14,
            fv.sma50,
            fv.rsi14,
            fv.macd,
            fv.bollingerW,
            fv.deviation,
            fv.volumeChange,
            fv.hlRange
        };

        std::vector<float> scaled(raw.size());
        for (size_t i = 0; i < raw.size(); ++i)
            scaled[i] = static_cast<float>((raw[i] - means_[i]) / scales_[i]);
        return scaled;
    }
};


// ============================================================
//  7. CSVLoader — reads yfinance-style CSV
// ============================================================

class CSVLoader {
public:
    explicit CSVLoader(const std::string& filepath)
        : filepath_(filepath) {}

    std::vector<PriceBar> load() {
        std::ifstream file(filepath_);
        if (!file.is_open())
            throw std::runtime_error("Cannot open file: " + filepath_);

        std::vector<PriceBar> bars;
        std::string line;
        std::getline(file, line); // skip header
        std::getline(file, line); // skip header
        std::getline(file, line); // skip header

        while (std::getline(file, line)) {
            if (line.empty()) continue;
            PriceBar bar = parseLine(line);
            if (bar.close > 0.0)
                bars.push_back(bar);
        }

        std::cout << "[CSVLoader] Loaded " << bars.size()
                  << " bars from " << filepath_ << "\n";
        return bars;
    }

private:
    std::string filepath_;

    PriceBar parseLine(const std::string& line) const {
        std::istringstream ss(line);
        std::string token;
        PriceBar bar;
        // Expected columns: Date,Open,High,Low,Close,Volume
        std::getline(ss, bar.date,  ',');
        std::getline(ss, token, ','); bar.open   = std::stod(token);
        std::getline(ss, token, ','); bar.high   = std::stod(token);
        std::getline(ss, token, ','); bar.low    = std::stod(token);
        std::getline(ss, token, ','); bar.close  = std::stod(token);
        std::getline(ss, token, ','); bar.volume = std::stol(token);
        return bar;
    }
};


// ============================================================
//  8. Benchmark — RAII timer
// ============================================================

class Benchmark {
public:
    explicit Benchmark(const std::string& label)
        : label_(label),
          start_(std::chrono::high_resolution_clock::now()) {}

    ~Benchmark() {
        auto end = std::chrono::high_resolution_clock::now();
        auto us  = std::chrono::duration_cast<std::chrono::microseconds>
                       (end - start_).count();
        std::cout << "    [Benchmark] " << label_
                  << " took " << us << " µs\n";
    }

private:
    std::string label_;
    std::chrono::high_resolution_clock::time_point start_;
};


// ============================================================
//  9. StockPredictor — top-level orchestrator
// ============================================================

class StockPredictor {
public:
    StockPredictor(const std::string& modelPath,
                   size_t windowSize = 26)  // 26 needed for MACD
        : runner_(modelPath),
          window_(windowSize),
          window50_(50),
          volumeWindow_(2),
          windowSize_(windowSize) {}

    void runOnFile(const std::string& csvPath) {
        CSVLoader loader(csvPath);
        auto bars = loader.load();

        std::cout << "\n" << std::string(60, '-') << "\n"
                  << "  Predictor running on : " << csvPath        << "\n"
                  << "  Window size          : " << windowSize_     << " bars\n"
                  << std::string(60, '-') << "\n\n";

        int buy = 0, sell = 0, hold = 0;

        for (const auto& bar : bars) {
            window_.push(bar.close);
            window50_.push(bar.close);
            volumeWindow_.push(bar.volume);
            highs_.push_back(bar.high);
            lows_.push_back(bar.low);
            if (highs_.size() > 26) highs_.pop_front();
            if (lows_.size()  > 26) lows_.pop_front();
            if (!window_.isFull()) continue;

            FeatureVector fv;
            Signal        sig;
            {
                Benchmark b("inference");
                fv  = buildFeatures();
                sig = runner_.classify(fv);
            }

            logSignal(bar, sig);
            fv.print();
            std::cout << "\n";

            if (sig == Signal::BUY)  ++buy;
            if (sig == Signal::SELL) ++sell;
            if (sig == Signal::HOLD) ++hold;
        }

        std::cout << std::string(60, '-') << "\n"
                  << "  SUMMARY  BUY: "  << buy
                  << "   SELL: " << sell
                  << "   HOLD: " << hold << "\n"
                  << std::string(60, '-') << "\n";
    }

    // TODO: add runLive(const std::string& wsUrl)
    //       Open a WebSocket, parse each incoming tick,
    //       push to window_, call buildFeatures() + classify()

private:
    ModelRunner runner_;
    PriceWindow window_;
    size_t      windowSize_;
    PriceWindow window50_;
    PriceWindow volumeWindow_;
    std::deque<double> highs_;
    std::deque<double> lows_;

    FeatureVector buildFeatures() const {
        auto prices = window_.toVector();
        FeatureVector fv;
        fv.sma14        = FeatureCalculator::sma(prices);
        fv.sma50        = window50_.isFull()
                        ? FeatureCalculator::sma(window50_.toVector())
                        : fv.sma14;
        fv.rsi14        = FeatureCalculator::rsi(prices);
        fv.macd         = FeatureCalculator::macd(prices);
        fv.bollingerW   = FeatureCalculator::bollingerWidth(prices);
        fv.deviation    = FeatureCalculator::priceDeviation(prices);

        // Volume % change
        auto vols = volumeWindow_.toVector();
        fv.volumeChange = (vols.size() == 2 && vols[0] > 0)
                        ? (vols[1] - vols[0]) / vols[0]
                        : 0.0;

        // Intraday range = (high - low) / close
        if (!highs_.empty() && !lows_.empty())
            fv.hlRange = (highs_.back() - lows_.back()) / prices.back();

        return fv;
    }

    void logSignal(const PriceBar& bar, Signal sig) const {
        std::cout << "  [" << signalToString(sig) << "]  " << bar << "\n";
    }
};


// ============================================================
//  main
// ============================================================

int main(int argc, char* argv[]) {
    std::string csvPath   = (argc > 1) ? argv[1] : "AAPL.csv";
    std::string modelPath = (argc > 2) ? argv[2] : "model.onnx";

    try {
        StockPredictor predictor(modelPath);
        predictor.runOnFile(csvPath);
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] " << e.what() << "\n";
        return 1;
    }

    return 0;
}

// ============================================================
//  COMPILE:
//    g++ -std=c++17 -O2 -Wall stock_predictor.cpp -o stock_predictor
//    ./stock_predictor AAPL.csv
//
//  GENERATE AAPL.csv FROM PYTHON:
//    import yfinance as yf
//    df = yf.download("AAPL", period="1y")
//    df.to_csv("AAPL.csv")
//
//  NEXT MILESTONES:
//    [ ] Run this on AAPL.csv and read the output
//    [ ] Trace through rsi() on paper with 5-6 prices
//    [ ] Add a new indicator (e.g. stochastic oscillator)
//    [ ] Export your Python model to ONNX, swap the stub
//    [ ] Implement runLive() with a WebSocket tick feed
// ============================================================