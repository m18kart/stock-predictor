# ============================================================
#  Real-Time Stock Price Predictor — Python ML Pipeline
#  Covers: EDA, feature engineering, model training,
#          evaluation, and ONNX export for C++ inference
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import onnxruntime as ort

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (classification_report, confusion_matrix,
                              ConfusionMatrixDisplay, roc_auc_score)
from xgboost import XGBClassifier

import warnings
warnings.filterwarnings("ignore")

sess = ort.InferenceSession("model.onnx")
print("Inputs :", [i.name for i in sess.get_inputs()])
print("Outputs:", [o.name for o in sess.get_outputs()])

# ============================================================
#  1. LOAD DATA
# ============================================================

df = pd.read_csv('AAPL.csv', skiprows=1)
df.rename(columns={df.columns[0]: 'Date'}, inplace=True)
df = df[df['Date'].str.match(r'\d{4}-\d{2}-\d{2}', na=False)]  # drop ticker rows
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
df.columns = ['Close', 'High', 'Low', 'Open', 'Volume']  # assign names directly

# Convert all columns to numeric (they're being read as strings)
df = df.apply(pd.to_numeric, errors='coerce')
df.dropna(inplace=True)

print(df.head())
print(df.dtypes)

# yfinance sometimes adds a ticker row — drop it if present
if df.index.dtype == object:
    df = df[pd.to_datetime(df.index, errors='coerce').notna()]
    df.index = pd.to_datetime(df.index)

# Keep only the columns we need
df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
df.dropna(inplace=True)

print("Shape      :", df.shape)
print("Date range :", df.index.min(), "→", df.index.max())
print(df.describe().round(2))


# ============================================================
#  2. EDA — Exploratory Data Analysis
# ============================================================

fig, axes = plt.subplots(3, 1, figsize=(14, 10))
fig.suptitle('AAPL — Exploratory Data Analysis', fontsize=14)

# 2a. Close price over time
axes[0].plot(df.index, df['Close'], color='steelblue', linewidth=1)
axes[0].set_title('Close Price')
axes[0].set_ylabel('USD')

# 2b. Daily return distribution
df['DailyReturn'] = df['Close'].pct_change()
axes[1].hist(df['DailyReturn'].dropna(), bins=80, color='salmon', edgecolor='white')
axes[1].set_title('Daily Return Distribution')
axes[1].set_xlabel('Return')

# 2c. Volume
axes[2].bar(df.index, df['Volume'], color='mediumseagreen', width=1.5)
axes[2].set_title('Trading Volume')
axes[2].set_ylabel('Shares')

plt.tight_layout()
plt.savefig('eda.png', dpi=150)
plt.show()
print("[EDA] Saved → eda.png")


# ============================================================
#  3. FEATURE ENGINEERING
#  These match the indicators computed in C++ FeatureCalculator
# ============================================================

def compute_rsi(series, period=14):
    delta  = series.diff()
    gain   = delta.clip(lower=0)
    loss   = -delta.clip(upper=0)
    # Wilder smoothing (same as C++ implementation)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs  = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def compute_macd(series, fast=12, slow=26):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    return ema_fast - ema_slow

def compute_bollinger_width(series, period=20):
    sma    = series.rolling(period).mean()
    stddev = series.rolling(period).std()
    upper  = sma + 2 * stddev
    lower  = sma - 2 * stddev
    return (upper - lower) / sma  # normalised width

# --- Apply all indicators ---
df['SMA14']         = df['Close'].rolling(14).mean()
df['SMA50']         = df['Close'].rolling(50).mean()
df['RSI14']         = compute_rsi(df['Close'])
df['MACD']          = compute_macd(df['Close'])
df['BollingerW']    = compute_bollinger_width(df['Close'])
df['Deviation']     = (df['Close'] - df['SMA14']) / df['SMA14']
df['VolumeChange']  = df['Volume'].pct_change()
df['HL_Range']      = (df['High'] - df['Low']) / df['Close']  # intraday range

# --- Label: 1 if next day close is higher, 0 otherwise ---
df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

df.dropna(inplace=True)

print("\nFeature sample:")
print(df[['Close','SMA14','RSI14','MACD','BollingerW','Target']].tail(5).round(3))
print("\nClass balance:\n", df['Target'].value_counts())

print(df['Target'].value_counts())
ratio = (df['Target'] == 0).sum() / (df['Target'] == 1).sum()
print(f"scale_pos_weight should be: {ratio:.2f}")

# ============================================================
#  4. FEATURE CORRELATION HEATMAP
# ============================================================

feature_cols = ['SMA14', 'SMA50', 'RSI14', 'MACD',
                'BollingerW', 'Deviation', 'VolumeChange', 'HL_Range']

plt.figure(figsize=(10, 7))
sns.heatmap(df[feature_cols + ['Target']].corr().round(2),
            annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5)
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.savefig('correlation.png', dpi=150)
plt.show()
print("[EDA] Saved → correlation.png")


# ============================================================
#  5. TRAIN / TEST SPLIT
#  Important: no shuffle — time-series order must be preserved
# ============================================================

X = df[feature_cols].values
y = df['Target'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

print(f"\nTrain size : {X_train.shape[0]} rows")
print(f"Test  size : {X_test.shape[0]} rows")


# ============================================================
#  6. TRAIN THREE MODELS & COMPARE
# ============================================================

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'SVM (RBF)'          : SVC(kernel='rbf', probability=True),
    'XGBoost'            : XGBClassifier(n_estimators=200,
                                          max_depth=4,
                                          learning_rate=0.05,
                                          scale_pos_weight=1,
                                          eval_metric='logloss',
                                          random_state=42),
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = (y_pred == y_test).mean()
    auc = roc_auc_score(y_test, y_prob)
    cv  = cross_val_score(model, X_train, y_train, cv=5,
                          scoring='accuracy').mean()

    results[name] = {'Accuracy': acc, 'ROC-AUC': auc, 'CV Accuracy': cv}
    print(f"\n{'='*40}")
    print(f"  {name}")
    print(f"{'='*40}")
    print(classification_report(y_test, y_pred,
                                 target_names=['DOWN','UP']))

results_df = pd.DataFrame(results).T.round(4)
print("\nModel comparison:")
print(results_df.sort_values('ROC-AUC', ascending=False))


# ============================================================
#  7. CONFUSION MATRICES
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle('Confusion Matrices', fontsize=13)

for ax, (name, model) in zip(axes, models.items()):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=['DOWN', 'UP'])
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(name)

plt.tight_layout()
plt.savefig('confusion_matrices.png', dpi=150)
plt.show()
print("[Eval] Saved → confusion_matrices.png")


# ============================================================
#  8. FEATURE IMPORTANCE (XGBoost)
# ============================================================

xgb_model = models['XGBoost']
importances = pd.Series(xgb_model.feature_importances_,
                         index=feature_cols).sort_values(ascending=True)

plt.figure(figsize=(8, 5))
importances.plot(kind='barh', color='steelblue')
plt.title('XGBoost Feature Importance')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150)
plt.show()
print("[Eval] Saved → feature_importance.png")


# ============================================================
#  9. BACKTEST — simulate trading signals on test period
# ============================================================

test_dates  = df.index[-len(y_test):]
test_prices = df['Close'].values[-len(y_test):]
y_pred_xgb  = models['XGBoost'].predict(X_test)

# Strategy: buy if model says UP, stay out otherwise
strategy_returns = np.where(y_pred_xgb == 1,
                             np.diff(test_prices, prepend=test_prices[0])
                             / test_prices, 0)
buyhold_returns  = np.diff(test_prices, prepend=test_prices[0]) / test_prices

strategy_cum = (1 + strategy_returns).cumprod()
buyhold_cum  = (1 + buyhold_returns).cumprod()

plt.figure(figsize=(13, 5))
plt.plot(test_dates, buyhold_cum,  label='Buy & Hold',      color='gray')
plt.plot(test_dates, strategy_cum, label='XGBoost Strategy', color='steelblue')
plt.title('Backtest: XGBoost Strategy vs Buy & Hold')
plt.ylabel('Cumulative Return')
plt.legend()
plt.tight_layout()
plt.savefig('backtest.png', dpi=150)
plt.show()

final_strat = strategy_cum[-1] - 1
final_bh    = buyhold_cum[-1]  - 1
print(f"\n[Backtest] Strategy return : {final_strat:.2%}")
print(f"[Backtest] Buy & Hold      : {final_bh:.2%}")


# ============================================================
#  10. EXPORT BEST MODEL TO ONNX  (bridge to C++)
# ============================================================

try:
    from onnxmltools import convert_xgboost
    from onnxmltools.convert.common.data_types import FloatTensorType

    n_features   = X_train.shape[1]
    initial_type = [('float_input', FloatTensorType([None, n_features]))]

    onnx_model = convert_xgboost(models['XGBoost'], initial_types=initial_type)

    with open('model.onnx', 'wb') as f:
        f.write(onnx_model.SerializeToString())

    print("[ONNX] Model exported → model.onnx")

except ImportError:
    print("[ONNX] Run: pip install onnxmltools")


# ============================================================
#  11. SAVE SCALER PARAMS FOR C++
#  Your C++ FeatureCalculator must normalise inputs the same way
# ============================================================

scaler_params = pd.DataFrame({
    'feature': feature_cols,
    'mean'   : scaler.mean_,
    'scale'  : scaler.scale_
})
scaler_params.to_csv('scaler_params.csv', index=False)
print("\n[Scaler] Parameters saved → scaler_params.csv")
print("         Load these in C++ before calling ModelRunner::predict()")
print(scaler_params.to_string(index=False))