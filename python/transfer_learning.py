# ============================================================
#  Transfer Learning — Index (parent) → Stock (child)
#  Parent: S&P 500 (^GSPC)   Child: AAPL
#  Transfer mechanism: parent probability as 9th input feature
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import mlflow
import mlflow.xgboost
import warnings
import matplotlib
matplotlib.use('Agg')  # non-interactive backend — saves to file instead of popup
import matplotlib.pyplot as plt

from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report
from onnxmltools import convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType
import yfinance as yf

warnings.filterwarnings("ignore")

# ============================================================
#  CONFIG
# ============================================================

FEATURE_COLS = ['SMA14', 'SMA50', 'RSI14', 'MACD',
                'BollingerW', 'Deviation', 'VolumeChange', 'HL_Range']

PARENT_TICKER = '^GSPC'
CHILD_TICKER  = 'AAPL'
PARENT_PERIOD = '5y'
CHILD_PERIOD  = '3y'

PARENT_PARAMS = {
    'n_estimators'    : 500,
    'max_depth'       : 5,
    'learning_rate'   : 0.03,
    'subsample'       : 0.8,
    'colsample_bytree': 0.8,
    'eval_metric'     : 'logloss',
    'random_state'    : 42,
}

CHILD_ROUNDS = 150
CHILD_LR     = 0.01
CHILD_DEPTH  = 3


# ============================================================
#  FEATURE ENGINEERING  (identical to stock_prediction.py)
# ============================================================

def compute_rsi(series, period=14):
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs       = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def compute_macd(series, fast=12, slow=26):
    return (series.ewm(span=fast, adjust=False).mean() -
            series.ewm(span=slow, adjust=False).mean())

def compute_bollinger_width(series, period=20):
    sma    = series.rolling(period).mean()
    stddev = series.rolling(period).std()
    return ((sma + 2*stddev) - (sma - 2*stddev)) / sma

def build_features(ticker, period):
    print(f"  Downloading {ticker} ({period})...")
    raw = yf.download(ticker, period=period, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = raw[['Open','High','Low','Close','Volume']].copy()
    df = df.apply(pd.to_numeric, errors='coerce').dropna()

    df['SMA14']        = df['Close'].rolling(14).mean()
    df['SMA50']        = df['Close'].rolling(50).mean()
    df['RSI14']        = compute_rsi(df['Close'])
    df['MACD']         = compute_macd(df['Close'])
    df['BollingerW']   = compute_bollinger_width(df['Close'])
    df['Deviation']    = (df['Close'] - df['SMA14']) / df['SMA14']
    df['VolumeChange'] = df['Volume'].pct_change()
    df['HL_Range']     = (df['High'] - df['Low']) / df['Close']
    df['Target']       = (df['Close'].shift(-1) > df['Close']).astype(int)

    df.dropna(inplace=True)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    assert df[FEATURE_COLS].isnull().sum().sum() == 0, \
        f"NaNs still present in {ticker}"

    return df[FEATURE_COLS].values, df['Target'].values, df.index, df

def time_split(X, y, test_ratio=0.2):
    split = int(len(X) * (1 - test_ratio))
    return X[:split], X[split:], y[:split], y[split:]

def scale(X_train, X_test):
    scaler  = StandardScaler()
    return scaler.fit_transform(X_train), scaler.transform(X_test), scaler

def evaluate(name, y_test, y_pred, y_prob):
    acc = (y_pred == y_test).mean()
    auc = roc_auc_score(y_test, y_prob)
    print(f"\n  {name}")
    print(f"  Accuracy : {acc:.4f}   ROC-AUC : {auc:.4f}")
    print(classification_report(y_test, y_pred,
                                 target_names=['DOWN','UP'], digits=3))
    return acc, auc


# ============================================================
#  PHASE 1 — Train parent model on S&P 500
# ============================================================

def train_parent(exp_id):
    print("\n" + "="*55)
    print("  PHASE 1 — Parent model on S&P 500 (^GSPC)")
    print("="*55)

    X, y, _, _ = build_features(PARENT_TICKER, PARENT_PERIOD)
    X_train, X_test, y_train, y_test = time_split(X, y)
    X_train_s, X_test_s, scaler      = scale(X_train, X_test)
    print(f"  Train : {len(X_train)}   Test : {len(X_test)}")

    with mlflow.start_run(run_name="parent_GSPC", experiment_id=exp_id):
        parent = XGBClassifier(**PARENT_PARAMS)
        parent.fit(X_train_s, y_train,
                   eval_set=[(X_test_s, y_test)],
                   verbose=False)

        y_pred = parent.predict(X_test_s)
        y_prob = parent.predict_proba(X_test_s)[:, 1]
        acc, auc = evaluate("Parent (^GSPC)", y_test, y_pred, y_prob)

        mlflow.log_params({'ticker': PARENT_TICKER, 'role': 'parent',
                           'n_estimators': PARENT_PARAMS['n_estimators'],
                           'max_depth': PARENT_PARAMS['max_depth']})
        mlflow.log_metrics({'accuracy': round(acc,4), 'roc_auc': round(auc,4)})
        mlflow.xgboost.log_model(parent, "parent_model")

        # Export parent ONNX
        n = len(FEATURE_COLS)
        onnx_m = convert_xgboost(parent,
                     initial_types=[('float_input', FloatTensorType([None, n]))])
        with open('model_parent_GSPC.onnx', 'wb') as f:
            f.write(onnx_m.SerializeToString())
        print("  Exported → model_parent_GSPC.onnx")

    return parent, scaler, acc, auc


# ============================================================
#  PHASE 2 — Fine-tune child model on AAPL
#
#  Transfer mechanism: parent probability added as 9th feature
#  Child model sees parent's confidence alongside RSI/MACD/etc.
#  This avoids XGBoost base_margin ONNX compatibility issues
#  while achieving the same transfer learning effect.
# ============================================================

def train_child(parent_model, exp_id):
    print("\n" + "="*55)
    print(f"  PHASE 2 — Child model on {CHILD_TICKER} (transfer)")
    print("="*55)

    X, y, _, _ = build_features(CHILD_TICKER, CHILD_PERIOD)
    X_train, X_test, y_train, y_test = time_split(X, y)
    X_train_s, X_test_s, child_scaler = scale(X_train, X_test)
    print(f"  Train : {len(X_train)}   Test : {len(X_test)}")

    # ── Transfer step ──────────────────────────────────────
    # Parent's probability becomes the 9th input feature.
    # Child learns: "given RSI=X, MACD=Y, and parent thinks 72%
    # chance UP — what is the AAPL-specific answer?"
    parent_train_prob = parent_model.predict_proba(X_train_s)[:, 1]
    parent_test_prob  = parent_model.predict_proba(X_test_s)[:, 1]

    X_train_aug = np.hstack([X_train_s, parent_train_prob.reshape(-1,1)])
    X_test_aug  = np.hstack([X_test_s,  parent_test_prob.reshape(-1,1)])

    # ── Also evaluate parent alone on AAPL (baseline) ──────
    parent_preds = parent_model.predict(X_test_s)
    parent_probs = parent_model.predict_proba(X_test_s)[:, 1]
    parent_alone_acc, parent_alone_auc = evaluate(
        f"Parent alone on {CHILD_TICKER}", y_test, parent_preds, parent_probs)

    # ── Train child with 9 features ────────────────────────
    with mlflow.start_run(run_name=f"child_{CHILD_TICKER}", experiment_id=exp_id):
        child = XGBClassifier(
            n_estimators    = CHILD_ROUNDS,
            max_depth       = CHILD_DEPTH,
            learning_rate   = CHILD_LR,
            subsample       = 0.8,
            colsample_bytree= 0.8,
            eval_metric     = 'logloss',
            random_state    = 42
        )
        child.fit(X_train_aug, y_train,
                  eval_set=[(X_test_aug, y_test)],
                  verbose=False)

        y_pred = child.predict(X_test_aug)
        y_prob = child.predict_proba(X_test_aug)[:, 1]
        acc, auc = evaluate(f"Child ({CHILD_TICKER}) with transfer",
                             y_test, y_pred, y_prob)

        mlflow.log_params({
            'ticker'         : CHILD_TICKER,
            'transfer_from'  : PARENT_TICKER,
            'n_features'     : 9,
            'fine_tune_rounds': CHILD_ROUNDS,
            'max_depth'      : CHILD_DEPTH,
            'role'           : 'child',
        })
        mlflow.log_metrics({
            'child_accuracy'       : round(acc, 4),
            'child_roc_auc'        : round(auc, 4),
            'parent_alone_accuracy': round(parent_alone_acc, 4),
            'accuracy_lift'        : round(acc - parent_alone_acc, 4),
        })
        mlflow.xgboost.log_model(child, "child_model")

        # ── ONNX export — standard XGBClassifier, no hacks ─
        n = X_train_aug.shape[1]   # 9
        onnx_m = convert_xgboost(child,
                     initial_types=[('float_input', FloatTensorType([None, n]))])
        fname = f'model_child_{CHILD_TICKER}.onnx'
        with open(fname, 'wb') as f:
            f.write(onnx_m.SerializeToString())
        print(f"  Exported → {fname}")

        # ── Save scaler — 9 features including parent_prob ─
        pd.DataFrame({
            'feature': FEATURE_COLS + ['parent_prob'],
            'mean'   : list(child_scaler.mean_) + [float(parent_train_prob.mean())],
            'scale'  : list(child_scaler.scale_) + [float(parent_train_prob.std())]
        }).to_csv(f'scaler_child_{CHILD_TICKER}.csv', index=False)
        print(f"  Scaler  → scaler_child_{CHILD_TICKER}.csv")
        print(f"  Note: C++ ModelRunner now needs 9 inputs (8 features + parent_prob)")

    return acc, auc, parent_alone_acc, parent_alone_auc


# ============================================================
#  COMPARISON CHART
# ============================================================

def plot_comparison(parent_acc, parent_auc,
                    child_acc,  child_auc,
                    standalone_acc=0.60, standalone_auc=0.55):
    labels = ['Accuracy', 'ROC-AUC']
    x = np.arange(len(labels))
    w = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w, [parent_acc,     parent_auc],     w,
           label='Parent alone on AAPL', color='#B4B2A9')
    ax.bar(x,     [standalone_acc, standalone_auc], w,
           label='Standalone XGBoost',   color='#378ADD')
    ax.bar(x + w, [child_acc,      child_auc],      w,
           label='Child (transfer)',      color='#1D9E75')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylim(0.4, 0.8)
    ax.set_title(f'Transfer Learning: ^GSPC → {CHILD_TICKER}', fontsize=13)
    ax.set_ylabel('Score')
    ax.axhline(0.5, color='red', linewidth=0.8, linestyle='--')
    ax.legend()

    for bar in ax.patches:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2,
                h + 0.004, f'{h:.3f}',
                ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    import os
    plot_dir = os.environ.get('PLOT_DIR', 'assets')
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir, 'transfer_report.png'), dpi=150)
    plt.show()
    print("[Plot] Saved → transfer_report.png")


# ============================================================
#  MAIN
# ============================================================

if __name__ == '__main__':
    db_path = os.environ.get('MLFLOW_TRACKING_URI', f"sqlite:///{os.path.abspath('mlflow.db')}")
    mlflow.set_tracking_uri(db_path)

    exp_name   = "stock-transfer-learning"
    experiment = mlflow.get_experiment_by_name(exp_name)
    exp_id     = (experiment.experiment_id if experiment
                  else mlflow.create_experiment(exp_name))

    print(f"MLflow  → sqlite:///{db_path}")
    print(f"View    → mlflow ui --backend-store-uri sqlite:///{db_path}")

    # Phase 1
    parent_model, parent_scaler, p_acc, p_auc = train_parent(exp_id)

    # Phase 2
    c_acc, c_auc, pa_acc, pa_auc = train_child(parent_model, exp_id)

    # Chart
    plot_comparison(pa_acc, pa_auc, c_acc, c_auc)

    # Summary
    print("\n" + "="*55)
    print("  SUMMARY")
    print("="*55)
    print(f"  Parent on ^GSPC           : acc={p_acc:.4f}  auc={p_auc:.4f}")
    print(f"  Parent alone on AAPL      : acc={pa_acc:.4f}  auc={pa_auc:.4f}")
    print(f"  Child (transfer) on AAPL  : acc={c_acc:.4f}  auc={c_auc:.4f}")
    print(f"  Lift from transfer        : {c_acc - pa_acc:+.4f}")
    print()
    print("  Ready for C++ inference:")
    print(f"  ./stock_predictor AAPL.csv model_child_{CHILD_TICKER}.onnx")

# ============================================================
#  INSTALL:  pip install mlflow yfinance xgboost onnxmltools scikit-learn
#  RUN:      python python/transfer_learning.py
#  UI:       mlflow ui --backend-store-uri sqlite:///mlflow.db
# ============================================================