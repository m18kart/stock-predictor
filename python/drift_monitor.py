# ============================================================
#  Drift Detection — Stock Predictor
#  Detects when incoming market data has statistically shifted
#  away from the distribution the model was trained on.
# ============================================================

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from datetime import datetime

warnings.filterwarnings("ignore")

# ============================================================
#  CONFIG
# ============================================================

FEATURE_COLS = ['SMA14', 'SMA50', 'RSI14', 'MACD',
                'BollingerW', 'Deviation', 'VolumeChange', 'HL_Range']

PRICE_SCALE_FEATURES = ['SMA14', 'SMA50']
DRIFT_CHECK_FEATURES = ['RSI14', 'MACD', 'BollingerW',
                        'Deviation', 'VolumeChange', 'HL_Range']

TICKER = 'AAPL'
REFERENCE_PERIOD = '3y'
CURRENT_WINDOW_DAYS = 30

PSI_THRESHOLD = 0.2
KS_PVALUE_THRESHOLD = 0.05


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

    df.dropna(inplace=True)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df


def population_stability_index(reference, current, bins=10):
    breakpoints = np.linspace(reference.min(), reference.max(), bins + 1)
    breakpoints[0]  = -np.inf
    breakpoints[-1] = np.inf

    ref_counts = np.histogram(reference, bins=breakpoints)[0]
    cur_counts = np.histogram(current,   bins=breakpoints)[0]

    ref_pct = np.where(ref_counts == 0, 0.0001, ref_counts / len(reference))
    cur_pct = np.where(cur_counts == 0, 0.0001, cur_counts / len(current))

    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return psi


def ks_test(reference, current):
    statistic, p_value = stats.ks_2samp(reference, current)
    return statistic, p_value


def check_feature_drift(reference_df, current_df, feature):
    ref_values = reference_df[feature].values
    cur_values = current_df[feature].values

    psi = population_stability_index(ref_values, cur_values)
    ks_stat, ks_pvalue = ks_test(ref_values, cur_values)

    drifted = (psi > PSI_THRESHOLD) or (ks_pvalue < KS_PVALUE_THRESHOLD)

    return {
        'feature'       : feature,
        'psi'           : round(float(psi), 4),
        'ks_statistic'  : round(float(ks_stat), 4),
        'ks_pvalue'     : round(float(ks_pvalue), 4),
        'ref_mean'      : round(float(np.mean(ref_values)), 4),
        'cur_mean'      : round(float(np.mean(cur_values)), 4),
        'drifted'       : bool(drifted),
    }


def run_drift_check(ticker=TICKER):
    print(f"\n{'='*60}")
    print(f"  Drift Detection Report — {ticker}")
    print(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*60}\n")

    print(f"  Loading reference data ({REFERENCE_PERIOD})...")
    full_df = build_features(ticker, REFERENCE_PERIOD)

    current_df   = full_df.tail(CURRENT_WINDOW_DAYS)
    reference_df = full_df.iloc[:-CURRENT_WINDOW_DAYS]

    print(f"  Reference period : {reference_df.index.min().date()} → {reference_df.index.max().date()}  ({len(reference_df)} rows)")
    print(f"  Current period   : {current_df.index.min().date()} → {current_df.index.max().date()}  ({len(current_df)} rows)")
    print()

    results = []
    for feature in FEATURE_COLS:
        result = check_feature_drift(reference_df, current_df, feature)
        result['category'] = 'price_scale' if feature in PRICE_SCALE_FEATURES else 'normalized'
        results.append(result)

        status = "DRIFTED" if result['drifted'] else "stable"
        flag   = "WARN" if result['drifted'] else "OK  "
        tag    = "[price-scale]" if result['category'] == 'price_scale' else "[normalized] "
        print(f"  {flag} {tag} {feature:15s}  PSI={result['psi']:6.3f}  "
              f"KS_p={result['ks_pvalue']:6.3f}  "
              f"ref_mean={result['ref_mean']:8.3f}  "
              f"cur_mean={result['cur_mean']:8.3f}  [{status}]")

    print(f"\n  Note: SMA14/SMA50 are price-scale features — they will always")
    print(f"  show drift for an appreciating/declining stock. The verdict")
    print(f"  below is based only on normalized/ratio features.\n")

    normalized_results = [r for r in results if r['category'] == 'normalized']
    n_drifted = sum(r['drifted'] for r in normalized_results)
    n_total   = len(normalized_results)
    print(f"  {n_drifted}/{n_total} normalized features show significant drift")

    if n_drifted >= 3:
        print("  WARN  RECOMMENDATION: Retrain model — multiple features have drifted")
        verdict = "RETRAIN_RECOMMENDED"
    elif n_drifted >= 1:
        print("  WARN  RECOMMENDATION: Monitor closely — some drift detected")
        verdict = "MONITOR"
    else:
        print("  OK  RECOMMENDATION: Model is healthy — no action needed")
        verdict = "HEALTHY"

    print(f"{'='*60}\n")

    report = {
        'ticker'      : ticker,
        'generated_at': datetime.now().isoformat(),
        'reference_period': {
            'start': str(reference_df.index.min().date()),
            'end'  : str(reference_df.index.max().date()),
            'rows' : len(reference_df),
        },
        'current_period': {
            'start': str(current_df.index.min().date()),
            'end'  : str(current_df.index.max().date()),
            'rows' : len(current_df),
        },
        'features'      : results,
        'n_drifted_normalized': n_drifted,
        'total_normalized_features': n_total,
        'verdict'       : verdict,
        'note': 'Verdict based on normalized features only (RSI, MACD, '
                'BollingerW, Deviation, VolumeChange, HL_Range). '
                'SMA14/SMA50 excluded — raw price-scale features always '
                'drift for trending stocks and are not actionable signals.',
    }

    os.makedirs('drift_reports', exist_ok=True)
    report_path = f"drift_reports/drift_{ticker}_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"  Report saved -> {report_path}")

    plot_drift(normalized_results, ticker)

    return report


def plot_drift(results, ticker):
    features = [r['feature'] for r in results]
    psi_vals = [r['psi'] for r in results]
    colors   = ['#D85A30' if r['drifted'] else '#1D9E75' for r in results]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(features, psi_vals, color=colors)
    ax.axvline(PSI_THRESHOLD, color='red', linestyle='--', linewidth=1,
              label=f'Drift threshold (PSI={PSI_THRESHOLD})')
    ax.set_xlabel('Population Stability Index (PSI)')
    ax.set_title(f'Feature Drift Report — {ticker}')
    ax.legend()

    for bar, val in zip(bars, psi_vals):
        ax.text(val + 0.005, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=9)

    plt.tight_layout()
    plot_dir = os.environ.get('PLOT_DIR', 'assets')
    os.makedirs(plot_dir, exist_ok=True)
    out_path = os.path.join(plot_dir, 'drift_report.png')
    plt.savefig(out_path, dpi=150)
    print(f"  Plot saved -> {out_path}")


if __name__ == '__main__':
    ticker = sys.argv[1] if len(sys.argv) > 1 else TICKER
    report = run_drift_check(ticker)

    if report['verdict'] == 'RETRAIN_RECOMMENDED':
        sys.exit(1)
    sys.exit(0)
