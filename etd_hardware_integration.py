"""
ETD Hardware-Software Integration Script
==========================================
Two honestly-separated outputs, per design decision with supervisor:

1. REAL-TIME LAYER (works right now, no ML needed):
   Reads the Arduino's live A/B/GAP stream and reports instant bypass-theft
   detection, matching the hardware's own buzzer/LED logic. Genuine live detection.

2. ACCUMULATED-PATTERN LAYER (the ML classification):
   The trained model classifies a customer from a LONG consumption history, not
   from one instant sample. This script logs live readings toward a real
   accumulating history, and SEPARATELY demonstrates the model's classification
   on real held-out SGCC customers — clearly labelled as NOT from the live session.

IMPORTANT — corrected against the actual saved model (verified 2026-06-25):
   The earlier draft of this script assumed the model wanted 39 features
   (26 raw + 8 stats + 5 FFT) via KNNImputer -> MinMaxScaler. The saved
   etd_model.pkl actually expects:
       * 1,034 daily readings as input (not 26)
       * 173 ENGINEERED features (not 39)
       * imputer = None (row-wise median imputation is used instead)
       * scaler = StandardScaler fit on the 173 ENGINEERED features
         (not a MinMaxScaler on raw daily readings)
       * pipeline order: raw -> impute -> engineer -> scale -> models
   To guarantee train/inference consistency, this script now IMPORTS the exact
   functions used in training from etd_pipeline.py rather than reimplementing them.

Author: Tunji (190408030)
"""

import pickle
import os
import csv
import time
import argparse
from datetime import datetime

import numpy as np
import pandas as pd

# Import the EXACT training-time functions — do not reimplement these.
from etd_pipeline import impute_rowwise_median, engineer_features, _calendar_masks

# ─────────────────────────────────────────────
# CONFIG — edit these to match your setup
# ─────────────────────────────────────────────

SERIAL_PORT = 'COM10'          # change to match Tools > Port in Arduino IDE
BAUD_RATE = 115200             # matches Serial.begin(115200) in your sketch
MODEL_PATH = 'etd_model.pkl'
LOG_FILE = 'live_readings_log.csv'
HISTORICAL_DEMO_FILE = 'demo_historical_sample.csv'

WINDOW_DAYS = 1034             # the model's ACTUAL training window (~2.8 years)
BYPASS_THRESHOLD_W = 10.0      # matches the Arduino's THEFT_THRESHOLD

# Decision threshold from the software-side threshold analysis: 0.435 maximises
# recall while holding FPR at the 0.0103 target (vs 0.5 default -> FPR 0.0058).
DECISION_THRESHOLD = 0.435


# ─────────────────────────────────────────────
# LAYER 1 — Real-time hardware bypass detection
# ─────────────────────────────────────────────

def parse_line(line):
    """
    Parse 'A:11.20 B:11.00 C:0.00 GAP:0.20' into a dict of floats.

    Returns None for unusable lines. A line is unusable if it carries no GAP
    field, because GAP is what the bypass decision is made on.

    NOTE: missing fields are left ABSENT, never defaulted to 0.0. Defaulting
    would display a fabricated '0.00W' for a sensor whose reading was never
    transmitted — indistinguishable from a genuine zero reading.
    """
    try:
        values = {}
        for p in line.strip().split():
            if ':' not in p:
                continue
            key, val = p.split(':', 1)
            values[key] = float(val)
    except (ValueError, AttributeError):
        return None
    return values if 'GAP' in values else None


def _fmt(values, key):
    """Format a field, or '--' if it never arrived (never fake a zero)."""
    return f"{values[key]:6.2f}W" if key in values else "    --  "


def report_realtime_status(values):
    """Instant hardware-layer bypass check — no ML involved. Same logic already
    running on the Arduino itself (GAP is computed on the Arduino)."""
    gap = values['GAP']                      # guaranteed present by parse_line
    theft = gap > BYPASS_THRESHOLD_W
    status = "BYPASS DETECTED" if theft else "normal"
    flag = " ⚠" if theft else ""
    missing = [k for k in ('A', 'B') if k not in values]
    warn = f"   [missing from stream: {','.join(missing)}]" if missing else ""
    print(f"[REAL-TIME]  A={_fmt(values,'A')}  B={_fmt(values,'B')}  "
          f"GAP={gap:6.2f}W  -> {status}{flag}{warn}")


# ─────────────────────────────────────────────
# LAYER 2a — Log live readings toward a real accumulating history
# ─────────────────────────────────────────────

def log_reading(values):
    """Append a timestamped reading. A real deployment would aggregate these into
    one consumption-per-day value — the shape the model expects."""
    exists = os.path.isfile(LOG_FILE)
    with open(LOG_FILE, 'a', newline='') as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(['timestamp', 'A_watts', 'B_watts', 'C_watts', 'gap_watts'])
        w.writerow([datetime.now().isoformat(),
                    values.get('A', 0.0), values.get('B', 0.0),
                    values.get('C', 0.0), values.get('GAP', 0.0)])


def days_logged_so_far():
    """Count distinct calendar days present in the log so far."""
    if not os.path.isfile(LOG_FILE):
        return 0
    try:
        df = pd.read_csv(LOG_FILE, parse_dates=['timestamp'])
    except Exception:
        return 0
    return 0 if df.empty else df['timestamp'].dt.date.nunique()


# ─────────────────────────────────────────────
# LAYER 2b — Run the trained model on a completed consumption history
# ─────────────────────────────────────────────

def classify_history(daily_rows, model_bundle, cal):
    """
    Runs the FULL trained pipeline on one or more customers' daily histories:
        raw (n, 1034) -> row-wise median impute -> engineer 173 features
                      -> StandardScaler -> 3 base classifiers -> LR meta-learner

    daily_rows : (n, 1034) array of daily consumption values (NaNs allowed)
    cal        : calendar masks from _calendar_masks(date_headers)
    Returns    : array of theft probabilities
    """
    base = model_bundle['base_classifiers']
    scaler = model_bundle['scaler']
    meta = model_bundle['meta_learner']

    raw = np.asarray(daily_rows, dtype=np.float32).reshape(len(daily_rows), -1)
    imputed = impute_rowwise_median(raw)
    feats = engineer_features(imputed, cal)
    feats = scaler.transform(feats)

    probs = np.column_stack([
        base['XGBoost'].predict_proba(feats)[:, 1],
        base['RandomForest'].predict_proba(feats)[:, 1],
        base['MLP'].predict_proba(feats)[:, 1],
    ])
    return meta.predict_proba(probs)[:, 1]


def run_historical_demo(model_bundle):
    """Loads real held-out SGCC customers and shows what the model's
    classification looks like on a complete consumption history. This is the
    honest stand-in for 'what live data would look like after full accumulation'."""

    if not os.path.isfile(HISTORICAL_DEMO_FILE):
        print(f"\n[INFO] No demo file at '{HISTORICAL_DEMO_FILE}'. "
              f"Run export_demo_sample.py to create it.\n")
        return

    df = pd.read_csv(HISTORICAL_DEMO_FILE)
    truth = df.pop('TRUE_FLAG').values if 'TRUE_FLAG' in df.columns else None

    date_headers = list(df.columns)
    if len(date_headers) != WINDOW_DAYS:
        print(f"\n[WARN] Demo file has {len(date_headers)} day-columns but the "
              f"model expects {WINDOW_DAYS}. Classification skipped.\n")
        return

    cal = _calendar_masks(date_headers)
    probs = classify_history(df.values, model_bundle, cal)

    print("\n--- Accumulated-Pattern Classification Demo ---")
    print(f"{len(df)} real held-out SGCC customers "
          f"({WINDOW_DAYS}-day histories, unseen during training)")
    print(f"Decision threshold: {DECISION_THRESHOLD} "
          f"(tuned for FPR <= 0.0103)\n")

    correct = 0
    for i, prob in enumerate(probs):
        pred = 'THEFT ' if prob >= DECISION_THRESHOLD else 'NORMAL'
        line = f"  customer #{i+1}:  {pred}  (p={prob:.1%})"
        if truth is not None:
            actual = 'THEFT' if truth[i] == 1 else 'NORMAL'
            hit = (truth[i] == 1) == (prob >= DECISION_THRESHOLD)
            correct += hit
            line += f"   [actual: {actual:<6} {'correct' if hit else 'MISSED'}]"
        print(line)

    if truth is not None:
        print(f"\n  {correct}/{len(probs)} correct on this small sample "
              f"(model's full test-set AUC = 0.823).")
    print("\n  NOTE: this classification uses real historical SGCC data, "
          "NOT this session's live readings.")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="ETD hardware/ML integration")
    ap.add_argument('--offline', action='store_true',
                    help="Skip the Arduino; run the ML demo only "
                         "(use to test without hardware connected).")
    ap.add_argument('--port', default=SERIAL_PORT, help="Serial port (e.g. COM10)")
    ap.add_argument('--list-ports', action='store_true',
                    help="List available COM ports and exit (find the Arduino).")
    args = ap.parse_args()

    if args.list_ports:
        import serial.tools.list_ports as lp
        ports = list(lp.comports())
        if not ports:
            print("No COM ports found. Is the Arduino plugged in?")
        for p in ports:
            print(f"  {p.device}  -  {p.description}")
        return

    print("Loading trained model...")
    with open(MODEL_PATH, 'rb') as f:
        model_bundle = pickle.load(f)
    print("Model loaded.\n")

    if args.offline:
        print("=" * 62)
        print("OFFLINE MODE — ML layer only, no hardware required")
        print("=" * 62)
        run_historical_demo(model_bundle)
        return

    import serial  # imported here so offline mode works without pyserial
    print(f"Opening serial connection on {args.port} at {BAUD_RATE} baud...")
    ser = serial.Serial(args.port, BAUD_RATE, timeout=2)
    time.sleep(2)
    print("Connected. Reading live hardware stream.\n")
    print("=" * 62)
    print("LAYER 1: Real-time bypass detection (live, working now)")
    print(f"LAYER 2: Accumulated-pattern ML classification (needs {WINDOW_DAYS}")
    print("         days of history — demo shown separately below)")
    print("=" * 62 + "\n")

    count = 0
    try:
        while True:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if not line:
                continue
            values = parse_line(line)
            if values is None:
                continue

            report_realtime_status(values)   # Layer 1 — instant, real
            log_reading(values)              # Layer 2a — accumulate
            count += 1

            if count % 5 == 0:
                print(f"    [ACCUMULATION] {days_logged_so_far()} of "
                      f"{WINDOW_DAYS} days logged.\n")

            if count == 20:
                run_historical_demo(model_bundle)

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        ser.close()


if __name__ == '__main__':
    main()
