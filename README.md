# A Machine Learning Approach to Electricity Theft Detection in Smart Energy Meter Infrastructure

Final-year project, Department of Electrical and Electronics Engineering, University of Lagos.

**Author:** OLATUNJI Oladipupo A. (190408030)
**Supervisor:** Dr. O. A. Alamu

The system has two layers. A hardware prototype detects physical bypass theft in
real time by energy balance (GAP = P_A − P_B). A two-stage stacked ensemble
(XGBoost, Random Forest and MLP base classifiers with a Logistic Regression
meta-learner) detects behavioural theft from daily consumption history.

## Repository contents

| Path | Purpose |
|---|---|
| `etd_pipeline.py` | Full ML pipeline: imputation, 173-feature engineering, SMOTETomek vs class-weighted comparison, stacking, threshold analysis, SHAP |
| `cnn_experiment.py` | 1D-CNN baseline used for comparison |
| `etd_hardware_integration.py` | Serial listener combining the live energy-balance check with the saved model |
| `etd_arduino/etd_arduino.ino` | Arduino Mega 2560 firmware (3 × PZEM-004T v4.0, I²C LCD, buzzer, LED) |
| `export_demo_sample.py`, `make_subset.py` | Dataset helpers |
| `make_figures_*.py`, `figures/` | Figure generators and the figures used in the report |
| `etd_model.pkl`, `etd_cnn.pt` | Trained ensemble and CNN weights |
| `report_metrics.json`, `report_artifacts.npz` | Results of the full-dataset run |
| `demo_historical_sample.csv`, `live_readings_log.csv` | Demo input and logged prototype readings |

## Results (full SGCC dataset, held-out test set of 8,475 customers)

| Metric | SMOTETomek | Class-weighted |
|---|---|---|
| Ensemble AUC | 0.7727 | 0.8229 |
| False positive rate | 0.0275 | 0.0058 |
| Recall | 0.2766 | 0.1909 |
| F1 (theft class) | 0.3521 | 0.3046 |

Cross-validation on SMOTETomek-resampled data overstated base-classifier AUC by
0.15 to 0.24 relative to the untouched test set (XGBoost 0.9710 → 0.7823,
Random Forest 0.9398 → 0.7874, MLP 0.9820 → 0.7425). Under class weighting the
out-of-fold and test figures agree.

Hardware: baseline gap −0.10 to +0.20 W, gap under simulated bypass
10.6–11.2 W, alarm threshold 10.0 W.

## Running

```bash
pip install -r requirements.txt
python etd_pipeline.py
python etd_hardware_integration.py --offline
python etd_hardware_integration.py --port COM10
```

The dataset is not included because of its size. The pipeline uses the public
SGCC (State Grid Corporation of China) electricity theft dataset: 42,372
customers, 1,034 daily readings each, with a `FLAG` label column. Download it
and set `DATASET_PATH` in `etd_pipeline.py` (and the equivalent path constants
in the other scripts) to its location.

The firmware needs the `PZEM004Tv40_R4` and `LiquidCrystal_I2C` Arduino libraries.
