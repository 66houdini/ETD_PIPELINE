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
| `multiseed_experiment.py` | Five-seed, three-condition experiment with a train/validation/test split and validation-set threshold selection |
| `cnn_experiment.py` | 1D-CNN baseline used for comparison |
| `etd_hardware_integration.py` | Serial listener combining the live energy-balance check with the saved model |
| `etd_arduino/etd_arduino.ino` | Arduino Mega 2560 firmware (3 × PZEM-004T v4.0, I²C LCD, buzzer, LED) |
| `export_demo_sample.py`, `make_subset.py` | Dataset helpers |
| `make_figures_*.py`, `figures/` | Figure generators and the figures used in the report |
| `etd_model.pkl`, `etd_cnn.pt` | Trained ensemble and CNN weights |
| `report_metrics.json`, `report_artifacts.npz` | Results of the full-dataset run |
| `multiseed_results.json`, `multiseed_summary.json` | Per-seed results and aggregates of the five-seed run |
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

## Multi-seed results (five seeds, 70/10/20 split, mean ± standard deviation)

Three ways of handling class imbalance were compared on identical partitions:
SMOTETomek applied before cross-validation, SMOTETomek applied inside each
training fold, and class weighting.

| Quantity | SMOTETomek before CV | SMOTETomek within folds | Class weighting |
|---|---|---|---|
| Ensemble test AUC | 0.7657 ± 0.0115 | 0.7957 ± 0.0066 | 0.8170 ± 0.0056 |
| XGBoost OOF − test AUC | +0.1972 ± 0.0073 | −0.0115 ± 0.0093 | −0.0126 ± 0.0092 |
| Random Forest OOF − test AUC | +0.1630 ± 0.0069 | −0.0055 ± 0.0067 | −0.0054 ± 0.0052 |
| MLP OOF − test AUC | +0.2543 ± 0.0109 | −0.0353 ± 0.0193 | −0.0396 ± 0.0164 |
| Test FPR at validation-selected threshold | 0.0097 ± 0.0023 | 0.0104 ± 0.0026 | 0.0100 ± 0.0014 |
| Test recall at that threshold | 0.1668 ± 0.0106 | 0.1729 ± 0.0151 | 0.2169 ± 0.0222 |

Resampling before cross-validation overstates base-classifier AUC by 0.16 to
0.25. Resampling within folds removes the overstatement and leaves test AUC
unchanged. Class weighting gives the best ensemble on every seed.

## Running

```bash
pip install -r requirements.txt
python etd_pipeline.py
python multiseed_experiment.py
python etd_hardware_integration.py --offline
python etd_hardware_integration.py --port COM10
```

The dataset is not included because of its size. The pipeline uses the public
SGCC (State Grid Corporation of China) electricity theft dataset: 42,372
customers, 1,034 daily readings each, with a `FLAG` label column. Download it
and set `DATASET_PATH` in `etd_pipeline.py` (and the equivalent path constants
in the other scripts) to its location.

The firmware needs the `PZEM004Tv40_R4` and `LiquidCrystal_I2C` Arduino libraries.
