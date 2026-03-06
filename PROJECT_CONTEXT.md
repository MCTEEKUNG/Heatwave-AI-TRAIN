# HEATWAVE-AI-Prediction — PROJECT CONTEXT

**Version**: 1.0.0  
**Date**: 2026-03-05  
**Status**: ✅ Initial Implementation Complete  
**Author**: HEATWAVE-AI Team

---

## 1. Project Overview

HEATWAVE-AI-Prediction is a **modular AI experimentation platform** for binary heatwave classification. It trains, benchmarks, and serves five machine learning models using ERA5 reanalysis climate data covering 2000–2015.

### Goals
- Train 5 ML models on ERA5 climate features
- Automatically benchmark and rank models by F1 Score on a leaderboard
- Persist trained models for future real-time inference
- Visualize results in a premium dark-mode web dashboard
- Launch the entire system with a single click via [`Start.bat`](Start.bat)

---

## 2. System Architecture

```
HEATWAVE-AI-Prediction/
│
├── Era5-data-2000-2026/         ← Raw ERA5 NetCDF files (surface + upper, 2000–2015)
│
├── config/
│   └── config.yaml              ← Central configuration (paths, thresholds, hyperparams)
│
├── utils/
│   ├── data_loader.py           ← ERA5 NetCDF → pandas DataFrame
│   ├── preprocessing.py         ← Feature engineering, heatwave labels, StandardScaler, splits
│   └── gpu_utils.py             ← CUDA availability helper
│
├── models/
│   ├── base_model.py            ← Abstract BaseModel interface
│   ├── balanced_random_forest.py
│   ├── xgboost_model.py
│   ├── lightgbm_model.py
│   ├── mlp_model.py             ← PyTorch 3-layer MLP with early stopping
│   └── kan_model.py             ← KAN with learnable B-spline activations (PyTorch)
│
├── training/
│   ├── trainer.py               ← Train → evaluate → save cycle
│   ├── cross_validation.py      ← StratifiedKFold CV helper
│   └── hyperparameter_tuning.py ← GridSearch / RandomSearch wrappers
│
├── pipelines/
│   └── training_pipeline.py     ← Orchestrates all 5 models end-to-end
│
├── evaluation/
│   ├── metrics.py               ← compute_metrics() → accuracy, precision, recall, F1, ROC-AUC
│   └── benchmark.py             ← Leaderboard builder from result JSONs
│
├── prediction/
│   ├── predictor.py             ← Loads model + scaler, runs inference
│   └── predict.py               ← CLI inference script
│
├── dashboard/
│   ├── app.py                   ← Flask app factory
│   ├── routes.py                ← API routes (/api/leaderboard, /api/best, /api/results)
│   └── templates/index.html     ← Premium dark-mode dashboard (Chart.js)
│
├── experiments/
│   ├── results/                 ← Per-model JSON results + leaderboard.json
│   └── models/                  ← Saved .pkl model files + scaler.pkl
│
├── data/
│   ├── raw/                     ← (reserved for future use)
│   └── processed/               ← (preprocessed CSVs if exported)
│
├── docs/
│   └── ProjectContext.md        ← Internal development notes
│
├── main.py                      ← Unified CLI entry point (train | dashboard | predict)
├── Start.bat                    ← One-click launcher (venv activation + menu)
└── requirements.txt             ← Python dependencies
```

---

## 3. Data Pipeline

| Step | Detail |
|---|---|
| **Source** | ERA5 surface NetCDF files, years 2000–2015 (`era5_surface_YYYY.nc`) |
| **Raw Variables** | `t2m` (2m temperature, K), `d2m` (dewpoint, K), `sp` (surface pressure, Pa), `u10`, `v10` (wind, m/s) |
| **Engineered Features** | `t2m_c` (°C), `d2m_c` (°C), `heat_index`, `wind_speed`, `sp` |
| **Heatwave Label** | `heatwave = 1` if `t2m ≥ 35 °C`, else `0` (highly imbalanced) |
| **Data Split** | 70 % train / 15 % validation / 15 % test — stratified, `random_state=42` |

---

## 4. Models

| # | Model | Library | Key Characteristics |
|---|---|---|---|
| 1 | **Balanced Random Forest** | `imbalanced-learn` | Handles class imbalance natively via balanced bootstrap sampling |
| 2 | **XGBoost** | `xgboost` | Gradient boosting; eval set with early stopping on `logloss` |
| 3 | **LightGBM** | `lightgbm` | Fast leaf-wise gradient boosting; early stopping |
| 4 | **MLP Neural Network** | PyTorch | 3-layer MLP (256→128→64), ReLU activations, Dropout 0.3, Adam optimizer, early stopping (patience=10) |
| 5 | **KAN** | PyTorch (custom) | Kolmogorov–Arnold Network; learnable B-spline activations per edge; grid_size=5, spline_order=3 |

All models implement the [`BaseModel`](models/base_model.py) abstract interface: `train()`, `predict()`, `predict_proba()`, `save()`, `load()`.

---

## 5. Training Pipeline

**Entry**: [`pipelines/training_pipeline.py`](pipelines/training_pipeline.py) → `TrainingPipeline.run()`

Steps:
1. Load ERA5 NetCDF data via [`ERA5DataLoader`](utils/data_loader.py)
2. Preprocess and label via [`HeatwavePreprocessor`](utils/preprocessing.py)
3. Stratified train / val / test split
4. For each of 5 models:
   - Instantiate model with config hyperparameters
   - Train via [`Trainer`](training/trainer.py)
   - Evaluate on test set → compute metrics
   - Save result JSON to [`experiments/results/`](experiments/results/)
   - Save model `.pkl` to [`experiments/models/`](experiments/models/)
5. Rebuild [`leaderboard.json`](experiments/results/leaderboard.json) ranked by F1 Score

**Start training**:
```bash
python main.py --mode train
# or via Start.bat → option 1
```

---

## 6. Prediction System

**CLI usage**:
```bash
python main.py --mode predict --model xgboost --input data/processed/sample.csv
python main.py --mode predict --model lightgbm --input data.csv --output results.csv --proba
python prediction/predict.py --model kan --input data.csv
```

**Programmatic usage**:
```python
from prediction.predictor import HeatwavePredictor
predictor = HeatwavePredictor("xgboost")
predictions = predictor.predict(df)
```

---

## 7. Dashboard

**URL**: `http://localhost:5000`  
**Stack**: Flask + Chart.js (dark-mode premium UI)

| Route | Description |
|---|---|
| `GET /` | Main dashboard HTML |
| `GET /api/leaderboard` | JSON leaderboard ranked by F1 |
| `GET /api/best` | Best model metadata |
| `GET /api/results` | All model result JSONs |

**Features**:
- Best Model hero card (accuracy, precision, recall, F1, ROC-AUC)
- Full leaderboard table sortable by metric
- 5 comparative charts: Accuracy, F1, Precision, Recall, ROC-AUC
- Training timeline view
- Auto-refresh every 60 seconds

**Start dashboard**:
```bash
python main.py --mode dashboard
# or via Start.bat → option 2
```

---

## 8. Configuration Reference

[`config/config.yaml`](config/config.yaml) is the single source of truth for all settings:

```
config.data.years                   List of ERA5 years to load (default: 2000–2015)
config.data.heatwave_threshold_celsius  Label threshold (default: 35.0)
config.split.train / val / test     Dataset proportions (70/15/15)
config.training.use_gpu             Enable CUDA for PyTorch models (default: true)
config.models.*                     Per-model hyperparameters
config.experiments.*                Output directories for models, results, logs
config.dashboard.port               Flask port (default: 5000)
```

---

## 9. GPU Support

[`utils/gpu_utils.py`](utils/gpu_utils.py) provides a CUDA detection helper. When `config.training.use_gpu: true` and a compatible GPU is available, PyTorch models (MLP, KAN) train on CUDA automatically. Sklearn-based models (BRF, XGBoost, LightGBM) use CPU multi-threading (`n_jobs: -1`).

---

## 10. Experiment Artifacts

All outputs are written to [`experiments/`](experiments/):

| File | Description |
|---|---|
| `experiments/models/<name>_model.pkl` | Serialized trained model |
| `experiments/models/scaler.pkl` | Fitted `StandardScaler` |
| `experiments/models/feature_names.pkl` | Ordered feature name list |
| `experiments/results/<name>_result.json` | Per-model metrics dict |
| `experiments/results/leaderboard.json` | All models ranked by F1 Score |

---

## 11. Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train all 5 models
python main.py --mode train

# 3. Launch dashboard
python main.py --mode dashboard

# 4. Run prediction
python main.py --mode predict --model xgboost --input data.csv

# OR — one-click Windows launcher
Start.bat
```

---

## 12. Current Implementation Status

| Component | Status |
|---|---|
| Directory structure | ✅ Complete |
| `config.yaml` | ✅ Complete |
| ERA5 data loader | ✅ Complete |
| Preprocessing pipeline | ✅ Complete |
| All 5 models | ✅ Complete |
| Metrics computation | ✅ Complete |
| Leaderboard benchmark | ✅ Complete |
| Trainer | ✅ Complete |
| Cross-validation helper | ✅ Complete |
| Hyperparameter tuning helper | ✅ Complete |
| Training pipeline | ✅ Complete |
| Prediction system (CLI + class) | ✅ Complete |
| Flask dashboard | ✅ Complete |
| `main.py` entry point | ✅ Complete |
| `Start.bat` launcher | ✅ Complete |
| GPU support scaffolding | ✅ Complete |

---

## 13. Roadmap

| Priority | Task |
|---|---|
| High | Full GPU training validation on CUDA hardware |
| High | Hyperparameter tuning runs per model, then retrain |
| Medium | Cross-validation metrics added to leaderboard |
| Medium | ERA5 data extension: 2016–2025 years |
| Medium | SHAP explainability plots in dashboard |
| Low | `/api/predict` REST endpoint for web-based inference |
| Low | Mobile-responsive dashboard layout |
| Low | Email notifications on training completion |

---

## 14. Dependencies

See [`requirements.txt`](requirements.txt) for the full locked dependency list. Key libraries:

| Library | Purpose |
|---|---|
| `xarray`, `netCDF4` | ERA5 NetCDF data ingestion |
| `pandas`, `numpy` | Data manipulation |
| `scikit-learn` | Preprocessing, metrics, CV, tree models |
| `imbalanced-learn` | Balanced Random Forest |
| `xgboost` | XGBoost model |
| `lightgbm` | LightGBM model |
| `torch` | PyTorch — MLP & KAN |
| `flask` | Web dashboard server |
| `pyyaml` | Config file parsing |
