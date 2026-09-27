<div align="center">

# FIRASA

*The sixth sense for credit risk.*

**An early-warning system that spots credit-card default months before it happens,
and tells the team what to do about it.**

Data Science Bootcamp capstone · SDA &amp; WeCloudData

</div>

---

![Firasa project summary](docs/summary.png)

## Overview

Banks usually learn that a customer is in trouble when a payment is already missed. Firasa
(Arabic for *insight / intuition*) looks for the signs earlier. It reads every customer's
monthly statements, estimates their probability of default, and sorts them into four risk
levels, each paired with one clear action for the credit team.

The project covers the full path from raw data to a working product:

1. **Exploratory analysis and leakage-free cleaning** of the American Express default dataset.
2. **Two modelling branches**: XGBoost on each customer's aggregated profile, and a GRU on
   their 13-month statement history.
3. **A fusion layer** that combines both branches into one calibrated probability.
4. **An interactive web dashboard** for analysts and managers.

## Key results

All results below are on a **held-out test set of 8,297 customers** that was never used for
training, tuning, or threshold selection, unless marked *validation*.

| Metric | Result |
|---|---|
| Fusion ROC-AUC, last statement | **0.9581** |
| Fusion ROC-AUC, 3 months before the last statement | **0.9433** |
| Fusion ROC-AUC, 6 months before the last statement | **0.9329** |
| Defaulters flagged High/Critical 6 months ahead *(validation)* | **85.7%** at ~14% false alarms |
| Actual default rate, Low → Critical *(validation)* | **1.2% → 15.3% → 36.7% → 83.1%** |

**Model ladder (ROC-AUC).** Every model had to beat the simpler one before it:

| Model | ROC-AUC | Evaluated on |
|---|---|---|
| Naive baseline (rank by `P_2` only) | 0.9154 | validation |
| Logistic Regression | 0.9513 | validation |
| LightGBM | 0.9558 | validation |
| LSTM (Branch 2 alternative) | 0.9527 | test |
| GRU (Branch 2) | 0.9529 | test |
| XGBoost (Branch 1) | 0.9579 | test |
| **Fusion: XGBoost + GRU** | **0.9581** | test |

The fusion's gain over XGBoost alone is statistically confirmed (bootstrap 95% CI excludes 0)
at 6 and 3 months before the last statement, which is exactly the early-warning window the
project targets. At the last statement the two are effectively tied.

## How it works

```
Amex monthly statements
        │
        ▼
EDA & cleaning ── customer-level sampling, train/test split by customer,
        │         train-only imputation and outlier capping
        ├──────────────────────────────┐
        ▼                              ▼
Branch 1 · XGBoost               Branch 2 · GRU
1,802 aggregated features        13-month sequences, one score per month
        └──────────────┬───────────────┘
                       ▼
        Fusion · logistic regression meta-model
        (calibrated probability, ~52% XGBoost / 48% GRU)
                       ▼
        4 risk levels + recommended action
        Low < 10% · Moderate < 30% · High < 60% · Critical ≥ 60%
                       ▼
                Web dashboard
```

**Explainability.** Branch 1 is explained with SHAP (`TreeExplainer`), Branch 2 with
gradient × input saliency, and the fusion layer through its two coefficients. The strongest
signals across every method are payment behaviour (`P_2`), balance (`B_1`, `B_8`), and
delinquency (`D_48`).

## The dashboard

A static web app in [`dashboard/`](dashboard/). It needs no backend and reads the exported
CSVs directly in the browser.

| Page | What it is for |
|---|---|
| **Home** (`index.html`) | Landing page with the live customer count |
| **Radar** (`radar.html`) | Every customer plotted by risk; click a point for a quick profile |
| **Customer Profile** (`profile.html`) | Searchable directory filtered by risk, with a full profile: score, 13-month trend, reasons from both branches, and the next step |
| **Manager Dashboard** (`manager.html`) | Portfolio KPIs, risk mix, action playbook, early warnings (close to Critical, rising fast), and a priority list with CSV export |

### Run it locally

Open `dashboard/index.html` directly, or serve the folder over HTTP:

```bash
cd dashboard
python -m http.server 8000
```

and open <http://localhost:8000>. Over HTTP the pages read `data/*.csv`; opened straight from
disk, the browser blocks that, so they read the same data from `data/offline-data.js`. After
changing the CSVs, rebuild that file with `python dashboard/build_offline_data.py`.

The dashboard can also be published as-is with **GitHub Pages**: set the Pages source to the
repository's `dashboard/` folder, or copy its contents to the published branch.

## Project structure

```
Firasa/
├── dashboard/                  # Web dashboard (static HTML/CSS/JS)
│   ├── index.html              #   Home
│   ├── radar.html              #   Risk radar
│   ├── profile.html            #   Customer directory and profile
│   ├── manager.html            #   Manager dashboard
│   ├── assets/                 #   Shared CSS/JS, card images, demo data
│   └── data/                   #   Exported scores and monthly trajectories (from notebook 06)
├── api/                        # Phase 6: FastAPI scoring service + drift monitor
│   ├── main.py                 #   Endpoints (/predict, /model-info, /monitoring/*, ...)
│   ├── inference.py             #   Loads models/, replays notebook 06's scoring pipeline
│   ├── drift.py                 #   Data drift monitoring (input z-test, output PSI)
│   ├── schemas.py / config.py   #   Request/response models, artifact paths
│   └── README.md                #   Full API + drift-monitoring docs
├── models/                     # Trained artifacts: XGBoost, GRU, scaler, encoder, fusion
├── notebooks/                  # Full modelling pipeline, run in order
├── logs/                       # Drift monitor's JSONL log (git-ignored, created at runtime)
├── Dockerfile / docker-compose.yml   # Containerizes the api/ service (Phase 6)
└── docs/                       # Project summary board (source of the image above)
```

## Notebooks

| # | Notebook | What it does |
|---|---|---|
| 01 | [EDA](notebooks/Firasa_01_EDA.ipynb) | Customer-level sampling, missing-value handling, leakage-free split, feature aggregation, feature dictionary |
| 02 | [Baseline: Logistic Regression](notebooks/Firasa_02_Baseline_LogisticRegression.ipynb) | Naive baselines and a Logistic Regression sanity check |
| 03 | [Branch 1: XGBoost & LightGBM](notebooks/Firasa_03_Branch1_XGBoost_LightGBM.ipynb) | Tree models, 5-fold CV, threshold tuning, held-out test, Branch 1 export |
| 04 | [Branch 2: GRU + Fusion](notebooks/Firasa_04_Branch2_GRU_Fusion.ipynb) | Sequence model, fusion layer, risk levels, calibration, bootstrap CIs, SHAP & saliency |
| 05 | [LSTM vs GRU](notebooks/Firasa_05_LSTM_vs_GRU_Decision.ipynb) | End-to-end LSTM alternative and the final comparison of every model |
| 06 | [Dashboard export](notebooks/Firasa_06_Dashboard_Export.ipynb) | Scores every test customer and exports the dashboard's data files |

### Reproducing the results

The notebooks were built for **Google Colab** with Google Drive for intermediate files.

1. Download the [Amex Parquet dataset](https://www.kaggle.com/datasets/odins0n/amex-parquet)
   from Kaggle (notebook 01 does this through the Kaggle API).
2. Run the notebooks in order, 01 → 06. Each one reads the files the previous one saved.
3. Copy the two CSVs exported by notebook 06 into `dashboard/data/`.

Main libraries: `pandas`, `polars`, `scikit-learn`, `xgboost`, `lightgbm`, `tensorflow`/`keras`,
`shap`, `matplotlib`, `seaborn`.

## Phase 6: model serving API, Docker & drift monitoring

The proposal's Phase 6 ("Containerize the model via Docker, expose it using a REST API
(FastAPI), and establish monitoring tools to track data drift over time") lives in
[`api/`](api/) — all three parts are implemented. It's a FastAPI service that loads the
trained XGBoost, GRU, and fusion artifacts from `models/` and replays the same scoring
pipeline `notebooks/Firasa_06_Dashboard_Export.ipynb` runs in bulk, per customer, on
demand, and logs every prediction to a rolling drift monitor.

```bash
cd Firasa
docker build -t firasa-api .
docker run --rm -p 8000:8000 firasa-api
# or, to persist drift history across restarts:
docker compose up --build
```

Verified end to end, including a real `docker build`/`docker run` on Windows with Docker
Desktop (not just this project's dev sandbox, whose network policy blocks container
registries) — see [`api/README.md`](api/README.md#verified-so-far).

`GET /monitoring/drift` reports Population Stability Index-style drift for the predicted
risk levels and a mean-shift test for the model's most important input features, both
against real reference numbers from the training artifacts and the held-out test set —
see [`api/README.md`](api/README.md#data-drift-monitoring) for the method and why input
and output drift are scored differently.

See [`api/README.md`](api/README.md) for the request format, a ready-made sample request,
and what's in scope versus a possible next step (live scoring from raw statement rows,
Branch-1 input drift). Cloud deployment (Azure) is planned as the next step after this.

## Limitations

- **Customer-level split, not out-of-time.** Train, validation, and test contain different
  customers but cover the same calendar period (March 2017 to March 2018). The results show
  generalisation to new customers, not yet to future months. A time-based split is the natural
  next step.
- **Sample, not the full dataset.** Models were trained on about 41,000 of the ~459,000
  customers to fit Colab's memory limits.
- **Display names are pseudonyms**, generated deterministically from each customer ID for the
  demo. They are not real people.

## Team

### Noura Mesfer Alzahrani 
### Layan Fahad Alhumaidi 
### Abeer Mohammed Alshahrani 


## Acknowledgements

Built as the capstone project of the Data Science Bootcamp by **SDA** and **WeCloudData**.
Data: the American Express Default Prediction competition, via the
[Amex Parquet](https://www.kaggle.com/datasets/odins0n/amex-parquet) dataset on Kaggle.
