# Firasa scoring API (Phase 6)

A FastAPI service that loads the three trained artifacts in `../models/`
(XGBoost Branch 1, GRU Branch 2, the logistic-regression fusion layer) and
replays, per customer and on demand, the exact same scoring pipeline that
`notebooks/Firasa_06_Dashboard_Export.ipynb` runs in bulk over the test set:
Branch 1 + Branch 2 -> logit fusion -> 4 risk levels + action.

Nothing is retrained here. This is the "expose it using a REST API
(FastAPI)" part of the proposal's Phase 6; `Dockerfile` at the project root
is the "containerize the model via Docker" part; `api/drift.py` (see
"Data drift monitoring" below) is the "establish monitoring tools to track
data drift over time" part -- all three pieces of Phase 6 are implemented.

## What the API needs as input, and why

- **`xgb_features`** -- the 1,802 Branch-1 features, already aggregated
  across the customer's full statement history (mean/std/min/max/last per
  raw column, missing-value flags, log-transforms, the two categorical
  columns). That cross-month aggregation is notebook 01's job and isn't
  reimplemented here -- in a real deployment it would come from an upstream
  feature pipeline / feature store, same as any production ML system splits
  "compute features" from "serve model". Missing keys default to `0.0`.
- **`monthly_statements`** -- up to 13 months of the *row-level* engineered
  features Branch 2 uses (the same per-statement values, not aggregated).
  The API applies the real `StandardScaler` + `OrdinalEncoder` and pads/
  truncates exactly as training did (pre-padding with zeros, keeping the
  most recent 13 months). Fewer than 13 months is fine; the categorical
  `D_63`/`D_64` can be given as the raw letter code or the pre-encoded
  number.

`GET /model-info` returns the exact feature name lists, risk bins, actions
and fusion coefficients, so a caller can see precisely what's expected
without reading this file.

## Run it locally (no Docker)

```bash
cd Firasa
pip install -r requirements.txt
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Open <http://localhost:8000/docs> for interactive Swagger UI.

## Run it with Docker

```bash
cd Firasa
docker build -t firasa-api .
docker run --rm -p 8000:8000 firasa-api
# or: docker compose up --build
```

## Smoke-test it

The 1,802-key request body is impractical to type by hand, so the API can
generate a structurally valid (synthetic, not a real customer) example for
you:

```bash
curl -X POST http://localhost:8000/sample-request -o sample.json
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d @sample.json
```

Expected shape of the response:

```json
{
  "customer_id": "SAMPLE-SYNTHETIC",
  "p_xgb": 0.297,
  "p_gru": 0.387,
  "p_fused": 0.186,
  "risk_level": "Moderate",
  "action": "Add to watchlist; send payment reminder / financial-wellness message",
  "xgb_contribution_pct": 0.626,
  "gru_contribution_pct": 0.374,
  "warnings": []
}
```

`GET /health` reports whether all three model artifacts are loaded (useful
as the container's health check and later as an Azure liveness probe).

## Data drift monitoring

`GET /monitoring/drift` and `GET /monitoring/summary` are the "establish
monitoring tools to track data drift over time" part of Phase 6, implemented
in `api/drift.py`. Every `/predict` call is logged (in memory, in a rolling
window of the most recent 500 predictions, and appended as one line to
`logs/predictions.jsonl` so history survives a container restart -- mount
`logs/` as a volume, which `docker-compose.yml` already does).

Two different things are tracked, deliberately with two different methods:

- **Input drift** -- a curated set of 10 Branch-2 features (`P_2`, `B_1`,
  `B_2`, `B_3`, `B_8`, `D_39`, `D_41`, `D_48`, `S_3`, `R_1`: the strongest
  signals called out in the final report's explainability section, plus a
  couple more so every one of the five interpretable feature groups --
  payment, balance, delinquency, spend, risk -- is covered). Each is
  compared to its real training-time reference mean/std, taken directly
  from the fitted `models/branch2_scaler.joblib` (not re-estimated, not
  assumed), using a one-sample z-test on the live window's mean:
  `mean_shift_z`, `|z| < 2` stable / `< 4` moderate_shift / else
  significant_shift. This is deliberately **not** a textbook PSI: PSI needs
  the reference's actual shape, which the serving side doesn't have (only
  its mean/std survive in the scaler), and an earlier version of this
  module that assumed the reference was Gaussian flagged ordinary,
  in-distribution traffic as "drifting" purely because several of these
  features are skewed, not Gaussian -- a false alarm every time. The
  z-test avoids that by only ever testing what it can actually support: has
  the mean moved.
- **Output drift** -- the live `risk_level` distribution against the real
  reference distribution measured on the 8,297-customer held-out test set
  (Low 56.7% / Moderate 8.9% / High 10.0% / Critical 24.4%). This one *is*
  standard PSI, because risk level is a real 4-category variable with a
  known real reference proportion per category -- no shape assumption
  needed here.

Both need `>= 5` requests in the window before reporting anything, and
flag `"reliable": false` under 30. Not covered: drift on Branch 1's 1,802
aggregated features -- no reference mean/std for that feature set survives
into any deployed artifact (Branch 1 needs no scaler), and recomputing one
needs the original training data, which this service doesn't have.

## Verified so far

This pipeline (model loading, scaling, encoding, padding, GRU inference,
logit fusion, risk binning) was run end-to-end against the real artifacts
and matches a from-scratch reimplementation of notebook 06's math exactly.

`docker build` / `docker run` have also been verified for real: run on an
actual Windows machine with Docker Desktop (this project's own sandbox
network policy blocks all container registries, so the build itself can
only be confirmed outside it). The build completed cleanly, and the
running container's `/predict` returned numbers identical to the
dev-sandbox verification above (`p_xgb=0.2967461943626404,
p_gru=0.3871452808380127, p_fused=0.1863216448522759, risk_level=Moderate`
for the synthetic `/sample-request` payload) -- same model, same answer,
in a real container.

The drift monitor (`api/drift.py`) was verified three ways: (1) feeding it
40 deliberately out-of-distribution requests (the all-zero synthetic
`/sample-request` payload) and confirming every monitored feature and the
output distribution correctly report `significant_shift`; (2) feeding it
60 requests sampled from each feature's own real reference mean/std and
confirming every one correctly reports `stable`; (3) killing and
restarting the FastAPI process against the same `logs/predictions.jsonl`
and confirming `predictions_replayed_from_log` picks the history back up
and the drift status is preserved -- i.e. "track drift over time" actually
survives a restart, not just within one process's memory.

## Not in scope here (possible next step)

- Live scoring straight from raw AmEx statement rows (no pre-aggregation)
  would mean porting notebook 01's full feature-engineering pipeline
  (customer-level aggregation, the specific log-transform and missing-flag
  column lists) into this service. That's a bigger, separate piece of
  work -- ask if you want it built next.
- Branch-1 (XGBoost) input drift is not monitored (see "Data drift
  monitoring" above) -- would need the original training data to compute
  reference stats for the 1,802 aggregated features, which isn't available
  to this service.
- The drift monitor's rolling window (500 predictions) and JSONL log are
  local to one running container; nothing aggregates drift across replicas
  or ships alerts anywhere (email/Slack/PagerDuty). Fine for one instance
  and for the proposal's academic scope; a real deployment would want a
  shared store (e.g. a small database) and an alerting hook instead.


## Deployed on Azure

The API is live: **https://firasa-api.azurewebsites.net/docs**

### Architecture
- **Azure Container Registry** (`firasaacr2026`) stores the Docker image built from this
  repo's root `Dockerfile`.
- **Azure App Service** (Linux, container-based, plan `firasa-plan`) pulls that image and
  runs it as `firasa-api`, exposing it publicly over HTTPS. `WEBSITES_PORT=8000` is set
  since the container listens on 8000, not the platform default.

### Redeploying after a code/model change
```bash
az acr login --name firasaacr2026
docker build -t firasaacr2026.azurecr.io/firasa-api:v2 .
docker push firasaacr2026.azurecr.io/firasa-api:v2
az webapp config container set --name firasa-api --resource-group firasa-rg \
  --docker-custom-image-name firasaacr2026.azurecr.io/firasa-api:v2
az webapp restart --name firasa-api --resource-group firasa-rg
```
Registry credentials are never committed here -- fetch them fresh with:
```bash
az acr credential show --name firasaacr2026
```

### Note for teammates on Azure for Students
This subscription tier restricts which regions you can deploy to. If `az acr create` /
`az appservice plan create` fails with `RequestDisallowedByAzure`, find your allowed
regions with:
```bash
az policy assignment list -o table
az policy assignment show --name sys.regionrestriction --query "parameters" -o json
```