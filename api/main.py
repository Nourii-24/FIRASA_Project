"""Firasa scoring API -- Phase 6 of the project proposal:
'Containerize the model via Docker, expose it using a REST API (FastAPI)'.

Run locally (no Docker):
    cd Firasa
    pip install -r requirements.txt
    uvicorn api.main:app --host 0.0.0.0 --port 8000

Then open http://localhost:8000/docs for interactive Swagger UI, or:
    curl http://localhost:8000/health
    curl -X POST http://localhost:8000/sample-request > sample.json
    curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d @sample.json
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from .inference import ModelService
from .schemas import HealthResponse, ModelInfoResponse, PredictRequest, PredictResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("firasa.api")

_service_holder: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading Firasa model artifacts...")
    _service_holder["service"] = ModelService()
    logger.info("Firasa model artifacts loaded. Ready to serve.")
    yield
    _service_holder.clear()


app = FastAPI(
    title="Firasa Credit Risk Early-Warning API",
    description=(
        "Serves the trained Firasa models (XGBoost Branch 1 + GRU Branch 2 "
        "+ logistic fusion) that were exported by the project's notebooks. "
        "This is the Phase 6 deliverable: the same scoring pipeline "
        "notebook 06 runs in bulk, exposed here per-customer over HTTP."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


def get_service() -> ModelService:
    service = _service_holder.get("service")
    if service is None:
        raise HTTPException(status_code=503, detail="Model service not ready yet.")
    return service


@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health():
    service = _service_holder.get("service")
    ok = service is not None
    return HealthResponse(
        status="ok" if ok else "starting",
        branch1_loaded=ok,
        branch2_loaded=ok,
        fusion_loaded=ok,
    )


@app.get("/model-info", response_model=ModelInfoResponse, tags=["ops"])
def model_info():
    service = get_service()
    return service.model_info()


@app.post("/sample-request", tags=["ops"])
def sample_request():
    """Returns a structurally valid, synthetic request body you can POST
    straight to /predict to confirm the service works end to end. It does
    not represent a real customer.
    """
    service = get_service()
    return service.sample_request()


@app.post("/predict", response_model=PredictResponse, tags=["scoring"])
def predict(request: PredictRequest):
    service = get_service()
    try:
        result = service.predict(
            xgb_features=request.xgb_features,
            monthly_statements=[m.model_dump() for m in request.monthly_statements],
        )
    except Exception as exc:  # noqa: BLE001 - surfaced to the caller as a 400
        logger.exception("Prediction failed")
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return PredictResponse(customer_id=request.customer_id, **result)


@app.get("/monitoring/drift", tags=["monitoring"])
def monitoring_drift():
    """Phase 6's third component: data drift monitoring.

    input_drift: a one-sample z-test of the live window's mean against
    each curated Branch-2 feature's real training-time reference mean/std
    (models/branch2_scaler.joblib). output_drift: standard PSI of the live
    risk_level distribution against the real held-out test-set reference.
    A feature or the output distribution needs a handful of /predict calls
    in the rolling window before it reports at all, and 30+ before the
    result is flagged reliable -- see the response's own
    'input_drift_method' / 'output_drift_method' for the exact thresholds
    and why two different methods are used.
    """
    service = get_service()
    return service.drift_report()


@app.get("/monitoring/summary", tags=["monitoring"])
def monitoring_summary():
    """Cheap counters (no PSI computation) for a quick liveness check on
    the drift monitor itself: how many predictions have been logged and
    how many currently sit in the rolling window.
    """
    service = get_service()
    return service.drift_summary()
