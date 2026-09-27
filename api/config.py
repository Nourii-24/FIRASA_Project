"""Paths and lightweight config loading for the Firasa inference API.

Everything here just points at the artifacts already exported by
notebooks 03 (Branch 1) and 04 (Branch 2 + fusion) into ../models.
Nothing is retrained here -- same spirit as notebook 06.
"""
import json
import os
from pathlib import Path

# Allow overriding via env var so the same image can be pointed at a
# different artifacts folder without a rebuild, if ever needed.
MODELS_DIR = Path(os.environ.get("FIRASA_MODELS_DIR", Path(__file__).resolve().parent.parent / "models"))

BRANCH1_MODEL_PATH = MODELS_DIR / "branch1_xgboost.json"
BRANCH1_CONFIG_PATH = MODELS_DIR / "branch1_config.json"
BRANCH2_MODEL_PATH = MODELS_DIR / "branch2_gru.keras"
BRANCH2_SCALER_PATH = MODELS_DIR / "branch2_scaler.joblib"
BRANCH2_ENCODER_PATH = MODELS_DIR / "branch2_ordinal_encoder.joblib"
FUSION_CONFIG_PATH = MODELS_DIR / "branch2_fusion_config.json"
FUSION_MODEL_PATH = MODELS_DIR / "fusion_meta_model.joblib"

# Where the drift monitor appends one JSON line per prediction. Mount this
# as a volume (see docker-compose.yml) so drift history survives a
# container restart -- otherwise it's still fine, just in-memory-only for
# that container's lifetime.
LOGS_DIR = Path(os.environ.get("FIRASA_LOGS_DIR", Path(__file__).resolve().parent.parent / "logs"))
DRIFT_LOG_PATH = LOGS_DIR / "predictions.jsonl"


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
