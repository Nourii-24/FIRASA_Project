"""
Central place for every path the app needs. If you move a file, change it
here once instead of hunting through app.py and src/.
"""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

XGB_MODEL_PATH = MODELS_DIR / "branch1_xgboost.json"
XGB_CONFIG_PATH = MODELS_DIR / "branch1_config.json"

GRU_MODEL_PATH = MODELS_DIR / "branch2_gru.keras"
SCALER_PATH = MODELS_DIR / "branch2_scaler.joblib"
ENCODER_PATH = MODELS_DIR / "branch2_ordinal_encoder.joblib"
META_MODEL_PATH = MODELS_DIR / "fusion_meta_model.joblib"
FUSION_CONFIG_PATH = MODELS_DIR / "branch2_fusion_config.json"

# Main snapshot table: one row per customer (name, risk level, action, SHAP/GRU reasons).
CUSTOMER_SCORES_PATH = DATA_DIR / "customer_scores_with_shap.csv"

# Long-format table: many rows per customer (month_index, risk_score), for the
# profile page's pulse chart.
CUSTOMER_TRAJECTORIES_PATH = DATA_DIR / "customer_trajectories.csv"

# Risk level colors, kept identical to the notebooks so the dashboard
# matches every chart already shown in the project.
LEVEL_COLORS = {
    "Low": "steelblue",
    "Moderate": "#8FBF9F",
    "High": "#DAA520",
    "Critical": "#B22222",
}
