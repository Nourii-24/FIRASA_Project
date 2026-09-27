"""Loads the Firasa artifacts once and runs the exact scoring pipeline
notebook 06 (Dashboard Export) uses: Branch 1 (XGBoost) + Branch 2 (GRU)
-> logit fusion (logistic regression) -> 4 risk levels + action.

Nothing is retrained or refit here. This module only replays, at request
time and for one customer, the same steps notebook 06 replays in bulk for
the whole test set.
"""
import logging
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from tensorflow import keras

from . import config, drift

logger = logging.getLogger("firasa.inference")

EPS = 1e-6


def _logit(p: float) -> float:
    p = min(max(p, EPS), 1 - EPS)
    return float(np.log(p / (1 - p)))


class ModelService:
    """Singleton-ish holder for the loaded models/configs. Instantiate once
    at app startup (see main.py's lifespan) and reuse across requests --
    loading the GRU and XGBoost models is the expensive part.
    """

    def __init__(self) -> None:
        self.b1_config = config.load_json(config.BRANCH1_CONFIG_PATH)
        self.fusion_config = config.load_json(config.FUSION_CONFIG_PATH)

        self.b1_features: List[str] = self.b1_config["features"]
        self.b1_cat_categories: Dict[str, List[str]] = self.b1_config.get("categorical_categories", {})

        self.seq_cols: List[str] = self.fusion_config["seq_cols"]
        self.cat_cols: List[str] = self.fusion_config["cat_cols"]
        self.non_cat_cols: List[str] = [c for c in self.seq_cols if c not in self.cat_cols]
        self.max_len: int = self.fusion_config["max_len"]
        self.risk_bins: List[float] = self.fusion_config["risk_bins"]
        self.risk_levels: List[str] = self.fusion_config["risk_levels"]
        self.risk_actions: Dict[str, str] = self.fusion_config["risk_actions"]
        self.fusion_coefficients: Dict[str, float] = self.fusion_config["fusion_coefficients"]

        # Sanity check baked in at load time rather than discovered at
        # request time: the encoder's fitted column order must line up
        # with where D_63/D_64 sit in seq_cols (they're expected last).
        if self.non_cat_cols + self.cat_cols != self.seq_cols:
            raise RuntimeError(
                "branch2_fusion_config.json's seq_cols does not have the "
                "categorical columns last -- the sequence-building code "
                "below assumes non_cat_cols + cat_cols == seq_cols."
            )

        logger.info("Loading Branch 1 (XGBoost) from %s", config.BRANCH1_MODEL_PATH)
        self.xgb_model = xgb.XGBClassifier()
        self.xgb_model.load_model(str(config.BRANCH1_MODEL_PATH))

        logger.info("Loading Branch 2 (GRU) from %s", config.BRANCH2_MODEL_PATH)
        self.gru_model = keras.models.load_model(str(config.BRANCH2_MODEL_PATH))

        logger.info("Loading scaler / ordinal encoder / fusion meta-model")
        self.scaler = joblib.load(config.BRANCH2_SCALER_PATH)
        self.ord_enc = joblib.load(config.BRANCH2_ENCODER_PATH)
        self.meta = joblib.load(config.FUSION_MODEL_PATH)

        # Phase 6, third component: data drift monitoring. Reference
        # mean/std for the monitored features comes straight from this
        # already-fitted scaler -- the real training-time statistics.
        logger.info("Initializing drift monitor (log: %s)", config.DRIFT_LOG_PATH)
        self.drift_monitor = drift.DriftMonitor(
            feature_names=list(self.scaler.feature_names_in_),
            means=self.scaler.mean_,
            stds=self.scaler.scale_,
            log_path=config.DRIFT_LOG_PATH,
        )

    # ---------------------------------------------------------------- Branch 1
    def _build_b1_row(self, xgb_features: Dict[str, float]) -> Tuple[pd.DataFrame, List[str]]:
        warnings: List[str] = []
        unknown = [k for k in xgb_features if k not in self.b1_features]
        if unknown:
            warnings.append(
                f"{len(unknown)} unknown xgb_features key(s) ignored (not in branch1_config features)."
            )

        row = {}
        for feat in self.b1_features:
            val = xgb_features.get(feat)
            if val is None:
                row[feat] = 0.0
                continue
            if feat in self.b1_cat_categories and isinstance(val, str):
                categories = self.b1_cat_categories[feat]
                row[feat] = float(categories.index(val)) if val in categories else -1.0
            else:
                row[feat] = float(val)

        df = pd.DataFrame([row], columns=self.b1_features)
        return df, warnings

    def score_branch1(self, xgb_features: Dict[str, float]) -> Tuple[float, List[str]]:
        df, warnings = self._build_b1_row(xgb_features)
        p_xgb = float(self.xgb_model.predict_proba(df)[:, 1][0])
        return p_xgb, warnings

    # ---------------------------------------------------------------- Branch 2
    def _encode_month(self, month: dict) -> Tuple[np.ndarray, List[str]]:
        warnings: List[str] = []
        features = dict(month.get("features") or {})

        unknown = [k for k in features if k not in self.non_cat_cols]
        if unknown:
            warnings.append(
                f"{len(unknown)} unknown monthly_statements feature key(s) ignored."
            )

        numeric_row = np.array(
            [float(features.get(c, 0.0)) for c in self.non_cat_cols], dtype="float32"
        )

        cat_values = []
        for col in self.cat_cols:
            raw = month.get(col)
            categories = self.b1_cat_categories.get(col) or []
            if raw is None:
                cat_values.append(categories[0] if categories else 0)
                warnings.append(f"{col} not given for a month; defaulted to '{categories[0] if categories else 0}'.")
            elif isinstance(raw, str):
                cat_values.append(raw)
            else:
                # already an encoded numeric code -- pass through
                cat_values.append(raw)

        # Ordinal-encode any string categories; numeric codes pass straight through.
        needs_encoding = any(isinstance(v, str) for v in cat_values)
        if needs_encoding:
            encoded = self.ord_enc.transform(pd.DataFrame([cat_values], columns=self.cat_cols))[0]
        else:
            encoded = np.array(cat_values, dtype="float32")

        row = np.concatenate([numeric_row, np.asarray(encoded, dtype="float32")])
        return row, warnings

    def score_branch2(self, monthly_statements: List[dict]) -> Tuple[float, List[str]]:
        warnings: List[str] = []
        rows = []
        for month in monthly_statements[-self.max_len :]:  # keep only the most recent max_len, like truncating='pre'
            row, w = self._encode_month(month)
            warnings.extend(w)
            rows.append(row)

        seq_raw = np.stack(rows).astype("float32")
        seq_df = pd.DataFrame(seq_raw, columns=self.seq_cols)
        seq_scaled = self.scaler.transform(seq_df)

        X_seq = keras.preprocessing.sequence.pad_sequences(
            [seq_scaled], maxlen=self.max_len, dtype="float32", padding="pre", truncating="pre", value=0.0
        )
        pred = self.gru_model.predict(X_seq, verbose=0)
        p_gru = float(pred[0, self.max_len - 1, 0])
        return p_gru, warnings

    # ---------------------------------------------------------------- Fusion
    def fuse(self, p_xgb: float, p_gru: float) -> Tuple[float, str, str, float, float]:
        z_xgb = _logit(p_xgb)
        z_gru = _logit(p_gru)
        Z = np.array([[z_xgb, z_gru]])
        p_fused = float(self.meta.predict_proba(Z)[:, 1][0])

        bins = [-np.inf] + list(self.risk_bins) + [np.inf]
        idx = int(np.digitize([p_fused], bins, right=False)[0] - 1)
        idx = min(max(idx, 0), len(self.risk_levels) - 1)
        risk_level = self.risk_levels[idx]
        action = self.risk_actions[risk_level]

        contrib_xgb = abs(self.fusion_coefficients["xgb"] * z_xgb)
        contrib_gru = abs(self.fusion_coefficients["gru"] * z_gru)
        total = contrib_xgb + contrib_gru
        xgb_pct = contrib_xgb / total if total > 0 else 0.5
        gru_pct = contrib_gru / total if total > 0 else 0.5

        return p_fused, risk_level, action, xgb_pct, gru_pct

    # ---------------------------------------------------------------- End to end
    def predict(self, xgb_features: dict, monthly_statements: List[dict]) -> dict:
        p_xgb, w1 = self.score_branch1(xgb_features)
        p_gru, w2 = self.score_branch2(monthly_statements)
        p_fused, risk_level, action, xgb_pct, gru_pct = self.fuse(p_xgb, p_gru)
        self.drift_monitor.log_prediction(monthly_statements, p_fused, risk_level)
        return {
            "p_xgb": p_xgb,
            "p_gru": p_gru,
            "p_fused": p_fused,
            "risk_level": risk_level,
            "action": action,
            "xgb_contribution_pct": xgb_pct,
            "gru_contribution_pct": gru_pct,
            "warnings": w1 + w2,
        }

    def sample_request(self) -> dict:
        """A structurally valid but synthetic request body, for smoke-testing
        the API (e.g. via /docs) without hand-building a 1,802-key payload.
        This is NOT a real customer -- every numeric value is 0.0 and every
        category is the first one in that column's category list.
        """
        xgb_features = {f: 0.0 for f in self.b1_features}
        month = {
            "features": {c: 0.0 for c in self.non_cat_cols},
            "D_63": self.b1_cat_categories.get("D_63", ["CL"])[0],
            "D_64": self.b1_cat_categories.get("D_64", ["O"])[0],
        }
        return {
            "customer_id": "SAMPLE-SYNTHETIC",
            "xgb_features": xgb_features,
            "monthly_statements": [month, month, month],
        }

    def drift_report(self) -> dict:
        return self.drift_monitor.report()

    def drift_summary(self) -> dict:
        return self.drift_monitor.summary()

    def model_info(self) -> dict:
        return {
            "branch1_n_features": len(self.b1_features),
            "branch1_features": self.b1_features,
            "branch2_sequence_features": self.non_cat_cols,
            "branch2_categorical_features": self.cat_cols,
            "branch2_categorical_categories": self.b1_cat_categories,
            "max_sequence_length": self.max_len,
            "risk_bins": self.risk_bins,
            "risk_levels": self.risk_levels,
            "risk_actions": self.risk_actions,
            "fusion_coefficients": self.fusion_coefficients,
        }
