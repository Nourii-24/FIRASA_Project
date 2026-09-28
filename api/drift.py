"""Data drift monitoring -- the third component of the proposal's Phase 6
('...expose it using a REST API (FastAPI), and establish monitoring tools
to track data drift over time').

Two kinds of drift are tracked, both against REAL reference numbers already
produced by earlier phases of the project -- nothing here is a synthetic or
assumed baseline:

1. INPUT drift, for a curated set of the model's most important numeric
   Branch-2 features -- the ones the report and the GRU saliency analysis
   single out as the strongest signals (P_2, B_1, B_8, D_48), rounded out
   with a couple more from each of the five interpretable feature groups
   (payment, balance, delinquency, spend, risk) so every group is covered.
   The reference mean/std for each of these comes straight from
   models/branch2_scaler.joblib -- the real StandardScaler fit on the
   training data, not a re-estimated or assumed number.

   This is NOT scored as a textbook Population Stability Index. PSI needs
   a reference *shape* (a histogram/quantiles from the real training
   sample), and the serving side here only has that sample's mean and std
   (the two numbers a StandardScaler stores), not the raw values. An
   earlier version of this module assumed the reference was Gaussian and
   built PSI bins from it; testing that against synthetic in-distribution
   traffic showed it flags features as "drifting" purely from ordinary
   skew (several of these engineered columns are bounded near 0 with a
   long right tail, not remotely Gaussian), which would be a false alarm
   every time, not a real one. So input drift instead uses a one-sample
   z-test on the mean: z = (live_window_mean - reference_mean) /
   (reference_std / sqrt(n)). By the central limit theorem this is a valid
   test that the live window's mean differs from the reference population
   mean regardless of that population's actual shape, as long as n isn't
   tiny. It catches a genuine mean shift (the headline symptom of feature
   drift -- a change in who's applying, not just noise); it will NOT catch
   a shape-only change that leaves the mean unchanged (e.g. the variance
   widening). live_std / reference_std is reported alongside it as an
   informational spread_ratio for exactly that reason, without its own
   pass/fail threshold.

2. OUTPUT drift: the live distribution of predicted risk levels compared
   to the reference distribution measured on the real 8,297-customer
   held-out test set at the last statement (Low 56.7%, Moderate 8.9%,
   High 10.0%, Critical 24.4% -- notebook 06's export, also reported in
   the final report's dashboard section). This one IS real PSI: risk level
   is already a 4-category variable with a known real reference proportion
   per category, so no shape assumption is needed.

What this does NOT cover: drift on the 1,802 Branch-1 (XGBoost) aggregated
features. No per-feature reference mean/std for that feature set survives
into any deployed artifact (Branch 1 needs no scaler, unlike Branch 2), and
recomputing one needs the original training data, which this service does
not have access to. Flagged in api/README.md as a follow-up, alongside live
raw-statement scoring.

Persistence: each prediction's monitored values are appended as one JSON
line to FIRASA_LOGS_DIR/predictions.jsonl (see config.py). At startup, the
monitor replays up to `window_size` of the most recent lines so the rolling
window -- and therefore "over time" drift tracking -- survives a container
restart, as long as that logs/ directory is mounted as a volume (see
docker-compose.yml). Every write/read is best-effort: a read-only or
missing logs directory degrades to in-memory-only monitoring rather than
failing a request.
"""
import json
import logging
import math
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, List, Optional

import numpy as np

logger = logging.getLogger("firasa.drift")

# Curated Branch-2 features to monitor for input drift.
MONITORED_FEATURES = ["P_2", "B_1", "B_2", "B_3", "B_8", "D_39", "D_41", "D_48", "S_3", "R_1"]

# Reference output (risk-level) distribution, measured on the real
# 8,297-customer held-out test set at the last statement.
REFERENCE_RISK_LEVEL_DIST = {
    "Low": 4696 / 8297,
    "Moderate": 705 / 8297,
    "High": 856 / 8297,
    "Critical": 2040 / 8297,
}
RISK_LEVEL_ORDER = ["Low", "Moderate", "High", "Critical"]

# Thresholds for the mean-shift z-test (input drift, per feature): |z| below
# Z_STABLE is ordinary sampling noise, up to Z_MODERATE is worth a look, past
# that is a real shift at well past the 99.99th percentile of chance alone.
Z_STABLE = 2.0
Z_MODERATE = 4.0

# Thresholds for PSI (output/risk-level drift only -- see module docstring
# for why input drift does not also use PSI). These are the standard
# textbook cutoffs for PSI specifically, not comparable to the z-test above.
PSI_STABLE = 0.1
PSI_MODERATE = 0.25

MIN_SAMPLES_TO_REPORT = 5
RELIABLE_SAMPLES = 30

_SEVERITY = {"stable": 0, "moderate_shift": 1, "significant_shift": 2}


def _status_from_z(z: float) -> str:
    az = abs(z)
    if az < Z_STABLE:
        return "stable"
    if az < Z_MODERATE:
        return "moderate_shift"
    return "significant_shift"


def psi_status(value: float) -> str:
    if value < PSI_STABLE:
        return "stable"
    if value < PSI_MODERATE:
        return "moderate_shift"
    return "significant_shift"


def _psi(expected_pct: np.ndarray, actual_pct: np.ndarray, eps: float = 1e-4) -> float:
    expected_pct = np.clip(expected_pct, eps, None)
    actual_pct = np.clip(actual_pct, eps, None)
    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))


@dataclass
class _FeatureReference:
    mean: float
    std: float


class DriftMonitor:
    """In-memory rolling-window drift monitor, optionally backed by a JSONL
    log on disk for continuity across restarts. This is a lightweight,
    dependency-free complement to a real monitoring stack (Evidently /
    Grafana / a proper feature store) -- matching the project's own stated
    scope ('academic proof-of-concept, not production-ready').
    """

    def __init__(
        self,
        feature_names: List[str],
        means: np.ndarray,
        stds: np.ndarray,
        window_size: int = 500,
        log_path: Optional[Path] = None,
    ):
        self.window_size = window_size
        self._log_path = log_path
        self._reference: Dict[str, _FeatureReference] = {}
        name_to_idx = {n: i for i, n in enumerate(feature_names)}
        for feat in MONITORED_FEATURES:
            if feat not in name_to_idx:
                logger.warning("Monitored feature %r not found in scaler; skipping.", feat)
                continue
            i = name_to_idx[feat]
            self._reference[feat] = _FeatureReference(mean=float(means[i]), std=float(stds[i]))

        self._feature_window: Dict[str, Deque[float]] = {f: deque(maxlen=window_size) for f in self._reference}
        self._risk_level_window: Deque[str] = deque(maxlen=window_size)
        self._p_fused_window: Deque[float] = deque(maxlen=window_size)
        self._n_logged_this_process = 0
        self._started_at = time.time()
        self._replayed = 0

        if self._log_path is not None:
            self._load_history()

    # ------------------------------------------------------------- logging
    def log_prediction(self, monthly_statements: List[dict], p_fused: float, risk_level: str) -> None:
        self._n_logged_this_process += 1
        self._p_fused_window.append(p_fused)
        self._risk_level_window.append(risk_level)

        feature_values: Dict[str, float] = {}
        if monthly_statements:
            last_month_features = dict(monthly_statements[-1].get("features") or {})
            for feat in self._reference:
                if feat in last_month_features:
                    try:
                        val = float(last_month_features[feat])
                    except (TypeError, ValueError):
                        continue
                    self._feature_window[feat].append(val)
                    feature_values[feat] = val

        self._append_log(
            {
                "ts": time.time(),
                "risk_level": risk_level,
                "p_fused": p_fused,
                "features": feature_values,
            }
        )

    def _append_log(self, record: dict) -> None:
        if self._log_path is None:
            return
        try:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except OSError as exc:
            logger.warning("Could not write drift log to %s: %s", self._log_path, exc)

    def _load_history(self) -> None:
        if not self._log_path.exists():
            return
        try:
            with open(self._log_path, "r", encoding="utf-8") as f:
                lines = f.readlines()[-self.window_size :]
        except OSError as exc:
            logger.warning("Could not read drift log %s for replay: %s", self._log_path, exc)
            return

        replayed = 0
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            risk_level = rec.get("risk_level")
            p_fused = rec.get("p_fused")
            if risk_level is not None:
                self._risk_level_window.append(risk_level)
            if p_fused is not None:
                self._p_fused_window.append(p_fused)
            for feat, val in (rec.get("features") or {}).items():
                if feat in self._feature_window:
                    self._feature_window[feat].append(val)
            replayed += 1
        self._replayed = replayed
        if replayed:
            logger.info("Drift monitor: replayed %d historical prediction(s) from %s", replayed, self._log_path)

    # ------------------------------------------------------------- reports
    def _feature_drift(self, feat: str) -> Optional[dict]:
        ref = self._reference.get(feat)
        window = self._feature_window.get(feat)
        if ref is None or window is None or len(window) < MIN_SAMPLES_TO_REPORT:
            return None
        arr = np.array(window, dtype="float64")
        n = len(arr)
        live_mean = float(arr.mean())
        live_std = float(arr.std())
        ref_std = ref.std if ref.std > 1e-9 else 1e-9
        standard_error = ref_std / math.sqrt(n)
        z = (live_mean - ref.mean) / standard_error
        return {
            "feature": feat,
            "reference_mean": round(ref.mean, 4),
            "reference_std": round(ref.std, 4),
            "live_mean": round(live_mean, 4),
            "live_std": round(live_std, 4),
            "n_observations": n,
            "mean_shift_z": round(z, 3),
            "spread_ratio": round(live_std / ref_std, 3),
            "status": _status_from_z(z),
            "reliable": n >= RELIABLE_SAMPLES,
        }

    def _output_psi(self) -> Optional[dict]:
        n = len(self._risk_level_window)
        if n < MIN_SAMPLES_TO_REPORT:
            return None
        counts = {lvl: 0 for lvl in RISK_LEVEL_ORDER}
        for lvl in self._risk_level_window:
            if lvl in counts:
                counts[lvl] += 1
        total = sum(counts.values()) or 1
        actual_pct = np.array([counts[lvl] / total for lvl in RISK_LEVEL_ORDER])
        expected_pct = np.array([REFERENCE_RISK_LEVEL_DIST[lvl] for lvl in RISK_LEVEL_ORDER])
        value = _psi(expected_pct, actual_pct)
        return {
            "reference_distribution": {k: round(v, 4) for k, v in REFERENCE_RISK_LEVEL_DIST.items()},
            "live_distribution": {lvl: round(counts[lvl] / total, 4) for lvl in RISK_LEVEL_ORDER},
            "n_observations": total,
            "psi": round(value, 4),
            "status": psi_status(value),
            "reliable": total >= RELIABLE_SAMPLES,
        }

    def report(self) -> dict:
        feature_reports = []
        for f in MONITORED_FEATURES:
            r = self._feature_drift(f)
            if r is not None:
                feature_reports.append(r)
        output_report = self._output_psi()

        statuses = [r["status"] for r in feature_reports]
        if output_report:
            statuses.append(output_report["status"])

        return {
            "window_size": self.window_size,
            "predictions_logged_this_process": self._n_logged_this_process,
            "predictions_replayed_from_log": self._replayed,
            "predictions_in_window": len(self._risk_level_window),
            "monitoring_process_started_at": self._started_at,
            "input_drift": feature_reports,
            "input_drift_method": (
                "One-sample z-test of the live window's mean against each feature's real "
                "training-time reference mean/std (models/branch2_scaler.joblib) -- "
                "mean_shift_z, thresholds |z|<2 stable / <4 moderate_shift / else "
                "significant_shift. spread_ratio (live_std/reference_std) is informational "
                "only. Only these curated Branch-2 features are monitored, spanning all five "
                "interpretable groups; Branch-1's 1,802 aggregated features have no reference "
                "mean/std available to this service (see api/README.md). A feature needs at "
                f"least {MIN_SAMPLES_TO_REPORT} requests in the window to report at all, and "
                f"{RELIABLE_SAMPLES}+ ('reliable': true) before the result should be trusted."
            ),
            "output_drift": output_report,
            "output_drift_method": (
                "Standard Population Stability Index (PSI) of the live risk_level distribution "
                "against the real reference distribution from the 8,297-customer held-out test "
                "set. Thresholds: PSI<0.1 stable / <0.25 moderate_shift / else significant_shift."
            ),
            "overall_status": (
                max(statuses, key=lambda s: _SEVERITY[s]) if statuses else "insufficient_data"
            ),
        }

    def summary(self) -> dict:
        return {
            "predictions_logged_this_process": self._n_logged_this_process,
            "predictions_replayed_from_log": self._replayed,
            "predictions_in_window": len(self._risk_level_window),
            "window_size": self.window_size,
            "monitored_features": list(self._reference.keys()),
            "log_path": str(self._log_path) if self._log_path else None,
        }
