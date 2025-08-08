"""Drift detection using Evidently AI"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

try:
    from evidently import ColumnMapping
    from evidently.metric_preset import DataDriftPreset, DataQualityPreset
    from evidently.metrics import DatasetDriftMetric, DatasetMissingValuesMetric
    from evidently.report import Report

except ImportError as e:
    logging.error(f"Evidently import failed: {e}")

# Fix imports
try:
    from .config import FEATURE_CONFIG, MONITORING_CONFIG
except ImportError:
    from config import FEATURE_CONFIG, MONITORING_CONFIG

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DriftDetector:
    """Handles drift detection using Evidently AI"""

    def __init__(self):
        self.reference_data = None
        self.column_mapping = None
        self.drift_history = []
        self.reports_dir = Path(MONITORING_CONFIG["reports_dir"])
        self.drift_history_file = (
            Path(MONITORING_CONFIG["monitoring_data_dir"]) / "drift_history.json"
        )
        self._setup_column_mapping()
        self._load_drift_history()

    def _setup_column_mapping(self):
        """Setup Evidently column mapping"""
        try:
            # Define column mapping for Evidently
            self.column_mapping = ColumnMapping()

            # Numerical features
            self.column_mapping.numerical_features = FEATURE_CONFIG["numeric_features"]

            # Categorical features
            self.column_mapping.categorical_features = FEATURE_CONFIG[
                "categorical_features"
            ]

            # Target column (if available)
            self.column_mapping.target = "is_hit"

            logger.info("Column mapping setup completed")

        except Exception as e:
            logger.error(f"Failed to setup column mapping: {e}")
            self.column_mapping = None

    def initialize(self, reference_data: pd.DataFrame):
        """Initialize with reference data"""
        try:
            if reference_data is None or reference_data.empty:
                logger.warning("No reference data provided")
                return False

            self.reference_data = reference_data.copy()
            logger.info(
                "Drift detector initialized with %d reference samples",
                len(self.reference_data),
            )
            return True

        except Exception as e:
            logger.error(f"Failed to initialize drift detector: {e}")
            return False

    def detect_drift(self, current_data: List[Dict]) -> Dict[str, Any]:
        """Detect drift between reference and current data"""
        try:
            if self.reference_data is None:
                logger.warning("No reference data available for drift detection")
                return self._empty_drift_result()

            if not current_data:
                logger.warning("No current data provided for drift detection")
                return self._empty_drift_result()

            # Convert current data to DataFrame
            current_df = self._prepare_current_data(current_data)

            if current_df.empty:
                logger.warning("Current data is empty after preprocessing")
                return self._empty_drift_result()

            # Run drift detection
            drift_result = self._run_evidently_analysis(current_df)

            # Save drift result
            self._save_drift_result(drift_result)

            return drift_result

        except Exception as e:
            logger.error(f"Drift detection failed: {e}")
            return self._empty_drift_result(error=str(e))

    def _prepare_current_data(self, current_data: List[Dict]) -> pd.DataFrame:
        """Prepare current data for drift detection"""
        try:
            # Convert to DataFrame
            df = pd.DataFrame(current_data)

            if df.empty:
                return df

            # Ensure we have the same columns as reference data
            reference_columns = set(self.reference_data.columns)
            current_columns = set(df.columns)

            # Add missing columns with default values
            missing_columns = reference_columns - current_columns
            for col in missing_columns:
                if col in FEATURE_CONFIG["numeric_features"]:
                    df[col] = 0.0
                else:
                    df[col] = "Unknown"

            # Select only reference columns
            df = df[list(reference_columns)]

            # Handle data types
            for col in FEATURE_CONFIG["numeric_features"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

            for col in FEATURE_CONFIG["categorical_features"]:
                if col in df.columns:
                    df[col] = df[col].astype(str).fillna("Unknown")

            logger.info(f"Prepared {len(df)} current samples for drift detection")
            return df

        except Exception as e:
            logger.error(f"Failed to prepare current data: {e}")
            return pd.DataFrame()

    def _run_evidently_analysis(self, current_data: pd.DataFrame) -> Dict[str, Any]:
        """Run Evidently AI analysis"""
        try:
            # Create data drift report
            data_drift_report = Report(
                metrics=[
                    DataDriftPreset(),
                    DataQualityPreset(),
                    DatasetDriftMetric(),
                    DatasetMissingValuesMetric(),
                ]
            )

            # Run the report
            data_drift_report.run(
                reference_data=self.reference_data,
                current_data=current_data,
                column_mapping=self.column_mapping,
            )

            # Extract results
            drift_results = self._extract_drift_metrics(data_drift_report)

            # Save HTML report
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = self.reports_dir / f"drift_report_{timestamp}.html"
            data_drift_report.save_html(str(report_file))

            drift_results["report_file"] = str(report_file)
            drift_results["timestamp"] = datetime.now().isoformat()
            drift_results["predictions_count"] = len(current_data)

            logger.info(
                f"Evidently analysis completed. Drift score: {drift_results['data_drift_score']:.3f}"
            )
            return drift_results

        except Exception as e:
            logger.error(f"Evidently analysis failed: {e}")
            return self._fallback_drift_analysis(current_data)

    def _extract_drift_metrics(self, report: Report) -> Dict[str, Any]:
        """Extract key metrics from Evidently report"""
        try:
            # Get report results as dict
            report_dict = report.as_dict()

            # Extract data drift metrics
            data_drift_score = 0.0
            drifted_features = []

            # Look for drift metrics in the report
            for metric in report_dict.get("metrics", []):
                if metric.get("metric") == "DatasetDriftMetric":
                    result = metric.get("result", {})
                    data_drift_score = result.get("drift_score", 0.0)

                    # Count drifted features
                    drift_by_columns = result.get("drift_by_columns", {})
                    drifted_features = [
                        col
                        for col, info in drift_by_columns.items()
                        if info.get("drift_detected", False)
                    ]

            # Calculate additional metrics
            performance_drift = self._calculate_performance_drift(report_dict)
            avg_confidence = self._calculate_avg_confidence()

            return {
                "data_drift_score": data_drift_score,
                "performance_drift": performance_drift,
                "avg_confidence": avg_confidence,
                "drifted_features": drifted_features,
                "drift_detected": data_drift_score
                > MONITORING_CONFIG["data_drift_threshold"],
                "quality_issues": self._extract_quality_issues(report_dict),
            }

        except Exception as e:
            logger.error(f"Failed to extract drift metrics: {e}")
            return {
                "data_drift_score": 0.0,
                "performance_drift": 0.0,
                "avg_confidence": 1.0,
                "drifted_features": [],
                "drift_detected": False,
                "quality_issues": [],
            }

    def _calculate_performance_drift(self, report_dict: Dict) -> float:
        """Calculate performance drift metric"""
        try:
            # This is a simplified performance drift calculation
            # In practice, you'd compare model accuracy on reference vs current data

            # For now, use data drift as proxy for performance drift
            for metric in report_dict.get("metrics", []):
                if metric.get("metric") == "DatasetDriftMetric":
                    drift_score = metric.get("result", {}).get("drift_score", 0.0)
                    # Convert drift score to performance impact estimate
                    return min(drift_score * 0.5, 1.0)

            return 0.0

        except Exception as e:
            logger.error(f"Failed to calculate performance drift: {e}")
            return 0.0

    def _calculate_avg_confidence(self) -> float:
        """Calculate average prediction confidence"""
        try:
            # This would typically be calculated from actual predictions
            # For now, return a default value
            # In practice, you'd track confidence scores from predictions
            return 0.8

        except Exception as e:
            logger.error(f"Failed to calculate confidence: {e}")
            return 1.0

    def _extract_quality_issues(self, report_dict: Dict) -> List[str]:
        """Extract data quality issues"""
        try:
            issues = []

            for metric in report_dict.get("metrics", []):
                if metric.get("metric") == "DatasetMissingValuesMetric":
                    result = metric.get("result", {})
                    current_missing = result.get("current", {}).get(
                        "number_of_missing_values", 0
                    )
                    reference_missing = result.get("reference", {}).get(
                        "number_of_missing_values", 0
                    )

                    if current_missing > reference_missing * 1.5:
                        issues.append(
                            f"Increased missing values: {current_missing} vs {reference_missing}"
                        )

            return issues

        except Exception as e:
            logger.error(f"Failed to extract quality issues: {e}")
            return []

    def _fallback_drift_analysis(self, current_data: pd.DataFrame) -> Dict[str, Any]:
        """Fallback drift analysis when Evidently fails"""
        try:
            logger.info("Running fallback drift analysis...")

            # Simple statistical drift detection
            drift_score = 0.0
            drifted_features = []

            for col in FEATURE_CONFIG["numeric_features"]:
                if col in self.reference_data.columns and col in current_data.columns:
                    ref_mean = self.reference_data[col].mean()
                    curr_mean = current_data[col].mean()
                    ref_std = self.reference_data[col].std()

                    if ref_std > 0:
                        # Z-score based drift
                        z_score = abs(curr_mean - ref_mean) / ref_std
                        if z_score > 2.0:  # 2 standard deviations
                            drift_score += z_score / 10.0  # Normalize
                            drifted_features.append(col)

            return {
                "data_drift_score": min(drift_score, 1.0),
                "performance_drift": min(drift_score * 0.5, 1.0),
                "avg_confidence": 0.8,
                "drifted_features": drifted_features,
                "drift_detected": drift_score
                > MONITORING_CONFIG["data_drift_threshold"],
                "quality_issues": [],
                "fallback_used": True,
            }

        except Exception as e:
            logger.error(f"Fallback analysis failed: {e}")
            return self._empty_drift_result()

    def _empty_drift_result(self, error: str = None) -> Dict[str, Any]:
        """Return empty drift result"""
        return {
            "data_drift_score": 0.0,
            "performance_drift": 0.0,
            "avg_confidence": 1.0,
            "drifted_features": [],
            "drift_detected": False,
            "quality_issues": [],
            "predictions_count": 0,
            "timestamp": datetime.now().isoformat(),
            "error": error,
        }

    def _save_drift_result(self, drift_result: Dict[str, Any]):
        """Save drift result to history"""
        try:
            # Add to history
            self.drift_history.append(drift_result)

            # Keep only last 100 results
            if len(self.drift_history) > 100:
                self.drift_history = self.drift_history[-100:]

            # Save to file
            with open(self.drift_history_file, "w") as f:
                json.dump(self.drift_history, f, indent=2, default=str)

            logger.info("Drift result saved to history")

        except Exception as e:
            logger.error(f"Failed to save drift result: {e}")

    def _load_drift_history(self):
        """Load drift history from file"""
        try:
            if self.drift_history_file.exists():
                with open(self.drift_history_file, "r") as f:
                    self.drift_history = json.load(f)
                logger.info(f"Loaded {len(self.drift_history)} drift history records")
            else:
                self.drift_history = []

        except Exception as e:
            logger.error(f"Failed to load drift history: {e}")
            self.drift_history = []

    def get_drift_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent drift history"""
        return self.drift_history[-limit:] if self.drift_history else []

    def get_drift_summary(self) -> Dict[str, Any]:
        """Get drift summary statistics"""
        try:
            if not self.drift_history:
                return {"status": "no_data"}

            recent_results = self.drift_history[-10:]  # Last 10 results

            avg_drift_score = np.mean(
                [r.get("data_drift_score", 0) for r in recent_results]
            )
            max_drift_score = np.max(
                [r.get("data_drift_score", 0) for r in recent_results]
            )
            drift_trend = len(
                [r for r in recent_results if r.get("drift_detected", False)]
            )

            return {
                "status": "active",
                "avg_drift_score": float(avg_drift_score),
                "max_drift_score": float(max_drift_score),
                "drift_detections": drift_trend,
                "total_checks": len(self.drift_history),
                "last_check": self.drift_history[-1].get("timestamp")
                if self.drift_history
                else None,
            }

        except Exception as e:
            logger.error(f"Failed to generate drift summary: {e}")
            return {"status": "error", "error": str(e)}
