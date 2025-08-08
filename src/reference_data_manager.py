"""Reference data management for monitoring baseline"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

# Fix imports
try:
    from .config import DATA_CONFIG, FEATURE_CONFIG, MONITORING_CONFIG
except ImportError:
    from config import DATA_CONFIG, FEATURE_CONFIG, MONITORING_CONFIG

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ReferenceDataManager:
    """Manages reference dataset for drift detection"""

    def __init__(self):
        self.reference_data = None
        self.reference_stats = None
        self.is_data_loaded = False
        self.reference_file_path = (
            Path(MONITORING_CONFIG["monitoring_data_dir"]) / "reference_data.csv"
        )
        self.stats_file_path = (
            Path(MONITORING_CONFIG["monitoring_data_dir"]) / "reference_stats.json"
        )

    def load_reference_data(self) -> bool:
        """Load or create reference dataset"""
        try:
            # Try to load existing reference data
            if self.reference_file_path.exists():
                logger.info("Loading existing reference data...")
                self.reference_data = pd.read_csv(self.reference_file_path)

                if self.stats_file_path.exists():
                    with open(self.stats_file_path, "r") as f:
                        self.reference_stats = json.load(f)

                self.is_data_loaded = True
                logger.info(
                    f"Reference data loaded: {len(self.reference_data)} samples"
                )
                return True
            else:
                # Create new reference data from training dataset
                return self._create_reference_data()

        except Exception as e:
            logger.error(f"Failed to load reference data: {e}")
            return False

    def _create_reference_data(self) -> bool:
        """Create reference dataset from original training data"""
        try:
            logger.info("Creating new reference dataset...")

            # Load original dataset
            original_data = pd.read_csv(DATA_CONFIG["data_path"])

            # Sample data for reference (to avoid huge reference sets)
            sample_size = min(
                MONITORING_CONFIG["reference_data_size"], len(original_data)
            )
            reference_sample = original_data.sample(n=sample_size, random_state=42)

            # Preprocess the same way as training pipeline
            processed_reference = self._preprocess_reference_data(reference_sample)

            # Calculate statistics
            self.reference_stats = self._calculate_reference_stats(processed_reference)

            # Save reference data and stats
            processed_reference.to_csv(self.reference_file_path, index=False)
            with open(self.stats_file_path, "w") as f:
                json.dump(self.reference_stats, f, indent=2)

            self.reference_data = processed_reference
            self.is_data_loaded = True

            logger.info(
                f"Reference dataset created: {len(self.reference_data)} samples"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to create reference data: {e}")
            return False

    def _preprocess_reference_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess reference data same as inference pipeline"""
        try:
            df = data.copy()

            # Convert release_date
            df["release_date"] = pd.to_datetime(df["release_date"])
            df["release_year"] = df["release_date"].dt.year

            # Budget categories
            df["budget_category"] = pd.cut(
                df["budget"],
                bins=FEATURE_CONFIG["budget_bins"],
                labels=FEATURE_CONFIG["budget_labels"],
            )

            # Genre features - simplified extraction
            df["genre_count"] = df["genres"].apply(self._extract_genre_count)
            df["main_genre"] = df["genres"].apply(self._extract_main_genre)

            # Language feature
            df["is_english"] = (df["original_language"] == "en").astype(int)

            # Handle missing values
            df["release_year"].fillna(df["release_year"].median(), inplace=True)
            df["budget_category"].fillna("Ultra_Low", inplace=True)
            df["main_genre"].fillna("Unknown", inplace=True)

            # Select features for monitoring
            feature_columns = (
                FEATURE_CONFIG["numeric_features"]
                + FEATURE_CONFIG["categorical_features"]
            )

            # Add target for reference (if available)
            if "revenue" in df.columns:
                df["roi"] = df["revenue"] / (df["budget"] + 1)
                roi_threshold = df["roi"].quantile(0.7)
                df["is_hit"] = (df["roi"] >= roi_threshold).astype(int)
                feature_columns.append("is_hit")

            return df[feature_columns]

        except Exception as e:
            logger.error(f"Preprocessing failed: {e}")
            raise

    def _extract_genre_count(self, genres_str: str) -> int:
        """Extract genre count from JSON string"""
        try:
            import json

            genres = json.loads(genres_str.replace("'", '"'))
            return len(genres)
        except Exception:
            return 0

    def _extract_main_genre(self, genres_str: str) -> str:
        """Extract main genre from JSON string"""
        try:
            import json

            genres = json.loads(genres_str.replace("'", '"'))
            return genres[0]["name"] if genres else "Unknown"
        except Exception:
            return "Unknown"

    def _calculate_reference_stats(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate statistics for reference data"""
        try:
            stats = {
                "sample_size": len(data),
                "creation_date": pd.Timestamp.now().isoformat(),
                "feature_stats": {},
                "data_quality": {},
            }

            # Feature statistics
            for column in data.columns:
                if data[column].dtype in ["int64", "float64"]:
                    # Numerical features
                    stats["feature_stats"][column] = {
                        "type": "numerical",
                        "mean": float(data[column].mean()),
                        "std": float(data[column].std()),
                        "min": float(data[column].min()),
                        "max": float(data[column].max()),
                        "median": float(data[column].median()),
                        "missing_rate": float(data[column].isnull().mean()),
                    }
                else:
                    # Categorical features
                    value_counts = data[column].value_counts()
                    stats["feature_stats"][column] = {
                        "type": "categorical",
                        "unique_count": int(data[column].nunique()),
                        "most_common": str(value_counts.index[0])
                        if len(value_counts) > 0
                        else None,
                        "most_common_freq": float(value_counts.iloc[0] / len(data))
                        if len(value_counts) > 0
                        else 0,
                        "missing_rate": float(data[column].isnull().mean()),
                        "categories": value_counts.head(10).to_dict(),
                    }

            # Data quality metrics
            stats["data_quality"] = {
                "total_missing_rate": float(
                    data.isnull().sum().sum() / (len(data) * len(data.columns))
                ),
                "duplicate_rate": float(data.duplicated().mean()),
                "feature_count": len(data.columns),
            }

            return stats

        except Exception as e:
            logger.error(f"Failed to calculate stats: {e}")
            return {}

    def get_reference_data(self) -> Optional[pd.DataFrame]:
        """Get reference data for drift detection"""
        return self.reference_data if self.is_data_loaded else None

    def get_reference_stats(self) -> Dict[str, Any]:
        """Get reference statistics"""
        return self.reference_stats if self.reference_stats else {}

    def is_loaded(self) -> bool:
        """Check if reference data is loaded"""
        return self.is_data_loaded

    def update_reference_data(self, new_data: pd.DataFrame) -> bool:
        """Update reference data with new baseline"""
        try:
            logger.info("Updating reference data...")

            # Preprocess new data
            processed_data = self._preprocess_reference_data(new_data)

            # Calculate new stats
            new_stats = self._calculate_reference_stats(processed_data)

            # Save updated reference
            processed_data.to_csv(self.reference_file_path, index=False)
            with open(self.stats_file_path, "w") as f:
                json.dump(new_stats, f, indent=2)

            # Update in memory
            self.reference_data = processed_data
            self.reference_stats = new_stats

            logger.info(f"Reference data updated: {len(self.reference_data)} samples")
            return True

        except Exception as e:
            logger.error(f"Failed to update reference data: {e}")
            return False

    def get_feature_distribution(self, feature_name: str) -> Dict[str, Any]:
        """Get distribution for specific feature"""
        if not self.is_data_loaded or feature_name not in self.reference_data.columns:
            return {}

        try:
            feature_data = self.reference_data[feature_name]

            if feature_data.dtype in ["int64", "float64"]:
                # Numerical distribution
                return {
                    "type": "numerical",
                    "histogram": np.histogram(feature_data.dropna(), bins=20)[
                        0
                    ].tolist(),
                    "bin_edges": np.histogram(feature_data.dropna(), bins=20)[
                        1
                    ].tolist(),
                    "stats": self.reference_stats["feature_stats"].get(
                        feature_name, {}
                    ),
                }
            else:
                # Categorical distribution
                value_counts = feature_data.value_counts()
                return {
                    "type": "categorical",
                    "categories": value_counts.index.tolist(),
                    "frequencies": value_counts.values.tolist(),
                    "stats": self.reference_stats["feature_stats"].get(
                        feature_name, {}
                    ),
                }

        except Exception as e:
            logger.error(f"Failed to get distribution for {feature_name}: {e}")
            return {}
