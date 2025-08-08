"""MLflow model loading utilities for inference service"""

import logging
from typing import Any, Dict

import mlflow
import mlflow.sklearn
from sklearn.preprocessing import LabelEncoder

# Fix config import
try:
    from .config import FEATURE_CONFIG, INFERENCE_CONFIG, MLFLOW_CONFIG
except ImportError:
    from config import FEATURE_CONFIG, INFERENCE_CONFIG, MLFLOW_CONFIG

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelLoader:
    """Handles model loading and preprocessing artifacts from MLflow"""

    def __init__(self):
        self.model = None
        self.model_version = None
        self.model_stage = None
        self.feature_names = None
        self.encoders = {}
        self._setup_mlflow()

    def _setup_mlflow(self):
        """Initialize MLflow connection"""
        mlflow.set_tracking_uri(MLFLOW_CONFIG["tracking_uri"])
        logger.info(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")

    def load_latest_model(self) -> bool:
        """Load model with fallback: Production -> Staging -> Latest by version"""
        try:
            model_name = INFERENCE_CONFIG["model_name"]
            client = mlflow.MlflowClient()

            # Try Production first
            try:
                production_models = client.get_latest_versions(
                    model_name, stages=["Production"]
                )
                if production_models:
                    model_uri = f"models:/{model_name}/Production"
                    self.model = mlflow.sklearn.load_model(model_uri)
                    self.model_version = production_models[0].version
                    self.model_stage = "Production"
                    self.feature_names = (
                        FEATURE_CONFIG["numeric_features"]
                        + FEATURE_CONFIG["categorical_features"]
                    )
                    self._initialize_encoders()
                    logger.info(
                        f"Production model loaded: {model_name} v{self.model_version}"
                    )
                    return True
            except Exception as e:
                logger.warning(f"No Production model found: {e}")

            # Fallback to Staging
            try:
                staging_models = client.get_latest_versions(
                    model_name, stages=["Staging"]
                )
                if staging_models:
                    model_uri = f"models:/{model_name}/Staging"
                    self.model = mlflow.sklearn.load_model(model_uri)
                    self.model_version = staging_models[0].version
                    self.model_stage = "Staging"
                    self.feature_names = (
                        FEATURE_CONFIG["numeric_features"]
                        + FEATURE_CONFIG["categorical_features"]
                    )
                    self._initialize_encoders()
                    logger.info(
                        f"Staging model loaded: {model_name} v{self.model_version}"
                    )
                    return True
            except Exception as e:
                logger.warning(f"No Staging model found: {e}")

            # Final fallback: Latest version regardless of stage
            try:
                all_versions = client.search_model_versions(f"name='{model_name}'")
                if all_versions:
                    # Sort by version number (descending) and take the latest
                    latest_version = max(all_versions, key=lambda x: int(x.version))
                    model_uri = f"models:/{model_name}/{latest_version.version}"
                    self.model = mlflow.sklearn.load_model(model_uri)
                    self.model_version = latest_version.version
                    self.model_stage = latest_version.current_stage or "None"
                    self.feature_names = (
                        FEATURE_CONFIG["numeric_features"]
                        + FEATURE_CONFIG["categorical_features"]
                    )
                    self._initialize_encoders()
                    logger.info(
                        "Latest version model loaded: %s v%s (stage: %s)",
                        model_name,
                        self.model_version,
                        self.model_stage,
                    )
                    return True
            except Exception as e:
                logger.error(f"Failed to load latest version: {e}")

            # No models found at all
            logger.error(f"No models found in registry '{model_name}'.")
            logger.error("TO FIX: Run training pipeline first to create a model")
            return False

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    def _initialize_encoders(self):
        """Initialize label encoders for categorical features"""
        # These will be fitted with reference data or use predefined mappings
        self.encoders = {"budget": LabelEncoder(), "genre": LabelEncoder()}

        # Fit with known categories from your training data
        budget_categories = FEATURE_CONFIG["budget_labels"]
        self.encoders["budget"].fit(budget_categories)

        # Expanded list of common genres from movie datasets
        common_genres = [
            "Action",
            "Adventure",
            "Animation",
            "Comedy",
            "Crime",
            "Documentary",
            "Drama",
            "Family",
            "Fantasy",
            "History",
            "Horror",
            "Music",
            "Mystery",
            "Romance",
            "Science Fiction",
            "TV Movie",
            "Thriller",
            "War",
            "Western",
            "Unknown",
        ]
        self.encoders["genre"].fit(common_genres)

        logger.info("Encoders initialized successfully")

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded model"""
        if not self.model:
            return {
                "status": "No model loaded",
                "model_name": INFERENCE_CONFIG["model_name"],
                "tracking_uri": MLFLOW_CONFIG["tracking_uri"],
            }

        return {
            "status": "Model loaded",
            "model_name": INFERENCE_CONFIG["model_name"],
            "model_version": self.model_version,
            "model_stage": self.model_stage,
            "model_type": type(self.model).__name__,
            "feature_count": len(self.feature_names) if self.feature_names else 0,
            "features": self.feature_names,
            "tracking_uri": MLFLOW_CONFIG["tracking_uri"],
        }

    def is_model_loaded(self) -> bool:
        """Check if model is loaded and ready"""
        return self.model is not None

    def get_model_stages_info(self) -> Dict[str, Any]:
        """Get information about available model stages"""
        try:
            model_name = INFERENCE_CONFIG["model_name"]
            client = mlflow.MlflowClient()
            all_versions = client.search_model_versions(f"name='{model_name}'")

            if not all_versions:
                return {"error": "No models found", "model_name": model_name}

            stages_info = {}
            for version in all_versions:
                stage = (
                    version.current_stage
                    if version.current_stage != "None"
                    else "Unassigned"
                )
                if stage not in stages_info:
                    stages_info[stage] = []
                stages_info[stage].append(
                    {
                        "version": version.version,
                        "creation_timestamp": version.creation_timestamp,
                        "last_updated_timestamp": version.last_updated_timestamp,
                    }
                )

            return {
                "model_name": model_name,
                "total_versions": len(all_versions),
                "stages": stages_info,
            }

        except Exception as e:
            return {"error": str(e)}


# Global model loader instance
model_loader = ModelLoader()