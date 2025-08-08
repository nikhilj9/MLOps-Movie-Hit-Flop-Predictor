"""Model training, evaluation, and MLflow integration with Prefect tasks"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import mlflow
import mlflow.sklearn
import pandas as pd
from prefect import flow, task
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV

# Fix config import
try:
    from .config import MLFLOW_CONFIG, MODEL_CONFIG
except ImportError:
    from config import MLFLOW_CONFIG, MODEL_CONFIG

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@task(name="setup-mlflow", retries=2, retry_delay_seconds=5)
def setup_mlflow() -> str:
    """Initialize MLflow tracking"""
    os.makedirs(
        os.path.dirname(MLFLOW_CONFIG["tracking_uri"].replace("sqlite:///", "")),
        exist_ok=True,
    )

    mlflow.set_tracking_uri(MLFLOW_CONFIG["tracking_uri"])
    mlflow.set_experiment(MLFLOW_CONFIG["experiment_name"])
    os.environ["MLFLOW_DEFAULT_ARTIFACT_ROOT"] = MLFLOW_CONFIG["artifacts_path"]

    tracking_uri = mlflow.get_tracking_uri()
    logger.info(f"MLflow tracking URI: {tracking_uri}")
    logger.info(f"Experiment: {MLFLOW_CONFIG['experiment_name']}")

    return tracking_uri


@task(name="train-random-forest", retries=1, retry_delay_seconds=10)
def train_random_forest(
    X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series
) -> Tuple[Any, float, float, str]:
    """Train and tune Random Forest model"""

    with mlflow.start_run(run_name="random_forest_production"):
        # Hyperparameter tuning
        grid_search = GridSearchCV(
            RandomForestClassifier(random_state=MODEL_CONFIG.get("random_state", 42)),
            MODEL_CONFIG["param_grid"],
            cv=MODEL_CONFIG["cv_folds"],
            scoring="roc_auc",
            n_jobs=-1,
        )

        logger.info("Starting hyperparameter tuning...")
        grid_search.fit(X_train, y_train)

        # Get best model
        best_model = grid_search.best_estimator_

        # Make predictions
        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)[:, 1]

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)

        # Log to MLflow
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("auc", auc)
        mlflow.log_metric("cv_best_score", grid_search.best_score_)

        # Log model
        mlflow.sklearn.log_model(
            best_model,
            "model",
            registered_model_name=MLFLOW_CONFIG["model_registry_name"],
        )

        # Store model URI
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"

        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Cross-validation score: {grid_search.best_score_:.3f}")
        logger.info(f"Test Accuracy: {accuracy:.3f}")
        logger.info(f"Test AUC: {auc:.3f}")
        logger.info(f"Model registered: {model_uri}")

        return best_model, accuracy, auc, model_uri


@task(name="evaluate-model", retries=1)
def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Evaluate model performance and return metrics"""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    # Feature importance
    feature_importance = None
    if hasattr(model, "feature_importances_") and feature_names:
        feature_importance = dict(zip(feature_names, model.feature_importances_))
        feature_importance = dict(
            sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        )

    results = {
        "accuracy": accuracy,
        "auc": auc,
        "classification_report": classification_report(y_test, y_pred),
        "feature_importance": feature_importance,
    }

    logger.info(
        f"Model evaluation completed - Accuracy: {accuracy:.3f}, AUC: {auc:.3f}"
    )

    return results


@flow(name="model-training-pipeline")
def model_training_flow(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Train the production-ready model as a Prefect flow"""
    logger.info("Starting production model training...")

    # Setup MLflow
    tracking_uri = setup_mlflow()

    # Train the best model (Random Forest based on your analysis)
    model, accuracy, auc, model_uri = train_random_forest(
        X_train, y_train, X_test, y_test
    )

    # Evaluate model
    evaluation = evaluate_model(model, X_test, y_test, feature_names)

    # Combine results
    results = {
        "model": model,
        "model_uri": model_uri,
        "accuracy": accuracy,
        "auc": auc,
        "evaluation": evaluation,
        "feature_names": feature_names,
        "tracking_uri": tracking_uri,
    }

    logger.info("Production model training completed successfully")
    return results


# Keep the original class for backward compatibility
class ModelTrainer:
    """Handles model training, evaluation, and MLflow tracking"""

    def __init__(self):
        self.best_model = None
        self.model_uri = None

    def train_production_model(
        self, X_train, y_train, X_test, y_test, feature_names: list = None
    ) -> dict:
        """Train the production-ready model using Prefect flow"""
        return model_training_flow(X_train, y_train, X_test, y_test, feature_names)


# Convenience function for direct usage
def train_model(X_train, y_train, X_test, y_test, feature_names: list = None) -> dict:
    """Train model using the Prefect flow"""
    return model_training_flow(X_train, y_train, X_test, y_test, feature_names)
