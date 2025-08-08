"""Main pipeline orchestrator for movie hit prediction with Prefect master flow"""

import argparse
import logging
from typing import Any, Dict

from prefect import flow

try:
    # When run as module: python -m src.main
    from .data_pipeline import data_processing_flow
    from .feature_engineering import feature_engineering_flow
    from .model_training import model_training_flow
except ImportError:
    # When run directly: python src/main.py
    from data_pipeline import data_processing_flow
    from feature_engineering import feature_engineering_flow
    from model_training import model_training_flow

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@flow(name="complete-movie-prediction-pipeline", retries=1, retry_delay_seconds=30)
def complete_movie_prediction_pipeline(data_path: str = None) -> Dict[str, Any]:
    """Master flow that orchestrates the complete ML pipeline"""
    logger.info("=== STARTING MOVIE HIT PREDICTION TRAINING PIPELINE ===")

    # Step 1: Data Processing Flow
    logger.info("Step 1: Data Processing")
    df_processed, roi_threshold = data_processing_flow(data_path)

    # Step 2: Feature Engineering Flow
    logger.info("Step 2: Feature Engineering")
    train_test_data, encoders = feature_engineering_flow(df_processed)
    (
        X_train,
        X_test,
        y_train,
        y_test,
        X_train_balanced,
        y_train_balanced,
    ) = train_test_data

    # Step 3: Model Training Flow
    logger.info("Step 3: Model Training")
    feature_names = X_train.columns.tolist()
    training_results = model_training_flow(
        X_train_balanced, y_train_balanced, X_test, y_test, feature_names
    )

    # Step 4: Compile Final Results
    results = {
        "data_shape": df_processed.shape,
        "roi_threshold": roi_threshold,
        "feature_count": len(feature_names),
        "train_size": X_train.shape[0],
        "test_size": X_test.shape[0],
        "balanced_train_size": X_train_balanced.shape[0],
        "model": training_results["model"],
        "model_uri": training_results["model_uri"],
        "accuracy": training_results["accuracy"],
        "auc": training_results["auc"],
        "encoders": encoders,
        "feature_names": feature_names,
        "feature_importance": training_results["evaluation"]["feature_importance"],
        "tracking_uri": training_results["tracking_uri"],
    }

    # Log summary
    _log_pipeline_summary(results)

    logger.info("=== TRAINING PIPELINE COMPLETED SUCCESSFULLY ===")
    return results


def _log_pipeline_summary(results: Dict[str, Any]):
    """Log pipeline summary"""
    logger.info(
        f"Data processed: {results['data_shape'][0]} rows, {results['data_shape'][1]} columns"
    )
    logger.info(f"ROI threshold: {results['roi_threshold']:.2f}")
    logger.info(f"Features engineered: {results['feature_count']}")
    logger.info(
        f"Training samples: {results['train_size']} → {results['balanced_train_size']} (after SMOTE)"
    )
    logger.info(
        f"Model performance - Accuracy: {results['accuracy']:.3f}, AUC: {results['auc']:.3f}"
    )
    logger.info(f"Model registered: {results['model_uri']}")

    if results["feature_importance"]:
        logger.info("Top 5 Important Features:")
        for i, (feature, importance) in enumerate(
            list(results["feature_importance"].items())[:5]
        ):
            logger.info(f"  {i+1}. {feature}: {importance:.3f}")


class MoviePredictionPipeline:
    """Complete ML pipeline for movie hit prediction - now using Prefect master flow"""

    def __init__(self):
        self.results = {}

    def run_training_pipeline(self, data_path: str = None) -> dict:
        """Execute complete training pipeline using Prefect master flow"""
        self.results = complete_movie_prediction_pipeline(data_path)
        return self.results

    def get_model_artifacts(self) -> dict:
        """Return model and preprocessing artifacts for deployment"""
        if not self.results:
            raise ValueError("Pipeline must be run first")

        return {
            "model": self.results["model"],
            "encoders": self.results["encoders"],
            "feature_names": self.results["feature_names"],
            "model_uri": self.results["model_uri"],
        }


def main():
    """Main entry point with CLI support"""
    parser = argparse.ArgumentParser(description="Movie Hit Prediction Pipeline")
    parser.add_argument(
        "--data-path", type=str, default=None, help="Path to the movie data CSV file"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train"],
        default="train",
        help="Pipeline mode (currently only train supported)",
    )
    parser.add_argument(
        "--use-prefect",
        action="store_true",
        default=True,
        help="Use Prefect master flow (default: True)",
    )

    args = parser.parse_args()

    if args.mode == "train":
        if args.use_prefect:
            # Use the new Prefect master flow directly
            results = complete_movie_prediction_pipeline(args.data_path)
        else:
            # Use the wrapper class (for backward compatibility)
            pipeline = MoviePredictionPipeline()
            results = pipeline.run_training_pipeline(args.data_path)

        print("\n" + "=" * 60)
        print("PIPELINE EXECUTION SUMMARY")
        print("=" * 60)
        print(f"Data Shape: {results['data_shape']}")
        print(f"ROI Threshold: {results['roi_threshold']:.2f}")
        print(f"Feature Count: {results['feature_count']}")
        print(
            f"Training Samples: {results['train_size']} → {results['balanced_train_size']} (SMOTE)"
        )
        print("Model Type: Random Forest")
        print(f"Test Accuracy: {results['accuracy']:.3f}")
        print(f"Test AUC: {results['auc']:.3f}")
        print(f"Model URI: {results['model_uri']}")
        print(f"MLflow URI: {results['tracking_uri']}")
        print("=" * 60)

        return results


if __name__ == "__main__":
    main()
