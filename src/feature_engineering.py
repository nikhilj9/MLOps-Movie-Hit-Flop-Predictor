"""Feature engineering and preprocessing pipeline with Prefect tasks"""

import json
import logging
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from prefect import flow, task
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Fix config import
try:
    from .config import DATA_CONFIG, FEATURE_CONFIG
except ImportError:
    from config import DATA_CONFIG, FEATURE_CONFIG

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_genres(genres_str: str) -> tuple:
    """Extract genre count and main genre from JSON string"""
    try:
        genres = json.loads(genres_str.replace("'", '"'))
        return len(genres), genres[0]["name"] if genres else "Unknown"
    except Exception:
        return 0, "Unknown"


@task(name="engineer-features", retries=1)
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create all engineered features"""
    df = df.copy()

    # Time-based features
    df["release_year"] = df["release_date"].dt.year

    # Budget categories
    df["budget_category"] = pd.cut(
        df["budget"],
        bins=FEATURE_CONFIG["budget_bins"],
        labels=FEATURE_CONFIG["budget_labels"],
    )

    # Genre features
    df[["genre_count", "main_genre"]] = df["genres"].apply(
        lambda x: pd.Series(extract_genres(x))
    )

    # Language feature
    df["is_english"] = (df["original_language"] == "en").astype(int)

    logger.info("Features engineered successfully")
    logger.info(
        f"Release year range: {df['release_year'].min()}-{df['release_year'].max()}"
    )
    logger.info(f"Budget categories: {df['budget_category'].value_counts().to_dict()}")
    logger.info(f"English movies: {df['is_english'].mean()*100:.1f}%")

    return df


@task(name="prepare-features", retries=1)
def prepare_features(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
    """Select and prepare features for modeling"""
    df = df.copy()
    numeric_features = FEATURE_CONFIG["numeric_features"]
    categorical_features = FEATURE_CONFIG["categorical_features"]

    # Handle missing values
    df["release_year"].fillna(df["release_year"].median(), inplace=True)
    df["budget_category"].fillna("Ultra_Low", inplace=True)

    # Prepare feature matrix
    X = df[numeric_features + categorical_features].copy()

    # Create and fit encoders
    le_budget = LabelEncoder()
    le_genre = LabelEncoder()

    X["budget_category"] = le_budget.fit_transform(X["budget_category"])
    X["main_genre"] = le_genre.fit_transform(X["main_genre"])

    # Store encoders
    encoders = {"budget": le_budget, "genre": le_genre}

    y = df["is_hit"] if "is_hit" in df.columns else None

    logger.info(f"Final feature matrix: {X.shape}")
    logger.info(f"Features: {X.columns.tolist()}")

    return X, y, encoders


@task(name="train-test-split", retries=1)
def prepare_train_test_split(
    X: pd.DataFrame, y: pd.Series
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, pd.Series]:
    """Split data and handle class imbalance"""
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=DATA_CONFIG["test_size"],
        random_state=DATA_CONFIG["random_state"],
        stratify=y,
    )

    # Apply SMOTE for class balancing
    smote = SMOTE(random_state=DATA_CONFIG["random_state"])
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

    logger.info(f"Training set: {X_train.shape}")
    logger.info(f"Test set: {X_test.shape}")
    logger.info(f"Original training: {np.bincount(y_train)}")
    logger.info(f"Balanced training: {np.bincount(y_train_balanced)}")

    return X_train, X_test, y_train, y_test, X_train_balanced, y_train_balanced


@flow(name="feature-engineering-pipeline")
def feature_engineering_flow(
    df: pd.DataFrame,
) -> Tuple[
    Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, pd.Series],
    Dict[str, Any],
]:
    """Execute complete feature engineering pipeline as a Prefect flow"""
    logger.info("Starting feature engineering pipeline...")

    # Execute pipeline steps as tasks
    df_featured = engineer_features(df)
    X, y, encoders = prepare_features(df_featured)
    train_test_data = prepare_train_test_split(X, y)

    logger.info("Feature engineering pipeline completed successfully")
    return train_test_data, encoders


# Keep the original class for backward compatibility
class FeatureEngineer:
    """Handles all feature engineering operations"""

    def __init__(self):
        self.le_budget = LabelEncoder()
        self.le_genre = LabelEncoder()
        self.is_fitted = False

    def run_feature_pipeline(self, df: pd.DataFrame) -> tuple:
        """Execute complete feature engineering pipeline using Prefect flow"""
        train_test_data, encoders = feature_engineering_flow(df)

        # Store encoders for backward compatibility
        self.le_budget = encoders["budget"]
        self.le_genre = encoders["genre"]
        self.is_fitted = True

        return train_test_data, encoders

    def get_encoders(self) -> dict:
        """Return fitted encoders"""
        return {"budget": self.le_budget, "genre": self.le_genre}


# Convenience function for direct usage
def process_features(df: pd.DataFrame) -> tuple:
    """Process features using the Prefect flow"""
    return feature_engineering_flow(df)