"""Data loading, cleaning, and preparation pipeline with Prefect tasks"""

import logging
from typing import Optional

import pandas as pd
from prefect import flow, task

# Fix config import
try:
    from .config import DATA_CONFIG
except ImportError:
    from config import DATA_CONFIG

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@task(name="load-and-explore-data", retries=2, retry_delay_seconds=10)
def load_and_explore_data(file_path: str) -> pd.DataFrame:
    """Load data and return basic info"""
    try:
        df = pd.read_csv(file_path)

        logger.info(f"Dataset shape: {df.shape}")
        logger.info(f"Columns: {df.columns.tolist()}")
        logger.info(f"Missing values:\n{df.isnull().sum().to_dict()}")

        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise


@task(name="clean-data", retries=1)
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and prepare the dataset"""
    logger.info(f"Before cleaning: {len(df)} rows")

    # Remove duplicates
    df = df.drop_duplicates()
    logger.info(f"After duplicate removal: {len(df)} rows")

    # Remove zero runtime and budget (data errors)
    df = df[df["runtime"] > 0]
    df = df[df["budget"] > 0]
    logger.info(f"After cleaning: {len(df)} rows")

    # Convert release_date
    df["release_date"] = pd.to_datetime(df["release_date"])

    return df.copy()


@task(name="create-target-variable", retries=1)
def create_target_variable(df: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Create ROI-based hit/flop target and return df with threshold"""
    df["roi"] = df["revenue"] / (df["budget"] + 1)
    roi_threshold = df["roi"].quantile(DATA_CONFIG["roi_threshold_quantile"])
    df["is_hit"] = (df["roi"] >= roi_threshold).astype(int)

    logger.info(f"ROI threshold: {roi_threshold:.2f}")
    logger.info(f"Hit distribution: {df['is_hit'].value_counts().to_dict()}")
    logger.info(f"Hit percentage: {df['is_hit'].mean()*100:.1f}%")

    return df, roi_threshold


@flow(name="data-processing-pipeline")
def data_processing_flow(file_path: Optional[str] = None) -> tuple[pd.DataFrame, float]:
    """Execute complete data processing pipeline as a Prefect flow"""
    if file_path is None:
        file_path = DATA_CONFIG["data_path"]

    logger.info("Starting data processing pipeline...")

    # Execute pipeline steps as tasks
    raw_df = load_and_explore_data(file_path)
    clean_df = clean_data(raw_df)
    final_df, roi_threshold = create_target_variable(clean_df)

    logger.info("Data processing pipeline completed successfully")
    return final_df, roi_threshold


# Keep the original class for backward compatibility
class DataPipeline:
    """Handles all data processing operations"""

    def __init__(self):
        self.roi_threshold = None

    def run_pipeline(self, file_path: str = None) -> pd.DataFrame:
        """Execute complete data processing pipeline using Prefect flow"""
        final_df, roi_threshold = data_processing_flow(file_path)
        self.roi_threshold = roi_threshold
        return final_df


# Convenience function for direct usage
def load_and_process_data(file_path: str = None) -> pd.DataFrame:
    """Load and process data using the Prefect flow"""
    final_df, _ = data_processing_flow(file_path)
    return final_df
