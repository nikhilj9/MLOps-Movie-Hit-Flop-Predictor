"""Configuration settings for movie hit prediction pipeline"""

import os
from pathlib import Path

# Environment detection
ENVIRONMENT = os.getenv("ENVIRONMENT", "local")  # local, cloud
IS_CLOUD = ENVIRONMENT == "cloud"

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MLFLOW_DIR = PROJECT_ROOT / "mlflow_data"
ARTIFACTS_DIR = MLFLOW_DIR / "artifacts"

# Data configuration
DATA_CONFIG = {
    "data_path": str(DATA_DIR / "popular_movies.csv"),
    "test_size": 0.2,
    "random_state": 42,
    "roi_threshold_quantile": 0.7,
}

# Feature engineering configuration
FEATURE_CONFIG = {
    "budget_bins": [0, 1000000, 50000000, 150000000, float("inf")],
    "budget_labels": ["Ultra_Low", "Low", "Medium", "High"],
    "numeric_features": [
        "budget",
        "runtime",
        "vote_average",
        "vote_count",
        "popularity",
        "genre_count",
        "release_year",
    ],
    "categorical_features": ["budget_category", "main_genre", "is_english"],
}

# Model configuration
MODEL_CONFIG = {
    "best_model": "random_forest",
    "cv_folds": 3,
    "param_grid": {
        "n_estimators": [100, 200],
        "max_depth": [10, 20, None],
        "min_samples_split": [2, 5],
    },
}

# Cloud-aware MLflow configuration
if IS_CLOUD:
    # Cloud configuration - PostgreSQL RDS + S3
    DB_HOST = os.getenv("DB_HOST", "localhost")
    DB_NAME = os.getenv("DB_NAME", "mlflow")
    DB_USER = os.getenv("DB_USER", "mlflow")
    DB_PASSWORD = os.getenv("DB_PASSWORD", "")
    S3_BUCKET = os.getenv("S3_BUCKET", "movie-prediction-artifacts")

    MLFLOW_CONFIG = {
        "tracking_uri": f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:5432/{DB_NAME}",
        "experiment_name": "movie-hit-prediction",
        "model_registry_name": "movie_hit_predictor",
        "artifacts_path": f"s3://{S3_BUCKET}/artifacts",
    }
else:
    # Local configuration - SQLite
    MLFLOW_CONFIG = {
        "tracking_uri": f"sqlite:///{MLFLOW_DIR}/mlflow.db",
        "experiment_name": "movie-hit-prediction",
        "model_registry_name": "movie_hit_predictor",
        "artifacts_path": str(ARTIFACTS_DIR),
    }

# Create directories (only for local)
if not IS_CLOUD:
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MLFLOW_DIR, exist_ok=True)
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# Cloud-aware Prefect configuration
PREFECT_CONFIG = {
    # API URL changes based on environment
    "api_url": os.getenv("PREFECT_API_URL", "http://localhost:4200/api"),
    "server_host": "0.0.0.0",
    "server_port": 4200,
    # Flow deployment settings
    "deployment_name": "movie-prediction-training",
    "work_pool_name": "default-agent-pool",
    "storage_path": "/app/flows",
    # Scheduling (cron format)
    "training_schedule": "0 2 * * 1",  # Every Monday at 2 AM
    # Flow settings
    "flow_retries": 2,
    "task_retries": 1,
    "retry_delay_seconds": 30,
}

# Inference service configuration
INFERENCE_CONFIG = {
    "service_host": "0.0.0.0",
    "service_port": 8000,
    "model_version": "latest",
    "model_name": MLFLOW_CONFIG["model_registry_name"],
    "health_check_interval": 30,
    "batch_size_limit": 100,
    "timeout_seconds": 30,
}

# API configuration
API_CONFIG = {
    "title": "Movie Hit Prediction API",
    "description": "Predict if a movie will be a hit or flop based on movie features",
    "version": "1.0.0",
    "docs_url": "/docs",
    "redoc_url": "/redoc",
}

# Monitoring configuration
MONITORING_CONFIG = {
    "service_host": "0.0.0.0",
    "service_port": 9000,
    "check_interval_minutes": 15,
    # Drift detection thresholds
    "data_drift_threshold": 0.3,
    "model_performance_threshold": 0.1,
    "prediction_confidence_threshold": 0.6,
    # AWS settings
    "aws_region": os.getenv("AWS_REGION", "us-east-1"),
    "cloudwatch_namespace": "MoviePrediction/Monitoring",
    "sns_topic_name": "movie-prediction-alerts",
    # Reference data settings
    "reference_data_size": 1000,
    "monitoring_window_size": 100,
    # Alert settings
    "alert_cooldown_minutes": 60,
    "max_alerts_per_day": 10,
    # Storage paths (cloud-aware)
    "monitoring_data_dir": "/app/monitoring_data"
    if IS_CLOUD
    else str(PROJECT_ROOT / "monitoring_data"),
    "reports_dir": "/app/monitoring_reports"
    if IS_CLOUD
    else str(PROJECT_ROOT / "monitoring_reports"),
}

# Enhanced AWS configuration for cloud deployment
AWS_CONFIG = {
    "use_aws": os.getenv("USE_AWS_MONITORING", "true" if IS_CLOUD else "false").lower()
    == "true",
    "reviewer_email": os.getenv("REVIEWER_EMAIL", ""),
    "aws_access_key_id": os.getenv("AWS_ACCESS_KEY_ID", ""),
    "aws_secret_access_key": os.getenv("AWS_SECRET_ACCESS_KEY", ""),
    "test_mode": os.getenv(
        "MONITORING_TEST_MODE", "false" if IS_CLOUD else "true"
    ).lower()
    == "true",
    "region": os.getenv("AWS_REGION", "us-east-1"),
}

# Cloud-specific configurations
CLOUD_CONFIG = {
    "vpc_cidr": "10.0.0.0/16",
    "availability_zones": ["us-east-1a", "us-east-1b"],
    "instance_type": "t2.micro",  # Free tier
    "db_instance_class": "db.t2.micro",  # Free tier
    "min_instances": 1,
    "max_instances": 2,
    "ecr_repository": os.getenv("ECR_REPOSITORY", ""),
    "domain_name": os.getenv("DOMAIN_NAME", ""),
    "ssl_certificate_arn": os.getenv("SSL_CERTIFICATE_ARN", ""),
}

# Create monitoring directories (only for local)
if not IS_CLOUD:
    os.makedirs(MONITORING_CONFIG["monitoring_data_dir"], exist_ok=True)
    os.makedirs(MONITORING_CONFIG["reports_dir"], exist_ok=True)
