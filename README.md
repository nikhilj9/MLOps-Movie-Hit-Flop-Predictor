# Movie Hit Prediction MLOps Pipeline

A production-ready machine learning pipeline for predicting movie success using Random Forest classification.

## Project Structure
movie_prediction/
├── src/
│   ├── init.py           # Package initialization
│   ├── config.py             # Configuration settings
│   ├── data_pipeline.py      # Data loading and cleaning
│   ├── feature_engineering.py # Feature engineering
│   ├── model_training.py     # Model training and MLflow
│   └── main.py              # Main pipeline orchestrator
├── data/
│   └── popular_movies.csv   # Training data
├── mlflow_data/             # MLflow artifacts and database
├── requirements.txt         # Python dependencies
├── Dockerfile              # Container configuration
└── README.md               # This file

## Features

- **Modular Design**: Each component is independent and testable
- **MLflow Integration**: Experiment tracking and model registry
- **Data Validation**: Comprehensive data cleaning and validation
- **Feature Engineering**: Automated feature creation and encoding
- **Class Balancing**: SMOTE for handling imbalanced data
- **Hyperparameter Tuning**: Grid search for optimal parameters
- **Production Ready**: Containerized and deployable

## Quick Start

### Local Development

1. Install dependencies:
```bash
pip install -r requirements.txt

2. Run the training pipeline:
python -m src.main --mode train --data-path data/popular_movies.csv

Docker Deployment

1. Build the container:
docker build -t movie-prediction .

2. Run the container:
docker run -v $(pwd)/mlflow_data:/app/mlflow_data movie-prediction

Model Performance

Algorithm: Random Forest Classifier
Test Accuracy: ~77%
Test AUC: ~82%
Key Features: vote_count, budget, popularity

MLflow UI
Access the MLflow UI to view experiments:
mlflow ui --backend-store-uri sqlite:///mlflow_data/mlflow.db
