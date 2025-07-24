import mlflow
import mlflow.sklearn
import pandas as pd
from utils import load_config

# Load configuration
config = load_config("src/config.yaml")
mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])

# Load model from MLflow Model Registry
model_name = "MovieHitFlopModel-RandomForestClassifier"
stage = "Production"
model = mlflow.sklearn.load_model(f"models:/{model_name}/{stage}")

# Define sample input with 10 upcoming movie entries
sample_input_2025 = pd.DataFrame({
    "budget": [225000000, 400000000, 90000000, 150000000, 45000000,
               25000000, 100000000, 50000000, 14000000, 35000000],
    "runtime": [129, 169, 124, 98, 94, 103, 108, 110, 97, 125],
    "Action":          [1, 1, 1, 0, 1, 0, 0, 0, 0, 1],
    "Adventure":       [0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
    "Animation":       [0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
    "Comedy":          [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    "Crime":           [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    "Documentary":     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "Drama":           [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    "Family":          [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    "Fantasy":         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "History":         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "Horror":          [0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
    "Music":           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "Mystery":         [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    "Romance":         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "Science Fiction": [1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    "TV Movie":        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "Thriller":        [0, 0, 1, 0, 0, 1, 0, 1, 0, 0],
    "War":             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "Western":         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
})

# Predict for all records
predictions = model.predict(sample_input_2025)

# Print each prediction alongside the movie index
for idx, pred in enumerate(predictions, start=1):
    print(f"Movie {idx} prediction: {'Hit' if pred == 1 else 'Flop'}")