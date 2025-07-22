import mlflow
import mlflow.sklearn
import pandas as pd
from utils import load_config

config = load_config("src/config.yaml")
# Set the correct MLflow tracking URI
mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])

# Load model from Model Registry
model_name = "MovieHitFlopModel-RandomForestClassifier"
stage = "Production"  # or "Staging"
model = mlflow.sklearn.load_model(f"models:/{model_name}/{stage}")

# Sample input
sample_input = pd.DataFrame([{
    "budget": 90000000,
    "runtime": 125,
    "Action": 1,
    "Adventure": 0,
    "Animation": 0,
    "Comedy": 0,
    "Crime": 1,
    "Documentary": 0,
    "Drama": 0,
    "Family": 0,
    "Fantasy": 0,
    "History": 0,
    "Horror": 0,
    "Music": 0,
    "Mystery": 0,
    "Romance": 0,
    "Science Fiction": 0,
    "TV Movie": 0,
    "Thriller": 1,
    "War": 0,
    "Western": 0
}])

# Predict
prediction = model.predict(sample_input)
print("Prediction:", prediction)