import mlflow
from utils import load_config

config = load_config("src/config.yaml")
mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])

# Load model from MLflow registry by name and stage
model_name = "MovieHitFlopModel-RandomForestClassifier"
model_stage = "Production"

model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_stage}")