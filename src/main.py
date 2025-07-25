import traceback
import pandas as pd
import logging
import mlflow
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.utils import load_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load config and set MLflow tracking URI
config = load_config("src/config.yaml")
mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])

# Model registry info
model_name = "MovieHitFlopModel-RandomForestClassifier"
model_stage = "Production"

# Placeholder for the loaded model
model = None


def load_model():
    global model
    if model is None:
        logger.info(f"Loading model: {model_name} [{model_stage}]")
        model = mlflow.pyfunc.load_model(
            model_uri=f"models:/{model_name}/{model_stage}"
        )
    return model


class MovieFeatures(BaseModel):
    budget: float
    runtime: float
    Action: int
    Adventure: int
    Animation: int
    Comedy: int
    Crime: int
    Documentary: int
    Drama: int
    Family: int
    Fantasy: int
    History: int
    Horror: int
    Music: int
    Mystery: int
    Romance: int
    Science_Fiction: int
    TV_Movie: int
    Thriller: int
    War: int
    Western: int


app = FastAPI()


@app.get("/")
def read_root():
    return {"message": "Model is ready for prediction!"}


@app.post("/predict")
def predict(features: MovieFeatures):
    try:
        input_dict = features.dict()
        logger.info(f"Received input: {input_dict}")

        df = pd.DataFrame([input_dict])
        df.rename(
            columns={"Science_Fiction": "Science Fiction", "TV_Movie": "TV Movie"},
            inplace=True,
        )
        df = df.astype("float64")

        model_instance = load_model()
        prediction = model_instance.predict(df)[0]
        logger.info(f"Prediction: {prediction}")

        return {"prediction": int(prediction)}

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Prediction error. Check logs.")
