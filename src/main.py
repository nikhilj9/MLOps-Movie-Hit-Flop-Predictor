import pandas as pd
import logging
import mlflow
import joblib
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from fastapi import HTTPException
from utils import load_config

logging.basicConfig(level=logging.INFO)
config = load_config("src/config.yaml")
mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])

model_name = "MovieHitFlopModel-RandomForestClassifier"
model_stage = "Production"

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

#model = joblib.load("artifacts/model.pkl")
model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_stage}")

@app.get("/")
def read_root():
    return {"message": "Model is ready for prediction!"}

@app.post("/predict")
def predict(features: MovieFeatures):
    try:
        input_dict = features.model_dump()
        logging.info(f"Received input: {input_dict}")
        
        df = pd.DataFrame([input_dict])
        df.rename(columns={
            "Science_Fiction": "Science Fiction",
            "TV_Movie": "TV Movie"
        }, inplace=True)

        prediction = model.predict(df)[0]
        logging.info(f"Prediction: {prediction}")

        return {"prediction": int(prediction)}

    except Exception as e:
        logging.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail="Prediction error. Check logs.")
