"""FastAPI inference service for movie hit prediction"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, Optional

import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Fix imports
try:
    from .config import API_CONFIG, FEATURE_CONFIG, INFERENCE_CONFIG
    from .model_loader import model_loader
except ImportError:
    from config import API_CONFIG, FEATURE_CONFIG, INFERENCE_CONFIG
    from model_loader import model_loader

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title=API_CONFIG["title"],
    description=API_CONFIG["description"],
    version=API_CONFIG["version"],
    docs_url=API_CONFIG["docs_url"],
    redoc_url=API_CONFIG["redoc_url"],
)


# Pydantic models for request/response
class MovieInput(BaseModel):
    """Input schema for movie prediction"""

    budget: float = Field(..., description="Movie budget in USD", example=50000000)
    runtime: float = Field(..., description="Movie runtime in minutes", example=120)
    vote_average: float = Field(..., description="Average rating (0-10)", example=7.5)
    vote_count: int = Field(..., description="Number of votes", example=1500)
    popularity: float = Field(..., description="Popularity score", example=25.0)
    genres: str = Field(
        ..., description="Genres in JSON format", example="[{'name': 'Action'}]"
    )
    original_language: str = Field(
        ..., description="Original language code", example="en"
    )
    release_date: str = Field(
        ..., description="Release date (YYYY-MM-DD)", example="2023-01-15"
    )
    revenue: Optional[float] = Field(
        None, description="Revenue for ROI calculation", example=100000000
    )


class PredictionResponse(BaseModel):
    """Response schema for prediction"""

    prediction: int = Field(..., description="0 for flop, 1 for hit")
    probability: float = Field(..., description="Probability of being a hit (0-1)")
    confidence: str = Field(..., description="Confidence level")
    model_version: str = Field(..., description="Model version used")
    processed_at: str = Field(..., description="Processing timestamp")


class HealthResponse(BaseModel):
    """Health check response"""

    status: str
    model_loaded: bool
    model_info: Dict[str, Any]
    timestamp: str


# Feature engineering functions (from your pipeline)
def extract_genres(genres_str: str) -> tuple:
    """Extract genre count and main genre from JSON string"""
    try:
        genres = json.loads(genres_str.replace("'", '"'))
        return len(genres), genres[0]["name"] if genres else "Unknown"
    except Exception:
        return 0, "Unknown"


def preprocess_movie_data(movie_data: MovieInput) -> pd.DataFrame:
    """Preprocess raw movie data into model features"""
    try:
        # Create DataFrame from input
        data = {
            "budget": [movie_data.budget],
            "runtime": [movie_data.runtime],
            "vote_average": [movie_data.vote_average],
            "vote_count": [movie_data.vote_count],
            "popularity": [movie_data.popularity],
            "genres": [movie_data.genres],
            "original_language": [movie_data.original_language],
            "release_date": [movie_data.release_date],
        }
        df = pd.DataFrame(data)

        # Convert release_date
        df["release_date"] = pd.to_datetime(df["release_date"])
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

        # Handle missing values
        df["release_year"].fillna(df["release_year"].median(), inplace=True)
        df["budget_category"].fillna("Ultra_Low", inplace=True)

        # Select and order features
        feature_columns = (
            FEATURE_CONFIG["numeric_features"] + FEATURE_CONFIG["categorical_features"]
        )
        X = df[feature_columns].copy()

        # Encode categorical variables
        try:
            X["budget_category"] = model_loader.encoders["budget"].transform(
                X["budget_category"]
            )
        except ValueError:
            # Handle unknown categories
            X["budget_category"] = 0  # Default to first category

        try:
            X["main_genre"] = model_loader.encoders["genre"].transform(X["main_genre"])
        except ValueError:
            # Handle unknown genres
            X["main_genre"] = model_loader.encoders["genre"].transform(["Unknown"])[0]

        return X

    except Exception as e:
        logger.error(f"Preprocessing error: {e}")
        raise HTTPException(
            status_code=400, detail=f"Data preprocessing failed: {str(e)}"
        )


# API Endpoints
@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    logger.info("Starting inference service...")
    if not model_loader.load_latest_model():
        logger.error("Failed to load model on startup")
    else:
        logger.info("Model loaded successfully on startup")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if model_loader.is_model_loaded() else "unhealthy",
        model_loaded=model_loader.is_model_loaded(),
        model_info=model_loader.get_model_info(),
        timestamp=datetime.now().isoformat(),
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_movie_hit(movie: MovieInput):
    """Predict if movie will be a hit or flop"""
    if not model_loader.is_model_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Preprocess input data
        X = preprocess_movie_data(movie)

        # Make prediction
        prediction = model_loader.model.predict(X)[0]
        probability = model_loader.model.predict_proba(X)[0][1]  # Probability of hit

        # Determine confidence level
        if probability > 0.8 or probability < 0.2:
            confidence = "high"
        elif probability > 0.6 or probability < 0.4:
            confidence = "medium"
        else:
            confidence = "low"

        return PredictionResponse(
            prediction=int(prediction),
            probability=float(probability),
            confidence=confidence,
            model_version=str(model_loader.model_version)
            if model_loader.model_version
            else "unknown",
            processed_at=datetime.now().isoformat(),
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/model/info")
async def get_model_info():
    """Get information about the loaded model"""
    return model_loader.get_model_info()


@app.post("/model/reload")
async def reload_model():
    """Reload the latest model"""
    if model_loader.load_latest_model():
        return {"status": "success", "message": "Model reloaded successfully"}
    else:
        raise HTTPException(status_code=500, detail="Failed to reload model")


if __name__ == "__main__":
    uvicorn.run(
        app,
        host=INFERENCE_CONFIG["service_host"],
        port=INFERENCE_CONFIG["service_port"],
        log_level="info",
    )
