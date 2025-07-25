from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)


@patch("src.main.mlflow.pyfunc.load_model")
def test_predict_endpoint(mock_load_model):
    # Mock the MLflow model object
    mock_model = MagicMock()
    mock_model.predict.return_value = [1]  # Assuming 1 means "Hit"
    mock_load_model.return_value = mock_model

    # ✅ Payload that matches the MovieFeatures Pydantic model
    payload = {
        "budget": 10000000.0,
        "runtime": 120.0,
        "Action": 1,
        "Adventure": 0,
        "Animation": 0,
        "Comedy": 0,
        "Crime": 0,
        "Documentary": 0,
        "Drama": 0,
        "Family": 0,
        "Fantasy": 0,
        "History": 0,
        "Horror": 0,
        "Music": 0,
        "Mystery": 0,
        "Romance": 0,
        "Science_Fiction": 0,
        "TV_Movie": 0,
        "Thriller": 1,
        "War": 0,
        "Western": 0,
    }

    # Make POST request to FastAPI /predict endpoint
    response = client.post("/predict", json=payload)

    # Validate response
    assert response.status_code == 200
    assert response.json() == {"prediction": 1}
