from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import src.main  # import this way so we can patch correctly

client = TestClient(src.main.app)

sample_input = {
    "budget": 10000000,
    "runtime": 120,
    "Action": 1,
    "Adventure": 0,
    "Animation": 0,
    "Comedy": 0,
    "Crime": 0,
    "Documentary": 0,
    "Drama": 1,
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


@patch("src.main.mlflow.pyfunc.load_model")
def test_predict_endpoint_returns_correct_response(mock_load_model):
    # Mock the MLflow model's predict method
    mock_model = MagicMock()
    mock_model.predict.return_value = [1]
    mock_load_model.return_value = mock_model

    response = client.post("/predict", json=sample_input)

    assert response.status_code == 200
    assert "prediction" in response.json()
    assert response.json()["prediction"] == 1
