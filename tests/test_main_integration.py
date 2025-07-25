from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from src.main import app


@patch("src.main.load_model")
def test_predict_endpoint(mock_load_model):
    mock_model = MagicMock()
    mock_model.predict.return_value = [1]
    mock_load_model.return_value = mock_model

    client = TestClient(app)

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

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    assert response.json() == {"prediction": 1}


def test_root_endpoint():
    client = TestClient(app)
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {"message": "Model is ready for prediction!"}
