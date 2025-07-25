from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)


@patch("src.main.mlflow.pyfunc.load_model")
def test_predict_endpoint(mock_load_model):
    # Mock the MLflow model object
    mock_model = MagicMock()
    mock_model.predict.return_value = ["Hit"]
    mock_load_model.return_value = mock_model

    # Sample request payload (adjust to match your API schema)
    payload = {"budget": 10000000.0, "runtime": 120.0, "genres": ["Action", "Thriller"]}

    # Make POST request to FastAPI /predict endpoint
    response = client.post("/predict", json=payload)

    # Validate response
    assert response.status_code == 200
    assert response.json() == {"prediction": "Hit"}
