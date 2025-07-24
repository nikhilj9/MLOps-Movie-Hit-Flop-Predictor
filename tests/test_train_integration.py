from unittest.mock import patch, MagicMock
from src.train import train_and_log


@patch("src.train.mlflow.tracking.MlflowClient")
@patch("src.train.mlflow.tracking.fluent._get_or_start_run")
@patch("src.train.MlflowClient")
@patch("src.train.mlflow.set_tracking_uri")
@patch("src.train.mlflow.set_experiment")
@patch("src.train.mlflow.start_run")
@patch("src.train.mlflow.log_params")
@patch("src.train.mlflow.log_metrics")
@patch("src.train.mlflow.sklearn.log_model")
@patch("src.train.mlflow.register_model")
@patch("src.train.mlflow.active_run")
def test_train_and_log_pipeline_runs(
    mock_active_run,
    mock_register_model,
    mock_log_model,
    mock_log_metrics,
    mock_log_params,
    mock_start_run,
    mock_set_experiment,
    mock_set_tracking_uri,
    mock_mlflow_client,
    mock_get_or_start_run,
    mock_tracking_client,
):
    # Mock context manager
    mock_start_run.return_value.__enter__ = MagicMock()
    mock_start_run.return_value.__exit__ = MagicMock()

    # Mock run info
    mock_run_info = MagicMock()
    mock_run_info.info.run_id = "test-run-id"
    mock_active_run.return_value = mock_run_info
    mock_get_or_start_run.return_value = mock_run_info

    # Mock register result
    mock_result = MagicMock()
    mock_result.version = "1"
    mock_register_model.return_value = mock_result

    train_and_log()
