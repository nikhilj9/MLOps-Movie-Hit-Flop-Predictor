from prefect import flow, task
from prefect.task_runners import SequentialTaskRunner
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently import ColumnMapping
import pandas as pd
from src.train_and_register import train_and_log_model


@task
def load_data():
    reference = pd.read_csv("data/reference_dataset.csv")
    current = pd.read_csv("data/current_dataset.csv")
    return reference, current


@task
def check_data_drift(reference_df, current_df) -> bool:
    column_mapping = ColumnMapping()
    column_mapping.target = "true_label"
    column_mapping.prediction = "prediction"

    report = Report(metrics=[DataDriftPreset()])
    report.run(
        reference_data=reference_df,
        current_data=current_df,
        column_mapping=column_mapping,
    )
    result = report.as_dict()
    return result["metrics"][0]["result"]["dataset_drift"]


@task
def retrain_model():
    print("Drift detected. Retraining model...")
    train_and_log_model()  # this function handles model training + MLflow logging


@flow(
    name="Drift-Triggered Retraining Flow",
    task_runner=SequentialTaskRunner(),
    log_prints=True,
)
def monitor_and_retrain():
    ref, cur = load_data()
    drift = check_data_drift(ref, cur)

    if drift:
        retrain_model()
    else:
        print("No drift. No retraining needed.")
