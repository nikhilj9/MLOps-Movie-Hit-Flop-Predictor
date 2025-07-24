from evidently import ColumnMapping
from evidently.report import Report
from evidently.metrics import ClassificationQualityMetric
from evidently.metric_preset import DataDriftPreset
import pandas as pd
import os

# Create output directory
os.makedirs("monitoring_reports", exist_ok=True)

# Load datasets
reference_data = pd.read_csv("data/reference_dataset.csv")
current_data = pd.read_csv("data/current_dataset.csv")

# Define column mapping
column_mapping = ColumnMapping()
column_mapping.target = "true_label"
column_mapping.prediction = "prediction"

# Define report with performance + drift
report = Report(metrics=[
    ClassificationQualityMetric(),
    DataDriftPreset()
])

report.run(reference_data=reference_data, current_data=current_data, column_mapping=column_mapping)

report.save_html("monitoring_reports/classification_and_drift_report.html")

print("Monitoring report with drift saved: monitoring_reports/classification_and_drift_report.html")