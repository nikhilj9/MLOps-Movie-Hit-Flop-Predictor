from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently import ColumnMapping


def check_data_drift(reference_df, current_df) -> bool:
    """Run data drift check and return True if drift is detected."""

    # Define what target/prediction columns look like (optional but safe)
    column_mapping = ColumnMapping()
    column_mapping.target = "true_label"
    column_mapping.prediction = "prediction"

    # Build drift report
    drift_report = Report(metrics=[DataDriftPreset()])
    drift_report.run(
        reference_data=reference_df,
        current_data=current_df,
        column_mapping=column_mapping,
    )

    # Parse result as Python dict
    result = drift_report.as_dict()

    # Extract high-level drift result
    drift_detected = result["metrics"][0]["result"]["dataset_drift"]

    return drift_detected
