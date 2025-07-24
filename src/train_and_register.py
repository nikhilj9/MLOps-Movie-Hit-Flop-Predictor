import pandas as pd
import joblib
import mlflow
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def train_and_log_model():
    # Load current data (post-drift)
    df = pd.read_csv("data/current_dataset.csv")

    # Prepare features and labels
    X = df.drop(columns=["true_label", "prediction"], errors="ignore")
    y = df["true_label"]

    # Split into train/test (optional)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train simple model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # Save model
    os.makedirs("artifacts", exist_ok=True)
    model_path = "artifacts/model.pkl"
    joblib.dump(model, model_path)

    # Start MLflow tracking
    mlflow.set_tracking_uri("http://127.0.0.1:5000")  # adjust if using remote
    mlflow.set_experiment("Movie Hit-Flop Model")

    with mlflow.start_run():
        mlflow.log_params({"model_type": "LogisticRegression", "max_iter": 1000})
        mlflow.log_metrics({"accuracy": acc})
        mlflow.log_artifact(model_path, artifact_path="model")

        # Register the model to the Model Registry
        mlflow.sklearn.log_model(
            model, artifact_path="model", registered_model_name="MovieHitFlopModel"
        )

        print(f"Model retrained and logged to MLflow with accuracy: {acc:.4f}")
