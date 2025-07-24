import mlflow
import localmodel as dm
from sklearn.model_selection import train_test_split
from mlflow.models.signature import infer_signature

from data import load_data, preprocess_data
from model import train_model
from utils import load_config, evaluate_model
from mlflow import MlflowClient

def train_and_log():
    config = load_config("src/config.yaml")
    df = load_data(config["data"]["path"])
    df = preprocess_data(df, config["model"]["genres"], config["model"]["target"])

    feature_cols = ["budget", "runtime"] + config["model"]["genres"]
    X = df[feature_cols]
    y = df[config["model"]["target"]]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model, report, best_params = train_model(X_train, y_train, X_test, y_test)

    dm.save_model(model)

    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    with mlflow.start_run():
        mlflow.log_params(best_params)
        mlflow.log_metrics({
            "precision": report["1"]["precision"],
            "recall": report["1"]["recall"],
            "f1": report["1"]["f1-score"],
            "accuracy": report["accuracy"]
        })

        input_example = X_test.head(1).fillna(0).astype("float64")
        prediction = model.predict(input_example)
        signature = infer_signature(input_example, prediction)

        mlflow.sklearn.log_model(sk_model=model, artifact_path="model", input_example=input_example, signature=signature)

        print("Logged input example:")
        print(input_example)
        print("Prediction:", prediction)

        run_id = mlflow.active_run().info.run_id
        model_uri = f"runs:/{run_id}/model"
        model_name = "MovieHitFlopModel-RandomForestClassifier"

        client = MlflowClient()
        try:
            client.create_registered_model(model_name)
        except Exception:
            pass

        result = mlflow.register_model(model_uri=model_uri, name=model_name)
        print(f"Registered model version {result.version}")

