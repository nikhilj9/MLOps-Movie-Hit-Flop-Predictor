from prefect import flow, task
from data import load_data, preprocess_data
from model import train_model
from utils import load_config
from sklearn.model_selection import train_test_split
from mlflow.models.signature import infer_signature
import mlflow
import mlflow.sklearn


@task
def load_config_task():
    return load_config("src/config.yaml")


@task
def load_and_preprocess_data_task(path, genres, target):
    df = load_data(path)
    df = preprocess_data(df, genres, target)
    return df


@task
def split_data_task(df, genres, target):
    feature_cols = ["budget", "runtime"] + genres
    X = df[feature_cols]
    y = df[target]
    return train_test_split(X, y, test_size=0.2, random_state=42)


@task
def train_and_log_model_task(X_train, y_train, X_test, y_test, config):
    model, report, best_params = train_model(X_train, y_train, X_test, y_test)

    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    with mlflow.start_run():
        mlflow.log_params(best_params)
        mlflow.log_metrics(
            {
                "precision": report["1"]["precision"],
                "recall": report["1"]["recall"],
                "f1": report["1"]["f1-score"],
                "accuracy": report["accuracy"],
            }
        )

        input_example = X_test.head(1).fillna(0).astype("float64")
        prediction = model.predict(input_example)
        signature = infer_signature(input_example, prediction)

        mlflow.sklearn.log_model(
            sk_model=model,
            name="MovieHitFlopModel-RandomForestClassifier",
            input_example=input_example,
            signature=signature,
        )
        print("Prediction:", prediction)


@flow(name="Modular ML Training Flow")
def training_flow():
    config = load_config_task()
    df = load_and_preprocess_data_task(
        config["data"]["path"], config["model"]["genres"], config["model"]["target"]
    )
    X_train, X_test, y_train, y_test = split_data_task(
        df, config["model"]["genres"], config["model"]["target"]
    )
    train_and_log_model_task(X_train, y_train, X_test, y_test, config)


if __name__ == "__main__":
    training_flow()
