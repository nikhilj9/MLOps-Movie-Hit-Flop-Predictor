import yaml
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def load_config(path):
    with open(path, "r") as file:
        return yaml.safe_load(file)


def evaluate_model(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
