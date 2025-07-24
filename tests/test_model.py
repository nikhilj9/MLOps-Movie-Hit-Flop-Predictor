from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from src.model import train_model


def test_train_model_outputs():
    # Generate synthetic binary classification data
    X, y = make_classification(
        n_samples=100, n_features=5, n_classes=2, random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Call the model training function
    model, report, best_params = train_model(X_train, y_train, X_test, y_test)

    # Check model is fitted pipeline
    assert hasattr(model, "predict"), "Model must have predict method"

    # Check report contains expected keys
    assert "accuracy" in report, "Report must include accuracy score"

    # Check best_params is a non-empty dict
    assert isinstance(best_params, dict) and len(best_params) > 0
