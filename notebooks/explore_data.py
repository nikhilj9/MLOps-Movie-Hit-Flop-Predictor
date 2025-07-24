import pandas as pd
import ast
import mlflow
import mlflow.sklearn

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

# MLflow config
mlflow.set_tracking_uri(
    "file:///workspaces/MLOps-Movie-Hit-Flop-Predictor/notebooks/mlruns"
)
mlflow.set_experiment("movie-hit-flop-exp")


# Data loading
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


# Data preprocessing
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df["is_hit"] = (df["revenue"] >= 2 * df["budget"]).astype(int)
    df = df.dropna(subset=["is_hit"])
    df = df.drop(
        columns=[
            "poster_path",
            "spoken_languages",
            "production_companies",
            "production_countries",
        ]
    )
    df["overview"] = df["overview"].fillna("")
    df["genres"] = (
        df["genres"]
        .fillna("[]")
        .apply(lambda x: [d["name"] for d in ast.literal_eval(x)])
    )

    mlb = MultiLabelBinarizer()
    genre_df = pd.DataFrame(
        mlb.fit_transform(df["genres"]), columns=mlb.classes_, index=df.index
    )
    df = pd.concat([df.drop(columns=["genres"]), genre_df], axis=1)

    return df


# Training + Evaluation
def train_model(df: pd.DataFrame, genre_cols):
    feature_cols = ["budget", "runtime"] + [
        col for col in df.columns if col in genre_cols
    ]
    target_col = "is_hit"

    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", RandomForestClassifier(random_state=42)),
        ]
    )

    param_grid = {
        "model__n_estimators": [50, 100, 200],
        "model__max_depth": [None, 10, 20],
        "model__min_samples_split": [2, 5],
        "model__class_weight": ["balanced"],
    }

    grid_search = GridSearchCV(
        pipe, param_grid, cv=5, scoring="f1", verbose=1, n_jobs=-1
    )
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)

    return best_model, report, grid_search.best_params_


# Main execution
def main():
    mlflow.set_tracking_uri(
        "file:///workspaces/MLOps-Movie-Hit-Flop-Predictor/notebooks/mlruns"
    )
    mlflow.set_experiment("movie-hit-flop-exp")

    with mlflow.start_run():
        df = load_data("data/popular_movies.csv")

        genre_cols = [
            "Action",
            "Adventure",
            "Animation",
            "Comedy",
            "Crime",
            "Documentary",
            "Drama",
            "Family",
            "Fantasy",
            "History",
            "Horror",
            "Music",
            "Mystery",
            "Romance",
            "Science Fiction",
            "TV Movie",
            "Thriller",
            "War",
            "Western",
        ]

        df = preprocess_data(df)
        model, report, best_params = train_model(df, genre_cols)

        mlflow.log_metrics(
            {
                "precision": report["1"]["precision"],
                "recall": report["1"]["recall"],
                "f1": report["1"]["f1-score"],
                "accuracy": report["accuracy"],
            }
        )

        feature_cols = ["budget", "runtime"] + [
            col for col in df.columns if col in genre_cols
        ]

        input_example = df[feature_cols].iloc[[0]]
        signature = mlflow.models.infer_signature(
            input_example, model.predict(input_example)
        )

        mlflow.log_params(best_params)
        mlflow.sklearn.log_model(
            model, name="model", input_example=input_example, signature=signature
        )

        print(input_example)
        print("Prediction ", model.predict(input_example))


if __name__ == "__main__":
    main()
