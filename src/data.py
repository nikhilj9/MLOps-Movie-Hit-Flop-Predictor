import pandas as pd
import ast
from sklearn.preprocessing import MultiLabelBinarizer

def load_data(path):
    return pd.read_csv(path)

def preprocess_data(df, genres, target_col):
    df["is_hit"] = (df["revenue"] >= 2 * df["budget"]).astype(int)
    df = df.dropna(subset=["is_hit"])
    df = df.fillna(0)
    df = df.drop(columns=["poster_path", "spoken_languages", "production_companies", "production_countries"])
    df["overview"] = df["overview"].fillna("")

    if "genres" in df.columns:
        df["genres"] = df["genres"].fillna("[]").apply(lambda x: [d['name'] for d in ast.literal_eval(x)])
        mlb = MultiLabelBinarizer()
        genre_df = pd.DataFrame(mlb.fit_transform(df["genres"]), columns=mlb.classes_, index=df.index)
        df = pd.concat([df.drop(columns=["genres"]), genre_df], axis=1)

    for genre in genres:
        if genre not in df.columns:
            df[genre] = 0

    return df