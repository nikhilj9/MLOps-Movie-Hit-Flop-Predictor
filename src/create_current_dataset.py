import pandas as pd
import joblib

# 1. Load your trained model
model = joblib.load("artifacts/model.pkl")

# 2. Create new test inputs (simulate real-world data)
data = {
    "budget": [
        225000000,
        400000000,
        90000000,
        150000000,
        45000000,
        25000000,
        100000000,
        50000000,
        14000000,
        35000000,
    ],
    "runtime": [129, 169, 124, 98, 94, 103, 108, 110, 97, 125],
    "Action": [1, 1, 1, 0, 1, 0, 0, 0, 0, 1],
    "Adventure": [0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
    "Animation": [0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
    "Comedy": [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    "Crime": [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    "Documentary": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "Drama": [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    "Family": [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    "Fantasy": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "History": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "Horror": [0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
    "Music": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "Mystery": [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    "Romance": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "Science Fiction": [1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    "TV Movie": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "Thriller": [0, 0, 1, 0, 0, 1, 0, 1, 0, 0],
    "War": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "Western": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
}

df_current = pd.DataFrame(data)

# 3. Make predictions
df_current["prediction"] = model.predict(df_current)

# 4. Add true labels (manually if needed)
df_current["true_label"] = [
    1,
    0,
    1,
    0,
    0,
    0,
    1,
    0,
    1,
    1,
]  # ← You decide this (assume actual outcomes)

# 5. Save to CSV
df_current.to_csv("data/current.csv", index=False)
