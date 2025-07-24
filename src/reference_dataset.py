import pandas as pd

def reference_dataset(X_test, y_test, best_model):

    columns = [
        "budget",
        "runtime",
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
        "Western"
    ]

    if not isinstance(X_test, pd.DataFrame):
        X_test = pd.DataFrame(X_test, columns=columns)  # add your feature names here

    # Select only the first 10 rows
    X_test_sample = X_test.head(10)
    y_test_sample = y_test[:10]

    # Predict on these 10 samples
    y_pred_sample = best_model.predict(X_test_sample)

    # Create a DataFrame combining features + true labels + predictions
    reference_df_sample = X_test_sample.copy()
    reference_df_sample['true_label'] = y_test_sample.values if hasattr(y_test_sample, 'values') else y_test_sample
    reference_df_sample['predicted'] = y_pred_sample

    # Save to CSV
    reference_df_sample.to_csv('data/reference_dataset.csv', index=False)