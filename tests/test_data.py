import pandas as pd
from src.data import preprocess_data


def test_preprocess_data_creates_is_hit_and_encodes_genres():
    # Sample input DataFrame
    df = pd.DataFrame(
        {
            "budget": [100, 200],
            "revenue": [300, 100],
            "overview": [None, "Good movie"],
            "poster_path": ["somepath", "otherpath"],
            "spoken_languages": ["en", "fr"],
            "production_companies": ["Company A", "Company B"],
            "production_countries": ["US", "UK"],
            "genres": [
                '[{"name": "Action"}, {"name": "Comedy"}]',
                '[{"name": "Drama"}]',
            ],
        }
    )

    genres = ["Action", "Comedy", "Drama", "Romance"]
    target_col = "is_hit"

    # Run preprocessing
    df_processed = preprocess_data(df, genres, target_col)

    # Check binary target
    assert "is_hit" in df_processed.columns
    assert df_processed["is_hit"].tolist() == [1, 0]

    # Check genre columns exist
    for genre in genres:
        assert genre in df_processed.columns

    # Check NaNs are handled
    assert df_processed["overview"].isnull().sum() == 0
    assert not df_processed.isnull().values.any()
