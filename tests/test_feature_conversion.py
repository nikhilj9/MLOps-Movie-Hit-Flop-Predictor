# test_feature_conversion.py

from src.data_pipeline import data_processing_flow
from src.feature_engineering import feature_engineering_flow


def test_feature_engineering():
    print("🧪 Testing Feature Engineering Conversion...")

    try:
        # Step 1: Get processed data
        print("Step 1: Loading and processing data...")
        df, roi_threshold = data_processing_flow()
        print(f"✅ Data loaded: {len(df)} rows")

        # Step 2: Run feature engineering
        print("Step 2: Running feature engineering...")
        train_test_data, encoders = feature_engineering_flow(df)

        # Unpack the results
        (
            X_train,
            X_test,
            y_train,
            y_test,
            X_train_balanced,
            y_train_balanced,
        ) = train_test_data

        # Step 3: Print results
        print("✅ Feature engineering completed!")
        print(f"   Training set: {X_train.shape}")
        print(f"   Test set: {X_test.shape}")
        print(f"   Balanced training set: {X_train_balanced.shape}")
        print(f"   Features: {list(X_train.columns)}")
        print(f"   Encoders available: {list(encoders.keys())}")

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


if __name__ == "__main__":
    success = test_feature_engineering()
    if success:
        print("🎉 Feature engineering conversion successful!")
    else:
        print("💥 Feature engineering conversion failed!")
