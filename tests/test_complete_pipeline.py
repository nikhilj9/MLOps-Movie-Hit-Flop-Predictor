# test_complete_pipeline.py

from src.data_pipeline import data_processing_flow
from src.feature_engineering import feature_engineering_flow
from src.model_training import model_training_flow


def test_complete_pipeline():
    print("Testing Complete Prefect Pipeline...")

    try:
        # Step 1: Data Processing
        print("\n Step 1: Data Processing...")
        df, roi_threshold = data_processing_flow()
        print(f" Data processed: {len(df)} rows, ROI threshold: {roi_threshold:.2f}")

        # Step 2: Feature Engineering
        print("\n🔧 Step 2: Feature Engineering...")
        train_test_data, encoders = feature_engineering_flow(df)
        (
            X_train,
            X_test,
            y_train,
            y_test,
            X_train_balanced,
            y_train_balanced,
        ) = train_test_data
        print(
            f" Features ready: {X_train.shape[1]} features, {len(X_train)} training samples"
        )

        # Step 3: Model Training
        print("\n Step 3: Model Training...")
        feature_names = list(X_train.columns)
        results = model_training_flow(
            X_train_balanced, y_train_balanced, X_test, y_test, feature_names
        )

        # Print Results
        print("Model trained successfully!")
        print(f"   Accuracy: {results['accuracy']:.3f}")
        print(f"   AUC: {results['auc']:.3f}")
        print(f"   Model URI: {results['model_uri']}")
        top_features = (
            list(results["evaluation"]["feature_importance"].keys())[:3]
            if results["evaluation"]["feature_importance"]
            else "N/A"
        )
        print(f"   Top 3 features: {top_features}")

        return True

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_complete_pipeline()
    if success:
        print("\n COMPLETE PIPELINE CONVERSION SUCCESSFUL!")
        print("All three components now run as Prefect flows!")
    else:
        print("\n Pipeline conversion failed!")
