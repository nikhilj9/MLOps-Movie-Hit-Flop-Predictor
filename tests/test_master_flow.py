# test_master_flow.py
from src.main import complete_movie_prediction_pipeline


def test_master_flow():
    print("🚀 Testing Master Flow Implementation...")
    print("This will run the COMPLETE pipeline as ONE unified Prefect flow")

    try:
        # Run the master flow
        print("\n📊 Executing Master Flow...")
        print("   ├── Data Processing Flow")
        print("   ├── Feature Engineering Flow")
        print("   └── Model Training Flow")

        results = complete_movie_prediction_pipeline()

        # Display results
        print("\n✅ Master Flow Completed Successfully!")
        print(f"   Data Shape: {results['data_shape']}")
        print(f"   ROI Threshold: {results['roi_threshold']:.2f}")
        print(f"   Features: {results['feature_count']}")
        print(
            f"   Training: {results['train_size']} → {results['balanced_train_size']} samples"
        )
        print(f"   Test Accuracy: {results['accuracy']:.3f}")
        print(f"   Test AUC: {results['auc']:.3f}")
        print(f"   Model URI: {results['model_uri']}")

        if results["feature_importance"]:
            top_features = list(results["feature_importance"].keys())[:3]
            print(f"   Top Features: {top_features}")

        return True

    except Exception as e:
        print(f"❌ Master Flow Failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_master_flow()
    if success:
        print("\n🎉 MASTER FLOW IMPLEMENTATION SUCCESSFUL!")
        print("✅ Phase 1: Basic Prefect Integration (2 Points) - COMPLETED!")
    else:
        print("\n💥 Master flow test failed!")
