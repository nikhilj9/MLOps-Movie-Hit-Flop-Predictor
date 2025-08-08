# test_pipeline.py
import sys

sys.path.append("src")  # Add src to path so we can import

from data_pipeline import data_processing_flow


def test_data_pipeline():
    print("Testing Prefect data pipeline...")
    try:
        # Test the flow
        result_df, threshold = data_processing_flow()
        print(
            f"✅ Success! Processed {len(result_df)} rows with ROI threshold: {threshold:.2f}"
        )
        print(f"Hit percentage: {result_df['is_hit'].mean()*100:.1f}%")
        print(f"Columns in final dataset: {result_df.columns.tolist()}")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_data_pipeline()
