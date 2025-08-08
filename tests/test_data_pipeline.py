import sys

sys.path.append("src")

from main import MoviePredictionPipeline

# Test the complete pipeline
pipeline = MoviePredictionPipeline()
results = pipeline.run_training_pipeline("data/popular_movies.csv")

print("\n=== FINAL TEST RESULTS ===")
print("Pipeline executed successfully!")
print(f"Data processed: {results['data_shape']}")
print(f"Model accuracy: {results['accuracy']:.3f}")
print(f"Model AUC: {results['auc']:.3f}")
print(f"Model registered: {results['model_uri']}")

# Test artifact retrieval
artifacts = pipeline.get_model_artifacts()
print(f"Artifacts available: {list(artifacts.keys())}")
