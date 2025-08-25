"""Integration tests for the complete MLOps pipeline"""

import pandas as pd
import pytest
import sys
import os
from unittest.mock import patch

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_pipeline import data_processing_flow
from feature_engineering import feature_engineering_flow
from model_training import model_training_flow

class TestPipelineIntegration:
    """Test complete end-to-end pipeline"""
    
    def setup_method(self):
        """Setup integration test data"""
        # Create MORE test data (SMOTE needs enough samples)
        self.test_data = pd.DataFrame({
            'title': [f'Movie {i}' for i in range(50)],  # Increased from 20 to 50
            'budget': [50000000 + i*10000000 for i in range(50)],
            'revenue': [100000000 + i*25000000 for i in range(50)],
            'runtime': [90 + i*2 for i in range(50)],  # Smaller increment for variety
            'vote_average': [5.0 + (i%10)*0.3 for i in range(50)],  # More variety
            'vote_count': [500 + i*50 for i in range(50)],
            'popularity': [20.0 + (i%15)*2 for i in range(50)],  # More variety
            'genres': [f"[{{'name': 'Genre{i%8}'}}]" for i in range(50)],  # More genres
            'original_language': ['en' if i%3==0 else 'fr' if i%3==1 else 'es' for i in range(50)],
            'release_date': [f'202{i%3}-{(i%12)+1:02d}-01' for i in range(50)],  # Varied dates
            'production_companies': [f'Company {i%10}' for i in range(50)]
        })
        # Save test data
        os.makedirs('test_data', exist_ok=True)
        self.test_data.to_csv('test_data/test_movies.csv', index=False)
    
    def teardown_method(self):
        """Clean up test files"""
        import shutil
        if os.path.exists('test_data'):
            shutil.rmtree('test_data')
        if os.path.exists('mlflow_data'):
            shutil.rmtree('mlflow_data')
    
    @patch('data_pipeline.DATA_CONFIG', {'data_path': 'test_data/test_movies.csv', 'test_size': 0.2, 'random_state': 42, 'roi_threshold_quantile': 0.7})
    def test_complete_pipeline_integration(self):
        """Test data → features → model pipeline"""
        # Step 1: Data processing
        df, roi_threshold = data_processing_flow()
        assert len(df) > 0
        assert 'is_hit' in df.columns
        assert roi_threshold > 0
        
        # Step 2: Feature engineering
        train_test_data, encoders = feature_engineering_flow(df)
        X_train, X_test, y_train, y_test, X_train_balanced, y_train_balanced = train_test_data
        
        assert X_train.shape[0] > 0
        assert X_test.shape[0] > 0
        assert len(encoders) == 2
        
        # Step 3: Model training
        feature_names = list(X_train.columns)
        with patch('model_training.GridSearchCV') as mock_grid, \
            patch('model_training.mlflow') as mock_mlflow:
            
            # Create a proper mock model that handles predict operations
            from sklearn.ensemble import RandomForestClassifier
            real_model = RandomForestClassifier(n_estimators=10, random_state=42)
            real_model.fit(X_train_balanced.iloc[:20], y_train_balanced.iloc[:20])  # Quick fit with subset
            
            mock_grid_instance = mock_grid.return_value
            mock_grid_instance.best_estimator_ = real_model
            mock_grid_instance.best_params_ = {'n_estimators': 10}
            mock_grid_instance.best_score_ = 0.8
            
            # Mock MLflow context manager
            mock_mlflow.start_run.return_value.__enter__ = lambda x: None
            mock_mlflow.start_run.return_value.__exit__ = lambda x, y, z, w: None
            mock_mlflow.active_run.return_value.info.run_id = "test_run"
            
            results = model_training_flow(
                X_train_balanced, y_train_balanced, X_test, y_test, feature_names
            )
        
        # Verify final results
        assert 'accuracy' in results
        assert 'auc' in results
        assert 'model' in results
        assert results['feature_names'] == feature_names