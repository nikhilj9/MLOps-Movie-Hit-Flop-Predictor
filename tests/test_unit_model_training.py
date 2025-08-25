"""Unit tests for model training functions"""

import pandas as pd
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from model_training import setup_mlflow, train_random_forest, evaluate_model, ModelTrainer

class TestModelTrainingFunctions:
    """Test cases for model training functions"""
    
    def setup_method(self):
        """Setup test data"""
        np.random.seed(42)
        self.X_train = pd.DataFrame({
            'feature1': np.random.rand(100),
            'feature2': np.random.rand(100),
            'feature3': np.random.rand(100)
        })
        self.X_test = pd.DataFrame({
            'feature1': np.random.rand(20),
            'feature2': np.random.rand(20),
            'feature3': np.random.rand(20)
        })
        self.y_train = pd.Series(np.random.randint(0, 2, 100))
        self.y_test = pd.Series(np.random.randint(0, 2, 20))
        self.feature_names = ['feature1', 'feature2', 'feature3']
    
    @patch('model_training.mlflow')
    @patch('model_training.os.makedirs')
    def test_setup_mlflow_creates_directory(self, mock_makedirs, mock_mlflow):
        """Test setup_mlflow.fn() creates directory and configures MLflow"""
        mock_mlflow.get_tracking_uri.return_value = "sqlite:///test.db"
        
        result = setup_mlflow.fn()
        
        mock_makedirs.assert_called_once()
        mock_mlflow.set_tracking_uri.assert_called_once()
        mock_mlflow.set_experiment.assert_called_once()
        assert result == "sqlite:///test.db"
    
    @patch('model_training.mlflow')
    @patch('model_training.GridSearchCV')
    def test_train_random_forest_training_process(self, mock_grid_search, mock_mlflow):
        """Test train_random_forest.fn() training process"""
        # Setup mocks
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([1, 0, 1, 0])
        mock_model.predict_proba.return_value = np.array([[0.3, 0.7], [0.8, 0.2], [0.4, 0.6], [0.9, 0.1]])
        
        mock_grid_instance = MagicMock()
        mock_grid_instance.best_estimator_ = mock_model
        mock_grid_instance.best_params_ = {'n_estimators': 100}
        mock_grid_instance.best_score_ = 0.85
        mock_grid_search.return_value = mock_grid_instance
        
        mock_mlflow.active_run.return_value.info.run_id = "test_run_id"
        
        # Mock the context manager
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=None)
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=None)
        
        # Test data (smaller for faster testing)
        X_test_small = self.X_test.iloc[:4]
        y_test_small = pd.Series([1, 0, 1, 0])
        
        model, accuracy, auc, model_uri = train_random_forest.fn(
            self.X_train, self.y_train, X_test_small, y_test_small
        )
        
        # Verify GridSearchCV was called
        mock_grid_search.assert_called_once()
        mock_grid_instance.fit.assert_called_once()
        
        # Verify MLflow logging
        mock_mlflow.log_params.assert_called_once()
        mock_mlflow.log_metric.assert_called()
        
        # Verify return values
        assert model == mock_model
        assert isinstance(accuracy, (int, float))
        assert isinstance(auc, (int, float))
        assert "runs:/" in model_uri
    
    def test_evaluate_model_basic_metrics(self):
        """Test evaluate_model.fn() calculates basic metrics"""
        # Create a simple mock model
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([1, 0, 1, 0])
        mock_model.predict_proba.return_value = np.array([[0.2, 0.8], [0.9, 0.1], [0.3, 0.7], [0.8, 0.2]])
        mock_model.feature_importances_ = np.array([0.5, 0.3, 0.2])
        
        y_test_small = pd.Series([1, 0, 1, 0])
        X_test_small = self.X_test.iloc[:4]
        
        results = evaluate_model.fn(mock_model, X_test_small, y_test_small, self.feature_names)
        
        assert 'accuracy' in results
        assert 'auc' in results
        assert 'classification_report' in results
        assert 'feature_importance' in results
        
        # Check feature importance structure
        assert isinstance(results['feature_importance'], dict)
        assert len(results['feature_importance']) == 3
    
    def test_evaluate_model_without_feature_names(self):
        """Test evaluate_model.fn() without feature names"""
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([1, 0])
        mock_model.predict_proba.return_value = np.array([[0.2, 0.8], [0.9, 0.1]])
        
        y_test_small = pd.Series([1, 0])
        X_test_small = self.X_test.iloc[:2]
        
        results = evaluate_model.fn(mock_model, X_test_small, y_test_small, None)
        
        assert results['feature_importance'] is None


class TestModelTrainerClass:
    """Test cases for ModelTrainer class"""
    
    def setup_method(self):
        """Setup test data"""
        np.random.seed(42)
        self.X_train = pd.DataFrame({
            'feature1': np.random.rand(50),
            'feature2': np.random.rand(50)
        })
        self.X_test = pd.DataFrame({
            'feature1': np.random.rand(10),
            'feature2': np.random.rand(10)
        })
        self.y_train = pd.Series(np.random.randint(0, 2, 50))
        self.y_test = pd.Series(np.random.randint(0, 2, 10))
        self.feature_names = ['feature1', 'feature2']
    
    @patch('model_training.model_training_flow')
    def test_train_production_model(self, mock_flow):
        """Test ModelTrainer.train_production_model() calls the flow"""
        mock_results = {
            'model': MagicMock(),
            'accuracy': 0.8,
            'auc': 0.85,
            'model_uri': 'test_uri'
        }
        mock_flow.return_value = mock_results
        
        trainer = ModelTrainer()
        results = trainer.train_production_model(
            self.X_train, self.y_train, self.X_test, self.y_test, self.feature_names
        )
        
        mock_flow.assert_called_once_with(
            self.X_train, self.y_train, self.X_test, self.y_test, self.feature_names
        )
        assert results == mock_results


def test_train_model_convenience_function():
    """Test the train_model convenience function"""
    from model_training import train_model
    
    # Create simple test data
    X_train = pd.DataFrame({'feature1': [1, 2, 3, 4], 'feature2': [5, 6, 7, 8]})
    X_test = pd.DataFrame({'feature1': [1, 2], 'feature2': [5, 6]})
    y_train = pd.Series([0, 1, 0, 1])
    y_test = pd.Series([0, 1])
    
    with patch('model_training.model_training_flow') as mock_flow:
        mock_flow.return_value = {'accuracy': 0.8}
        
        result = train_model(X_train, y_train, X_test, y_test)
        
        mock_flow.assert_called_once()
        assert result == {'accuracy': 0.8}