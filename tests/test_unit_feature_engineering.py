"""Unit tests for feature engineering functions"""

import pandas as pd
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from feature_engineering import extract_genres, engineer_features, prepare_features, prepare_train_test_split, FeatureEngineer

class TestFeatureEngineeringFunctions:
    """Test cases for feature engineering functions"""
    
    def setup_method(self):
        """Setup test data"""
        self.sample_df = pd.DataFrame({
            'title': ['Movie A', 'Movie B', 'Movie C', 'Movie D', 'Movie E', 'Movie F'],
            'budget': [100000000, 50000000, 200000000, 30000000, 150000000, 80000000],
            'revenue': [500000000, 100000000, 800000000, 60000000, 600000000, 200000000],
            'runtime': [120, 95, 150, 110, 130, 105],
            'vote_average': [7.5, 6.2, 8.1, 5.8, 7.8, 6.9],
            'vote_count': [1000, 500, 2000, 300, 1500, 800],
            'popularity': [50.5, 25.3, 75.8, 15.2, 60.4, 35.7],
            'genres': [
                "[{'name': 'Action'}, {'name': 'Adventure'}]",
                "[{'name': 'Comedy'}]",
                "[{'name': 'Drama'}, {'name': 'Thriller'}]",
                "[{'name': 'Horror'}]",
                "[{'name': 'Sci-Fi'}, {'name': 'Action'}]",
                "[{'name': 'Romance'}]"
            ],
            'original_language': ['en', 'fr', 'en', 'es', 'en', 'en'],
            'release_date': pd.to_datetime([
                '2020-01-15', '2019-06-20', '2021-12-10', 
                '2018-03-05', '2020-08-22', '2019-11-30'
            ]),
            'is_hit': [1, 0, 1, 0, 1, 0]
        })
    
    def test_extract_genres_valid_json(self):
        """Test genre extraction with valid JSON"""
        genres_str = "[{'name': 'Action'}, {'name': 'Adventure'}]"
        genre_count, main_genre = extract_genres(genres_str)
        
        assert genre_count == 2
        assert main_genre == 'Action'
    
    def test_extract_genres_empty_list(self):
        """Test genre extraction with empty list"""
        genres_str = "[]"
        genre_count, main_genre = extract_genres(genres_str)
        
        assert genre_count == 0
        assert main_genre == 'Unknown'
    
    def test_extract_genres_invalid_json(self):
        """Test genre extraction with invalid JSON"""
        genres_str = "invalid json"
        genre_count, main_genre = extract_genres(genres_str)
        
        assert genre_count == 0
        assert main_genre == 'Unknown'
    
    def test_engineer_features_creates_all_features(self):
        """Test that engineer_features.fn() creates all expected features"""
        result = engineer_features.fn(self.sample_df)
        
        # Check new columns exist
        expected_columns = [
            'release_year', 'budget_category', 'genre_count', 
            'main_genre', 'is_english'
        ]
        
        for col in expected_columns:
            assert col in result.columns, f"Missing column: {col}"
        
        # Check release year extraction
        assert result['release_year'].iloc[0] == 2020
        assert result['release_year'].iloc[1] == 2019
        
        # Check genre features
        assert result['genre_count'].iloc[0] == 2  # Action + Adventure
        assert result['genre_count'].iloc[1] == 1  # Comedy
        
        assert result['main_genre'].iloc[0] == 'Action'
        assert result['main_genre'].iloc[1] == 'Comedy'
        
        # Check language feature
        assert result['is_english'].iloc[0] == 1  # English
        assert result['is_english'].iloc[1] == 0  # French
        assert result['is_english'].iloc[2] == 1  # English
    
    def test_prepare_features_creates_feature_matrix(self):
        """Test prepare_features.fn() creates proper feature matrix"""
        df_featured = engineer_features.fn(self.sample_df)
        X, y, encoders = prepare_features.fn(df_featured)
        
        # Check output shapes
        assert X.shape[0] == 6  # 6 movies
        assert X.shape[1] == 10  # Expected feature count
        assert len(y) == 6
        
        # Check encoders returned
        assert 'budget' in encoders
        assert 'genre' in encoders
        
        # Check encoded features are numeric
        assert X['budget_category'].dtype in [np.int64, np.int32]
        assert X['main_genre'].dtype in [np.int64, np.int32]
    
    @patch('feature_engineering.SMOTE')
    def test_prepare_train_test_split_applies_smote(self, mock_smote):
        """Test prepare_train_test_split.fn() applies SMOTE correctly"""
        # Setup mock SMOTE
        mock_smote_instance = MagicMock()
        mock_smote_instance.fit_resample.return_value = (
            np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]),
            np.array([0, 0, 1, 1])
        )
        mock_smote.return_value = mock_smote_instance
        
        # Create test data
        X = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5, 6],
            'feature2': [10, 20, 30, 40, 50, 60],
            'feature3': [100, 200, 300, 400, 500, 600]
        })
        y = pd.Series([0, 0, 0, 1, 1, 1])
        
        result = prepare_train_test_split.fn(X, y)
        X_train, X_test, y_train, y_test, X_train_balanced, y_train_balanced = result
        
        # Check that SMOTE was called
        mock_smote.assert_called_once()
        mock_smote_instance.fit_resample.assert_called_once()
        
        # Check output shapes
        assert len(X_train) + len(X_test) == 6
        assert len(X_train_balanced) == 4  # From mock
        assert len(y_train_balanced) == 4  # From mock


class TestFeatureEngineerClass:
    """Test cases for FeatureEngineer class"""
    
    def setup_method(self):
        """Setup test data"""
        self.feature_engineer = FeatureEngineer()
        # Need more samples for train_test_split to work
        self.sample_df = pd.DataFrame({
            'title': [f'Movie {i}' for i in range(10)],
            'budget': [100000000 + i*10000000 for i in range(10)],
            'revenue': [500000000 + i*50000000 for i in range(10)],
            'runtime': [120 + i*5 for i in range(10)],
            'vote_average': [7.0 + i*0.1 for i in range(10)],
            'vote_count': [1000 + i*100 for i in range(10)],
            'popularity': [50.0 + i*5 for i in range(10)],
            'genres': [f"[{{'name': 'Genre{i}'}}]" for i in range(10)],
            'original_language': ['en' if i%2==0 else 'fr' for i in range(10)],
            'release_date': pd.to_datetime([f'202{i%3}-01-01' for i in range(10)]),
            'is_hit': [i%2 for i in range(10)]  # Alternating 0,1
        })
    
    def test_run_feature_pipeline_integration(self):
        """Test complete feature pipeline integration"""
        train_test_data, encoders = self.feature_engineer.run_feature_pipeline(self.sample_df)
        
        X_train, X_test, y_train, y_test, X_train_balanced, y_train_balanced = train_test_data
        
        # Check all outputs are present
        assert X_train is not None
        assert X_test is not None
        assert y_train is not None
        assert y_test is not None
        assert X_train_balanced is not None
        assert y_train_balanced is not None
        
        # Check encoders returned
        assert 'budget' in encoders
        assert 'genre' in encoders
        
        # Check class attributes updated
        assert self.feature_engineer.is_fitted == True
    
    def test_get_encoders(self):
        """Test encoder retrieval"""
        encoders = self.feature_engineer.get_encoders()
        
        assert 'budget' in encoders
        assert 'genre' in encoders
        assert encoders['budget'] == self.feature_engineer.le_budget
        assert encoders['genre'] == self.feature_engineer.le_genre


def test_process_features_function():
    """Test the convenience function"""
    sample_df = pd.DataFrame({
        'title': [f'Movie {i}' for i in range(8)],
        'budget': [100000000 + i*10000000 for i in range(8)],
        'revenue': [500000000 + i*50000000 for i in range(8)],
        'runtime': [120 + i*5 for i in range(8)],
        'vote_average': [7.0 + i*0.1 for i in range(8)],
        'vote_count': [1000 + i*100 for i in range(8)],
        'popularity': [50.0 + i*5 for i in range(8)],
        'genres': [f"[{{'name': 'Genre{i}'}}]" for i in range(8)],
        'original_language': ['en' if i%2==0 else 'fr' for i in range(8)],
        'release_date': pd.to_datetime([f'202{i%3}-01-01' for i in range(8)]),
        'is_hit': [i%2 for i in range(8)]
    })
    
    from feature_engineering import process_features
    train_test_data, encoders = process_features(sample_df)
    
    assert train_test_data is not None
    assert encoders is not None