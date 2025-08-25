"""Unit tests for data pipeline functions"""

import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_pipeline import load_and_explore_data, clean_data, create_target_variable


class TestDataPipeline:
    
    @pytest.fixture
    def sample_raw_data(self):
        """Create sample raw movie data for testing"""
        return pd.DataFrame({
            'title': ['Movie A', 'Movie B', 'Movie C', 'Movie A'],  # Duplicate
            'budget': [100000000, 0, 50000000, 100000000],  # Zero budget
            'runtime': [120, 90, 0, 120],  # Zero runtime
            'revenue': [200000000, 50000000, 75000000, 200000000],
            'vote_average': [7.5, 6.0, 8.0, 7.5],
            'vote_count': [1000, 500, 1500, 1000],
            'popularity': [85.5, 45.2, 92.1, 85.5],
            'release_date': ['2023-01-15', '2023-02-20', '2023-03-10', '2023-01-15'],
            'genres': ["[{'name': 'Action'}]"] * 4,
            'original_language': ['en', 'en', 'fr', 'en']
        })
    
    def test_load_and_explore_data_success(self, tmp_path, sample_raw_data):
        """Test successful data loading"""
        # Create temporary CSV file
        csv_file = tmp_path / "test_movies.csv"
        sample_raw_data.to_csv(csv_file, index=False)
        
        # Test the function directly (not as Prefect task)
        result = load_and_explore_data.fn(str(csv_file))
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 4
        assert list(result.columns) == list(sample_raw_data.columns)
    
    def test_load_and_explore_data_file_not_found(self):
        """Test error handling for missing file"""
        with pytest.raises(Exception):
            load_and_explore_data.fn("nonexistent_file.csv")
    
    def test_clean_data_removes_duplicates(self, sample_raw_data):
        """Test duplicate removal"""
        result = clean_data.fn(sample_raw_data)
        
        # After cleaning: removes duplicate + zero budget + zero runtime = 1 row left
        assert len(result) == 1  # Only Movie A remains valid
        assert result['title'].iloc[0] == 'Movie A'
    
    def test_clean_data_removes_zero_values(self, sample_raw_data):
        """Test removal of zero budget and runtime"""
        result = clean_data.fn(sample_raw_data)
        
        # Should remove rows with budget=0 or runtime=0
        assert all(result['budget'] > 0)
        assert all(result['runtime'] > 0)
    
    def test_clean_data_converts_date(self, sample_raw_data):
        """Test date conversion"""
        result = clean_data.fn(sample_raw_data)
        
        assert pd.api.types.is_datetime64_any_dtype(result['release_date'])
    
    def test_create_target_variable_roi_calculation(self):
        """Test ROI calculation and target creation"""
        test_data = pd.DataFrame({
            'budget': [100000000, 50000000, 200000000],
            'revenue': [300000000, 100000000, 400000000],
        })
        
        result_df, roi_threshold = create_target_variable.fn(test_data)
        
        # Check ROI calculation (revenue / (budget + 1))
        expected_roi = [
            300000000 / (100000000 + 1),  # ≈ 2.999...
            100000000 / (50000000 + 1),   # ≈ 1.999...  
            400000000 / (200000000 + 1)   # ≈ 1.999...
        ]
        
        # Use approximate comparison for floats
        for actual, expected in zip(result_df['roi'].tolist(), expected_roi):
            assert actual == pytest.approx(expected, rel=1e-6)
        
        # Check threshold calculation (70th percentile)
        expected_threshold = pd.Series(expected_roi).quantile(0.7)
        assert roi_threshold == pytest.approx(expected_threshold, rel=1e-6)
        
        # Check target creation
        assert 'is_hit' in result_df.columns
        assert result_df['is_hit'].dtype == int
    
    def test_create_target_variable_hit_distribution(self):
        """Test hit/flop distribution makes sense"""
        test_data = pd.DataFrame({
            'budget': [100000000] * 10,
            'revenue': [50000000, 100000000, 150000000, 200000000, 250000000,
                       300000000, 350000000, 400000000, 450000000, 500000000]
        })
        
        result_df, roi_threshold = create_target_variable.fn(test_data)
        
        # With 70th percentile, should have ~30% hits
        hit_percentage = result_df['is_hit'].mean()
        assert 0.2 <= hit_percentage <= 0.4  # Reasonable range

    def test_debug_clean_data(self, sample_raw_data):
        """Debug what actually remains after cleaning"""
        print("Original data:")
        print(sample_raw_data)
        
        result = clean_data.fn(sample_raw_data)
        
        print("\nAfter cleaning:")
        print(result)
        print(f"Remaining titles: {result['title'].tolist()}")