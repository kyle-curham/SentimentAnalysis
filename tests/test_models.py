"""
Tests for sentiment analysis models.
"""
import sys
from pathlib import Path
import unittest
import pandas as pd

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models import create_model, get_finbert_model, get_vader_model
import config


class TestSentimentModels(unittest.TestCase):
    """Test cases for sentiment analysis models."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sample_texts = [
            "The company reported strong earnings, beating analyst expectations.",
            "Investors are concerned about the company's declining market share.",
            "The stock price remained unchanged after the announcement."
        ]
        self.sample_df = pd.DataFrame({
            'id': [1, 2, 3],
            'text': self.sample_texts
        })
    
    def test_model_factory(self):
        """Test that models can be created using the factory function."""
        # Test default model
        default_model = create_model()
        self.assertIsNotNone(default_model)
        
        # Test FinBERT model
        finbert_model = create_model("finbert")
        self.assertIsNotNone(finbert_model)
        
        # Test VADER model
        vader_model = create_model("vader")
        self.assertIsNotNone(vader_model)
        
        # Test convenience functions
        finbert_model2 = get_finbert_model()
        self.assertIsNotNone(finbert_model2)
        
        vader_model2 = get_vader_model()
        self.assertIsNotNone(vader_model2)
        
        # Test invalid model name
        with self.assertRaises(ValueError):
            create_model("invalid_model_name")
    
    def test_sentiment_prediction_format(self):
        """Test that sentiment predictions have the correct format."""
        # Use VADER for faster tests
        model = get_vader_model()
        
        # Test single text prediction
        result = model.analyze_text(self.sample_texts[0])
        
        # Check result format
        self.assertIsInstance(result, dict)
        self.assertIn('positive', result)
        self.assertIn('negative', result)
        self.assertIn('neutral', result)
        self.assertIn('label', result)
        
        # Check values
        self.assertIsInstance(result['positive'], float)
        self.assertIsInstance(result['negative'], float)
        self.assertIsInstance(result['neutral'], float)
        self.assertIsInstance(result['label'], str)
        
        # Check that probabilities sum approximately to 1
        total_prob = result['positive'] + result['negative'] + result['neutral']
        self.assertAlmostEqual(total_prob, 1.0, places=1)
        
        # Check that label is one of the expected values
        self.assertIn(result['label'], ['positive', 'negative', 'neutral'])
    
    def test_dataframe_analysis(self):
        """Test that dataframe analysis works correctly."""
        # Use VADER for faster tests
        model = get_vader_model()
        
        # Analyze dataframe
        result_df = model.analyze_dataframe(self.sample_df, 'text')
        
        # Check that original columns are preserved
        self.assertIn('id', result_df.columns)
        self.assertIn('text', result_df.columns)
        
        # Check that sentiment columns are added
        self.assertIn('sentiment_positive', result_df.columns)
        self.assertIn('sentiment_negative', result_df.columns)
        self.assertIn('sentiment_neutral', result_df.columns)
        self.assertIn('sentiment_label', result_df.columns)
        
        # Check that we have the right number of rows
        self.assertEqual(len(result_df), len(self.sample_df))


if __name__ == '__main__':
    unittest.main() 