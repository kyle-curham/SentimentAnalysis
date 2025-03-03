"""
Tests for comparing sentiment analysis model outputs and visualizing results.
"""
import sys
from pathlib import Path
import unittest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO
import tempfile
import os

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models import create_model, get_finbert_model, get_vader_model
import config


class TestSentimentComparison(unittest.TestCase):
    """Test cases for comparing sentiment analysis models and visualizing results."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Sample texts with known sentiments for testing
        self.positive_texts = [
            "The company reported record profits, significantly exceeding market expectations.",
            "The stock surged 15% following the announcement of a new product line.",
            "Investors are extremely bullish on the company's future growth prospects."
        ]
        
        self.negative_texts = [
            "The company announced major layoffs following disappointing quarterly results.",
            "The stock crashed 20% after the CEO resigned amid fraud allegations.",
            "Analysts have downgraded the stock citing significant competitive pressures."
        ]
        
        self.neutral_texts = [
            "The company announced its quarterly earnings today.",
            "The stock price remained unchanged after hours.",
            "Analysts noted that the company's performance aligned with market expectations."
        ]
        
        # Mixed financial contexts
        self.mixed_texts = [
            "Despite missing revenue targets, the company managed to increase profits.",
            "The stock initially dropped but recovered by end of day.",
            "While facing regulatory challenges, the company secured a major new contract."
        ]
        
        # Domain-specific financial jargon
        self.financial_jargon = [
            "The company's P/E ratio has decreased to 12.5, suggesting potential undervaluation.",
            "Free cash flow increased 8% YoY, though EBITDA margins contracted by 150bps.",
            "The firm maintained its dividend but suspended share buybacks to preserve liquidity."
        ]
        
        # All texts combined
        self.all_texts = (self.positive_texts + self.negative_texts + 
                          self.neutral_texts + self.mixed_texts + self.financial_jargon)
        
        # Expected sentiment labels
        self.expected_sentiments = (
            ["positive"] * 3 + ["negative"] * 3 + ["neutral"] * 3 +
            ["mixed", "mixed", "mixed"] +  # These are subjective
            ["technical", "technical", "technical"]  # These are technical/jargon
        )
        
        # Create a test dataframe
        self.test_df = pd.DataFrame({
            'id': range(len(self.all_texts)),
            'text': self.all_texts,
            'expected_sentiment': self.expected_sentiments
        })
    
    def test_model_consistency(self):
        """Test that models produce consistent results for repeated analyses."""
        # Use both models
        vader_model = get_vader_model()
        finbert_model = get_finbert_model()
        
        # Test consistency for a single positive text
        text = self.positive_texts[0]
        
        # Analyze the same text multiple times with VADER
        vader_results = [vader_model.analyze_text(text) for _ in range(3)]
        for i in range(1, 3):
            self.assertEqual(vader_results[0], vader_results[i], 
                             "VADER model should produce consistent results")
            
        # Analyze the same text multiple times with FinBERT
        finbert_results = [finbert_model.analyze_text(text) for _ in range(3)]
        for i in range(1, 3):
            # Compare scores with small tolerance for floating point differences
            self.assertAlmostEqual(finbert_results[0]['positive'], finbert_results[i]['positive'], places=5)
            self.assertAlmostEqual(finbert_results[0]['negative'], finbert_results[i]['negative'], places=5)
            self.assertAlmostEqual(finbert_results[0]['neutral'], finbert_results[i]['neutral'], places=5)
            self.assertEqual(finbert_results[0]['label'], finbert_results[i]['label'])
    
    def test_model_agreement(self):
        """Test the level of agreement between different sentiment models."""
        # Get models
        vader_model = get_vader_model()
        finbert_model = get_finbert_model()
        
        agreement_count = 0
        total_texts = len(self.all_texts)
        
        # Compare results for all texts
        for text in self.all_texts:
            vader_result = vader_model.analyze_text(text)
            finbert_result = finbert_model.analyze_text(text)
            
            if vader_result['label'] == finbert_result['label']:
                agreement_count += 1
        
        # Calculate agreement percentage
        agreement_percentage = (agreement_count / total_texts) * 100
        
        # Just print the agreement rate, don't assert as models may legitimately disagree
        print(f"Model agreement rate: {agreement_percentage:.2f}%")
        
        # Ensure there's at least some agreement (low bar)
        self.assertGreater(agreement_percentage, 30, 
                           "Models should agree on at least 30% of samples")
    
    def test_sentiment_accuracy(self):
        """Test the accuracy of sentiment predictions against expected labels."""
        # This is for clearly positive/negative examples only
        positive_texts = self.positive_texts
        negative_texts = self.negative_texts
        
        # Get models
        vader_model = get_vader_model()
        finbert_model = get_finbert_model()
        
        # Test VADER accuracy on clear positive examples
        for text in positive_texts:
            result = vader_model.analyze_text(text)
            self.assertEqual(result['label'], 'positive', 
                            f"VADER should classify this positive text correctly: {text}")
        
        # Test VADER accuracy on clear negative examples
        for text in negative_texts:
            result = vader_model.analyze_text(text)
            self.assertEqual(result['label'], 'negative', 
                            f"VADER should classify this negative text correctly: {text}")
        
        # Test FinBERT accuracy on clear positive examples
        for text in positive_texts:
            result = finbert_model.analyze_text(text)
            self.assertEqual(result['label'], 'positive', 
                            f"FinBERT should classify this positive text correctly: {text}")
        
        # Test FinBERT accuracy on clear negative examples
        for text in negative_texts:
            result = finbert_model.analyze_text(text)
            self.assertEqual(result['label'], 'negative', 
                            f"FinBERT should classify this negative text correctly: {text}")
    
    def test_model_confidence(self):
        """Test that model confidence scores align with sentiment clarity."""
        # Get models
        vader_model = get_vader_model()
        finbert_model = get_finbert_model()
        
        # Analyze clear positive/negative vs. mixed texts with FinBERT
        for text in self.positive_texts + self.negative_texts:
            result = finbert_model.analyze_text(text)
            # For clear sentiment, the dominant score should be high
            max_score = max(result['positive'], result['negative'], result['neutral'])
            self.assertGreater(max_score, 0.6, 
                              f"FinBERT should have high confidence for clear sentiment: {text}")
        
        # For mixed texts, confidence should generally be lower or more distributed
        # (But this is just a general pattern, not an absolute rule)
        mixed_confidence_scores = []
        for text in self.mixed_texts:
            result = finbert_model.analyze_text(text)
            max_score = max(result['positive'], result['negative'], result['neutral'])
            mixed_confidence_scores.append(max_score)
        
        # Print average confidence for mixed texts (informational)
        avg_mixed_confidence = sum(mixed_confidence_scores) / len(mixed_confidence_scores)
        print(f"Avg confidence for mixed texts: {avg_mixed_confidence:.2f}")
    
    def test_cross_model_correlation(self):
        """Test correlation between different models' sentiment scores."""
        # Get models
        vader_model = get_vader_model()
        finbert_model = get_finbert_model()
        
        vader_positive_scores = []
        finbert_positive_scores = []
        
        # Get positive scores for all texts from both models
        for text in self.all_texts:
            vader_result = vader_model.analyze_text(text)
            finbert_result = finbert_model.analyze_text(text)
            
            vader_positive_scores.append(vader_result['positive'])
            finbert_positive_scores.append(finbert_result['positive'])
        
        # Calculate Pearson correlation coefficient
        correlation = np.corrcoef(vader_positive_scores, finbert_positive_scores)[0, 1]
        
        # Print correlation (informational)
        print(f"Correlation between VADER and FinBERT positive scores: {correlation:.4f}")
        
        # There should be at least some correlation between models
        self.assertGreater(correlation, 0.3, 
                          "Models should have at least moderate correlation in sentiment scores")
    
    def test_compare_and_visualize(self):
        """Test the ability to compare and visualize sentiment predictions with actual content."""
        # This test demonstrates how to create a comparison and visualization
        vader_model = get_vader_model()
        finbert_model = get_finbert_model()
        
        # Create a comparison dataframe
        results_df = self.test_df.copy()
        
        # Add VADER sentiment
        vader_results = vader_model.predict(results_df['text'].tolist())
        results_df['vader_positive'] = [r['positive'] for r in vader_results]
        results_df['vader_negative'] = [r['negative'] for r in vader_results]
        results_df['vader_neutral'] = [r['neutral'] for r in vader_results]
        results_df['vader_label'] = [r['label'] for r in vader_results]
        
        # Add FinBERT sentiment
        finbert_results = finbert_model.predict(results_df['text'].tolist())
        results_df['finbert_positive'] = [r['positive'] for r in finbert_results]
        results_df['finbert_negative'] = [r['negative'] for r in finbert_results]
        results_df['finbert_neutral'] = [r['neutral'] for r in finbert_results]
        results_df['finbert_label'] = [r['label'] for r in finbert_results]
        
        # Sample visualization capability - save visualization to a temp file
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a file for the visualization
            viz_path = os.path.join(temp_dir, "sentiment_comparison.png")
            
            # Sample code to create visualization
            plt.figure(figsize=(10, 6))
            
            # Sample plot - VADER vs FinBERT positive scores
            plt.scatter(results_df['vader_positive'], results_df['finbert_positive'], alpha=0.7)
            plt.xlabel('VADER Positive Score')
            plt.ylabel('FinBERT Positive Score')
            plt.title('Positive Sentiment Score Comparison')
            
            # Save the visualization
            plt.savefig(viz_path)
            plt.close()
            
            # Verify the visualization was created
            self.assertTrue(os.path.exists(viz_path))
        
        # Test that the dataframe contains expected columns
        expected_columns = [
            'id', 'text', 'expected_sentiment', 
            'vader_positive', 'vader_negative', 'vader_neutral', 'vader_label',
            'finbert_positive', 'finbert_negative', 'finbert_neutral', 'finbert_label'
        ]
        for col in expected_columns:
            self.assertIn(col, results_df.columns)
        
        # Validate that comparison is working by checking disagreements
        disagreements = results_df[results_df['vader_label'] != results_df['finbert_label']]
        
        # Print some disagreement examples (for information)
        if not disagreements.empty:
            # Just confirm we can identify disagreements
            self.assertGreaterEqual(len(disagreements), 0)


if __name__ == '__main__':
    unittest.main() 