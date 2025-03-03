"""
VADER model implementation for sentiment analysis.
VADER (Valence Aware Dictionary and sEntiment Reasoner) is a lexicon and rule-based sentiment
analysis tool specifically attuned to sentiments expressed in social media.
"""
from typing import Dict, List, Any
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from tqdm import tqdm

from src.models.base_model import BaseSentimentModel


class VADERModel(BaseSentimentModel):
    """VADER model for sentiment analysis."""
    
    def __init__(self, model_name: str = "vader", **kwargs: Any) -> None:
        """Initialize the VADER model.
        
        Args:
            model_name: Ignored for VADER (included for compatibility)
            **kwargs: Ignored for VADER (included for compatibility)
        """
        self.model_name = model_name
        self.analyzer = SentimentIntensityAnalyzer()
        print("VADER sentiment analyzer initialized")
    
    def predict(self, texts: List[str]) -> List[Dict[str, float]]:
        """Predict sentiment for a list of texts.
        
        Args:
            texts: List of text strings to analyze
            
        Returns:
            List of dictionaries containing sentiment scores
        """
        # Handle empty input
        if not texts:
            return []
        
        results = []
        for text in texts:
            # Get VADER sentiment scores
            scores = self.analyzer.polarity_scores(text)
            
            # VADER provides compound, pos, neg, neu scores
            # Convert to our standard format
            result = {
                "positive": scores["pos"],
                "negative": scores["neg"],
                "neutral": scores["neu"],
                # Determine label based on compound score
                "label": self._get_sentiment_label(scores["compound"])
            }
            results.append(result)
            
        return results
    
    def predict_batch(self, texts: List[str], batch_size: int = 16) -> List[Dict[str, float]]:
        """Predict sentiment for a list of texts in batches.
        
        Args:
            texts: List of text strings to analyze
            batch_size: Number of texts to process at once
            
        Returns:
            List of dictionaries containing sentiment scores
        """
        # Handle empty input
        if not texts:
            return []
        
        results = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Processing batches"):
            batch_texts = texts[i:i+batch_size]
            batch_results = self.predict(batch_texts)
            results.extend(batch_results)
            
        return results
    
    def _get_sentiment_label(self, compound_score: float) -> str:
        """Convert VADER compound score to sentiment label.
        
        Args:
            compound_score: VADER compound score (-1 to 1)
            
        Returns:
            Sentiment label: positive, negative, or neutral
        """
        if compound_score >= 0.05:
            return "positive"
        elif compound_score <= -0.05:
            return "negative"
        else:
            return "neutral" 