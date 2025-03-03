"""
Base model interface for sentiment analysis.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Union, Any, Optional
import pandas as pd


class BaseSentimentModel(ABC):
    """Base class for all sentiment analysis models."""
    
    @abstractmethod
    def __init__(self, model_name: str, **kwargs: Any) -> None:
        """Initialize the model.
        
        Args:
            model_name: Name of the model to load
            **kwargs: Additional arguments for model initialization
        """
        pass
    
    @abstractmethod
    def predict(self, texts: List[str]) -> List[Dict[str, float]]:
        """Predict sentiment for a list of texts.
        
        Args:
            texts: List of text strings to analyze
            
        Returns:
            List of dictionaries containing sentiment scores:
            [
                {
                    'positive': float,  # Positive sentiment score (0-1)
                    'negative': float,  # Negative sentiment score (0-1)
                    'neutral': float,   # Neutral sentiment score (0-1)
                    'label': str        # Most likely sentiment label
                },
                ...
            ]
        """
        pass
    
    @abstractmethod
    def predict_batch(self, texts: List[str], batch_size: int = 16) -> List[Dict[str, float]]:
        """Predict sentiment for a list of texts in batches.
        
        Args:
            texts: List of text strings to analyze
            batch_size: Number of texts to process at once
            
        Returns:
            List of dictionaries containing sentiment scores
        """
        pass
    
    def analyze_dataframe(self, df: pd.DataFrame, text_column: str, 
                         batch_size: int = 16, new_column_prefix: str = "sentiment_") -> pd.DataFrame:
        """Analyze texts in a dataframe and add sentiment columns.
        
        Args:
            df: Dataframe containing text data
            text_column: Name of the column containing text to analyze
            batch_size: Number of texts to process at once
            new_column_prefix: Prefix for the new sentiment columns
            
        Returns:
            DataFrame with added sentiment columns:
            - {prefix}positive: Positive sentiment score
            - {prefix}negative: Negative sentiment score 
            - {prefix}neutral: Neutral sentiment score
            - {prefix}label: Most likely sentiment label
        """
        if text_column not in df.columns:
            raise ValueError(f"Column {text_column} not found in dataframe")
        
        # Get a list of texts from the dataframe
        texts = df[text_column].tolist()
        
        # Predict sentiments in batches
        sentiments = self.predict_batch(texts, batch_size=batch_size)
        
        # Add sentiment columns to the dataframe
        df[f"{new_column_prefix}positive"] = [s['positive'] for s in sentiments]
        df[f"{new_column_prefix}negative"] = [s['negative'] for s in sentiments]
        df[f"{new_column_prefix}neutral"] = [s['neutral'] for s in sentiments]
        df[f"{new_column_prefix}label"] = [s['label'] for s in sentiments]
        
        return df
    
    def analyze_text(self, text: str) -> Dict[str, float]:
        """Analyze a single text string.
        
        Args:
            text: Text string to analyze
            
        Returns:
            Dictionary containing sentiment scores
        """
        results = self.predict([text])
        return results[0] 