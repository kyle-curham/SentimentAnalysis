"""
FinBERT model implementation for financial sentiment analysis.
"""
from typing import Dict, List, Any, Optional, Union
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

from src.models.base_model import BaseSentimentModel


class FinBERTModel(BaseSentimentModel):
    """FinBERT model for financial sentiment analysis."""
    
    def __init__(self, model_name: str = "yiyanghkust/finbert-tone", device: Optional[str] = None,
                 **kwargs: Any) -> None:
        """Initialize the FinBERT model.
        
        Args:
            model_name: Name or path of the model to load (default: "yiyanghkust/finbert-tone")
            device: Device to use for inference ("cpu" or "cuda"), default is None (auto-select)
            **kwargs: Additional arguments for model initialization
        """
        self.model_name = model_name
        
        # Determine device (use GPU if available)
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"Loading FinBERT model '{model_name}' on {self.device}...")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        
        # FinBERT-tone has 3 labels: positive, negative, neutral
        self.labels = ["positive", "negative", "neutral"]
        
        # Set model to evaluation mode
        self.model.eval()
        
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
            
        # Tokenize inputs
        encoded_inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)
        
        # Perform inference
        with torch.no_grad():
            outputs = self.model(**encoded_inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        # Convert predictions to list of dictionaries
        results = []
        for pred in predictions:
            pred_dict = {
                "positive": float(pred[0]),
                "negative": float(pred[1]),
                "neutral": float(pred[2]),
                "label": self.labels[torch.argmax(pred).item()]
            }
            results.append(pred_dict)
            
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