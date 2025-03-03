"""
Factory for creating sentiment analysis models.
"""
from typing import Dict, Any, Optional, Union

from src.models.base_model import BaseSentimentModel
from src.models.finbert_model import FinBERTModel
from src.models.vader_model import VADERModel
import config


def create_model(model_name: str = None, **kwargs: Any) -> BaseSentimentModel:
    """Create and return a sentiment analysis model.
    
    Args:
        model_name: Name of the model to create 
                   (finbert_finetuned, finbert, bert, distilbert, roberta, vader)
                    If None, uses the default model from config
        **kwargs: Additional arguments to pass to the model constructor
        
    Returns:
        Initialized sentiment analysis model
        
    Raises:
        ValueError: If the model name is not recognized
    """
    # Use default model if none specified
    if model_name is None:
        model_name = config.DEFAULT_MODEL
    
    model_name = model_name.lower()
    
    # Check if model name is valid
    if model_name not in config.AVAILABLE_MODELS:
        valid_models = list(config.AVAILABLE_MODELS.keys())
        raise ValueError(f"Unknown model: {model_name}. Available models: {valid_models}")
    
    # Get the pretrained model identifier or path
    model_path = config.AVAILABLE_MODELS[model_name]
    
    # Create the model based on type
    if model_name == "vader":
        return VADERModel(model_name=model_path, **kwargs)
    elif model_name in ["finbert_finetuned", "finbert", "bert", "distilbert", "roberta"]:
        return FinBERTModel(model_name=model_path, **kwargs)
    else:
        raise ValueError(f"Model '{model_name}' is in config but not implemented")
    

# Define an easy way to get models
def get_finbert_model(**kwargs: Any) -> FinBERTModel:
    """Get a FinBERT model instance."""
    return create_model("finbert", **kwargs)


def get_finetuned_finbert_model(**kwargs: Any) -> FinBERTModel:
    """Get a finetuned FinBERT model instance."""
    return create_model("finbert_finetuned", **kwargs)


def get_vader_model(**kwargs: Any) -> VADERModel:
    """Get a VADER model instance."""
    return create_model("vader", **kwargs) 