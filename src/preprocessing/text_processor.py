"""
Text processing module for cleaning and preparing text for sentiment analysis.
"""
import re
import string
from typing import List, Optional, Dict, Any, Callable
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd
from tqdm import tqdm


def ensure_nltk_resources() -> None:
    """Ensure that required NLTK resources are downloaded."""
    resources = ['punkt', 'stopwords', 'wordnet']
    for resource in resources:
        try:
            nltk.data.find(f'tokenizers/{resource}')
        except LookupError:
            nltk.download(resource, quiet=True)


# Download NLTK resources if needed
ensure_nltk_resources()


def basic_text_cleaning(text: str) -> str:
    """Perform basic text cleaning.
    
    Args:
        text: Text to clean
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove punctuation
    text = re.sub(f'[{re.escape(string.punctuation)}]', ' ', text)
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Strip whitespace
    text = text.strip()
    
    return text


def remove_stopwords(text: str, stop_words: Optional[List[str]] = None) -> str:
    """Remove stopwords from text.
    
    Args:
        text: Text to process
        stop_words: List of stopwords to remove (uses NLTK stopwords if None)
        
    Returns:
        Text with stopwords removed
    """
    if not text:
        return ""
    
    # Get default stopwords if none provided
    if stop_words is None:
        stop_words = stopwords.words('english')
    
    # Tokenize text
    tokens = word_tokenize(text)
    
    # Remove stopwords
    filtered_tokens = [word for word in tokens if word not in stop_words]
    
    # Join tokens back into a string
    filtered_text = ' '.join(filtered_tokens)
    
    return filtered_text


def lemmatize_text(text: str) -> str:
    """Lemmatize text to reduce words to their base form.
    
    Args:
        text: Text to lemmatize
        
    Returns:
        Lemmatized text
    """
    if not text:
        return ""
    
    # Initialize lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    # Tokenize text
    tokens = word_tokenize(text)
    
    # Lemmatize tokens
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    # Join tokens back into a string
    lemmatized_text = ' '.join(lemmatized_tokens)
    
    return lemmatized_text


def preprocess_text(text: str, clean: bool = True, remove_stops: bool = True,
                   lemmatize: bool = True, custom_stopwords: Optional[List[str]] = None) -> str:
    """Preprocess text with configurable steps.
    
    Args:
        text: Text to preprocess
        clean: Whether to perform basic cleaning
        remove_stops: Whether to remove stopwords
        lemmatize: Whether to lemmatize words
        custom_stopwords: Custom stopwords to remove
        
    Returns:
        Preprocessed text
    """
    if not text:
        return ""
    
    # Perform basic cleaning
    if clean:
        text = basic_text_cleaning(text)
    
    # Remove stopwords
    if remove_stops:
        text = remove_stopwords(text, custom_stopwords)
    
    # Lemmatize text
    if lemmatize:
        text = lemmatize_text(text)
    
    return text


def preprocess_dataframe(df: pd.DataFrame, text_column: str, 
                        new_column_name: Optional[str] = None,
                        clean: bool = True, 
                        remove_stops: bool = True,
                        lemmatize: bool = False,
                        custom_stopwords: Optional[List[str]] = None,
                        show_progress: bool = True) -> pd.DataFrame:
    """Preprocess text in a dataframe column.
    
    Args:
        df: Dataframe containing text data
        text_column: Name of the column containing text to preprocess
        new_column_name: Name of the new column for preprocessed text
                        (if None, overwrites the original column)
        clean: Whether to perform basic cleaning
        remove_stops: Whether to remove stopwords
        lemmatize: Whether to lemmatize words
        custom_stopwords: Custom stopwords to remove
        show_progress: Whether to show a progress bar
        
    Returns:
        DataFrame with preprocessed text
    """
    if text_column not in df.columns:
        raise ValueError(f"Column {text_column} not found in dataframe")
    
    # Set output column name
    output_column = new_column_name if new_column_name else text_column
    
    # Get a list of texts from the dataframe
    texts = df[text_column].tolist()
    
    # Process texts
    processed_texts = []
    iterator = tqdm(texts, desc="Preprocessing texts") if show_progress else texts
    
    for text in iterator:
        processed_text = preprocess_text(
            text, 
            clean=clean, 
            remove_stops=remove_stops,
            lemmatize=lemmatize,
            custom_stopwords=custom_stopwords
        )
        processed_texts.append(processed_text)
    
    # Add processed texts to the dataframe
    df[output_column] = processed_texts
    
    return df


# Financial-specific stopwords that might be irrelevant for sentiment
FINANCIAL_STOPWORDS = [
    'quarter', 'year', 'fiscal', 'company', 'business', 'market',
    'share', 'stock', 'price', 'dividend', 'investment', 'investor',
    'earnings', 'revenue', 'profit', 'loss', 'growth', 'decline',
    'inc', 'incorporated', 'corp', 'corporation', 'ltd', 'limited',
    'llc', 'plc', 'holdings', 'group', 'said', 'reported', 'according'
] 