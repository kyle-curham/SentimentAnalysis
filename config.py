"""
Configuration settings for the Sentiment Analysis project.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"

# Create directories if they don't exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR]:
    directory.mkdir(exist_ok=True)

# API keys and credentials (should be stored in .env file)
TWITTER_API_KEY = os.getenv("TWITTER_API_KEY", "")
TWITTER_API_SECRET = os.getenv("TWITTER_API_SECRET", "")
TWITTER_ACCESS_TOKEN = os.getenv("TWITTER_ACCESS_TOKEN", "")
TWITTER_ACCESS_SECRET = os.getenv("TWITTER_ACCESS_SECRET", "")
TWITTER_BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN", "")
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")

# Model settings
DEFAULT_MODEL = "finbert_finetuned"  # Changed default model to our finetuned version
AVAILABLE_MODELS = {
    "finbert_finetuned": "models/finbert_finetuned/final",  # Path to our finetuned model
    "finbert": "yiyanghkust/finbert-tone",
    "bert": "bert-base-uncased",
    "distilbert": "distilbert-base-uncased",
    "roberta": "roberta-base",
    "vader": "vader"  # VADER doesn't use a model path
}

# Data collection settings
DEFAULT_NEWS_SOURCES = [
    "reuters.com",
    "bloomberg.com",
    "cnbc.com",
    "wsj.com",
    "ft.com",
    "seekingalpha.com"
]

# Processing settings
MAX_TEXT_LENGTH = 512  # Maximum token length for transformer models
BATCH_SIZE = 16
NUM_WORKERS = 4

# Visualization settings
CHART_STYLE = "darkgrid"
DEFAULT_COLORS = ["#2C8ECC", "#FF6B6B", "#66C2A5"]  # Blue for positive, red for negative, green for neutral 