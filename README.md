# Sentiment Analysis for Publicly Traded Companies

This project analyzes sentiment from textual data related to publicly traded companies, including news articles and social media posts from Twitter/X. The sentiment analysis results can be used to correlate with stock price movements.

## Project Overview

- **Goal**: Classify sentiment (positive, negative, neutral) from text data about companies
- **Data Sources**: 
  - News articles via NewsAPI
  - Tweets from Twitter/X
- **Output**: Sentiment scores and labels for correlation with stock movements
- **Models**: Utilizes pretrained NLP models (FinBERT, BERT, DistilBERT, RoBERTa, VADER)

## Features

- Data collection from multiple sources:
  - News articles via NewsAPI
  - Tweets via Twitter API v2
- Text preprocessing and cleaning
- Sentiment analysis using multiple model options
- Result visualization and export
- Command-line interface for easy analysis
- Model comparison and evaluation tools
- Interactive sentiment explorer
- Fine-tuning tools for FinBERT

## Project Structure

```
sentiment-analysis/
├── data/                 # Storage for datasets and model outputs
├── models/               # Model implementations and fine-tuning code
├── notebooks/            # Jupyter notebooks for exploration and analysis
├── scripts/              # Utility scripts for data collection and processing
│   ├── compare_sentiment.py     # Script for comparing sentiment model results
│   ├── interactive_sentiment_explorer.py  # Interactive console-based explorer
│   ├── finetune_finbert.py      # Script for fine-tuning FinBERT on financial sentiment data
├── src/                  # Core application code
│   ├── data_collection/  # News and Twitter API integrations
│   ├── preprocessing/    # Text cleaning and preparation
│   ├── models/           # Model implementation
│   └── visualization/    # Plotting and dashboard code
│       ├── sentiment_comparison.py  # Tools for comparing sentiment with text
├── tests/                # Test suite
│   ├── test_models.py    # Basic model tests
│   ├── test_sentiment_comparison.py  # Model comparison tests
├── app.py                # Main application entry point
├── config.py             # Configuration settings
├── requirements.txt      # Project dependencies
└── README.md             # Project documentation
```

## Getting Started

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set up environment variables (create a `.env` file with your API keys):
   - News API: Get an API key from [NewsAPI.org](https://newsapi.org/)
   - Twitter API: Get API keys from [Twitter Developer Portal](https://developer.twitter.com/)
4. Run the application: `python app.py --companies AAPL --sources both`

## API Keys

To use this application, you'll need:

1. **NewsAPI Key**: Sign up at [NewsAPI.org](https://newsapi.org/)
2. **Twitter API Keys**: Apply for access at [Twitter Developer Portal](https://developer.twitter.com/)

Add these to your `.env` file:

```
NEWS_API_KEY=your_newsapi_key
TWITTER_API_KEY=your_twitter_api_key
TWITTER_API_SECRET=your_twitter_api_secret
TWITTER_BEARER_TOKEN=your_twitter_bearer_token
TWITTER_ACCESS_TOKEN=your_twitter_access_token
TWITTER_ACCESS_SECRET=your_twitter_access_secret
```

## Model Options

1. **FinBERT**: BERT model pretrained on financial texts (recommended default)
2. **BERT**: Standard bidirectional transformer model
3. **DistilBERT**: Lighter and faster version of BERT
4. **RoBERTa**: Optimized BERT variant with better training
5. **VADER**: Rule-based sentiment analyzer (best for social media)

## Fine-tuning FinBERT

The project includes a script to fine-tune the FinBERT model on financial sentiment datasets. This significantly improves model performance on financial texts:

```bash
# Install additional requirements for fine-tuning
pip install transformers datasets torch tqdm sklearn

# Fine-tune FinBERT using all available datasets
python scripts/finetune_finbert.py --epochs 3 --evaluate

# Fine-tune on specific datasets only
python scripts/finetune_finbert.py --datasets financial_phrasebank twitter_financial_news

# Save the model to a custom location
python scripts/finetune_finbert.py --output_dir ./models/my_finbert_model
```

The fine-tuning script:
1. Downloads and processes public financial sentiment datasets
2. Standardizes labels across datasets
3. Fine-tunes the FinBERT model
4. Evaluates the model performance
5. Saves the model for later use

After fine-tuning, update the `config.py` file to use your fine-tuned model:
```python
AVAILABLE_MODELS = {
    "finbert": "./models/finbert_finetuned/final_model",  # Point to your fine-tuned model
    # ... other models
}
```

## Usage Examples

Basic usage:
```bash
# Analyze news articles only
python app.py --companies AAPL --sources news

# Analyze tweets only
python app.py --companies TSLA --sources twitter

# Analyze both news and tweets
python app.py --companies MSFT GOOG --sources both
```

## Interactive Sentiment Explorer

The project includes a text-based interactive sentiment analysis tool that doesn't require Jupyter notebooks:

```bash
# Launch the interactive explorer
python scripts/interactive_sentiment_explorer.py
```

This provides a menu-driven interface to:
- Analyze text directly (paste any text and get sentiment scores)
- Analyze company news/tweets
- Analyze text from a CSV file
- View sample texts with their sentiment scores
- Run sentiment comparison tests

## Comparing Sentiment Models

To compare how different models analyze the same text:

```bash
# Compare models on a single text
python scripts/compare_sentiment.py --single-text "The company reported strong earnings."

# Compare models on company data
python scripts/compare_sentiment.py --company AAPL --data-source news --compare-models --show-disagreements

# Compare models on your own data
python scripts/compare_sentiment.py --file your_data.csv --compare-models
```

Results include:
- Sentiment scores from multiple models
- Visualization of score distributions
- HTML reports showing text alongside sentiment scores
- Examples where models disagree (useful for evaluating reliability)

For more detailed usage information, see [USAGE.md](USAGE.md).

## License

MIT

## Contributors

Kyle Curham