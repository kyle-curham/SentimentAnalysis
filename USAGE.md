# Sentiment Analysis Usage Guide

This guide explains how to use the Sentiment Analysis system for analyzing financial texts related to publicly traded companies.

## Setup

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Set up API keys by creating a `.env` file with the following:
   ```
   # NewsAPI Credentials
   NEWS_API_KEY=your_newsapi_key_from_newsapi_org
   
   # Twitter API Credentials
   TWITTER_API_KEY=your_twitter_api_key
   TWITTER_API_SECRET=your_twitter_api_secret
   TWITTER_BEARER_TOKEN=your_twitter_bearer_token
   TWITTER_ACCESS_TOKEN=your_twitter_access_token
   TWITTER_ACCESS_SECRET=your_twitter_access_token_secret
   ```

## Obtaining API Keys

### NewsAPI
1. Sign up for a free account at [NewsAPI.org](https://newsapi.org/register)
2. Copy your API key from the dashboard
3. Add it to your `.env` file

### Twitter API
1. Apply for a developer account at [Twitter Developer Portal](https://developer.twitter.com/)
2. Create a new project and app
3. Generate API keys, bearer token, and access tokens
4. Add them to your `.env` file

## Command-Line Interface

The main application provides a command-line interface for analyzing sentiment:

```bash
python app.py --companies AAPL MSFT GOOGL --model finbert --sources both
```

### Available Arguments

- `--companies`: One or more company ticker symbols to analyze (required)
- `--model`: Sentiment model to use (optional, default: finbert)
  - Options: finbert, bert, distilbert, roberta, vader
- `--output`: Directory to save results (optional, default: ./data/results)
- `--preprocess`: Enable text preprocessing (optional, default: True)
- `--days`: Number of days to look back for news and tweets (optional, default: 7)
- `--sources`: Data sources to analyze (optional, default: "news")
  - Options: news, twitter, both

### Examples

Analyze a single company's news using the default FinBERT model:
```bash
python app.py --companies AAPL
```

Analyze both news and tweets for multiple companies using VADER (good for social media):
```bash
python app.py --companies AAPL MSFT GOOGL --model vader --sources both
```

Analyze only tweets for a company:
```bash
python app.py --companies TSLA --model vader --sources twitter
```

Look back 14 days for news articles:
```bash
python app.py --companies NVDA --days 14
```

Specify an output directory:
```bash
python app.py --companies TSLA --output ./results/tesla_analysis
```

## Python API

You can also use the system programmatically in your own Python code:

```python
# For news articles
from src.data_collection.news_scraper import get_company_news_from_newsapi
from src.models import create_model
from src.preprocessing import preprocess_dataframe
import pandas as pd

# Collect news data
company = "AAPL"
news_df = get_company_news_from_newsapi(company, days=7)

# Preprocess data
processed_df = preprocess_dataframe(
    news_df,
    text_column='content',
    new_column_name='processed_text'
)

# Analyze sentiment
model = create_model("vader")  # or "finbert", etc.
results_df = model.analyze_dataframe(processed_df, 'processed_text')

# Work with results
positive_articles = results_df[results_df['sentiment_label'] == 'positive']
avg_positive_score = results_df['sentiment_positive'].mean()
print(f"Average positive sentiment: {avg_positive_score:.2f}")
```

For Twitter data:
```python
from src.data_collection.twitter_scraper import get_company_tweets

# Get tweets about a company
tweets_df = get_company_tweets("AAPL", max_results=100, days=7)

# Process tweets the same way as news articles
# ... rest of the analysis code is the same
```

## Jupyter Notebooks

For interactive exploration, check out the Jupyter notebooks in the `notebooks/` directory:

- `sentiment_analysis_demo.ipynb`: Demonstrates basic usage of the system
- Add your own notebooks for custom analyses

## Advanced Usage

### Using Different Models

The system supports several sentiment analysis models:

- **FinBERT**: Specialized for financial texts (recommended default)
- **BERT**: Standard bidirectional transformer model
- **DistilBERT**: Lighter and faster version of BERT
- **RoBERTa**: Optimized BERT variant
- **VADER**: Rule-based lexicon (great for social media)

You can select models using the factory function:

```python
from src.models import create_model

# Create any supported model
finbert_model = create_model("finbert")
vader_model = create_model("vader")
```

### Comparing News and Social Media Sentiment

One powerful use case is comparing sentiment between formal news sources and social media:

```python
from src.data_collection.news_scraper import get_company_news_from_newsapi
from src.data_collection.twitter_scraper import get_company_tweets
from src.models import create_model
import pandas as pd

company = "TSLA"

# Get data from both sources
news_df = get_company_news_from_newsapi(company, days=7)
tweets_df = get_company_tweets(company, days=7)

# Add source labels
news_df['source_type'] = 'news'
tweets_df['source_type'] = 'twitter'

# Combine data
combined_df = pd.concat([news_df, tweets_df], ignore_index=True)

# Analyze sentiment
model = create_model("vader")
results_df = model.analyze_dataframe(combined_df, 'content')

# Compare sentiment by source
news_sentiment = results_df[results_df['source_type'] == 'news']['sentiment_positive'].mean()
twitter_sentiment = results_df[results_df['source_type'] == 'twitter']['sentiment_positive'].mean()

print(f"News sentiment: {news_sentiment:.3f}")
print(f"Twitter sentiment: {twitter_sentiment:.3f}")
```

### Customizing Preprocessing

You can customize text preprocessing:

```python
from src.preprocessing import preprocess_dataframe, FINANCIAL_STOPWORDS

# Add custom financial stopwords
custom_stopwords = FINANCIAL_STOPWORDS + ['additional', 'custom', 'words']

# Apply custom preprocessing
processed_df = preprocess_dataframe(
    df,
    text_column='content',
    new_column_name='processed_text',
    clean=True,           # Apply basic cleaning (lowercase, remove punctuation)
    remove_stops=True,    # Remove stopwords
    lemmatize=True,       # Lemmatize words
    custom_stopwords=custom_stopwords
)
```

### Visualizing Results

The visualization module provides several ways to visualize sentiment:

```python
from src.visualization import (
    plot_sentiment_distribution,
    plot_sentiment_comparison
)

# Plot sentiment distribution
fig = plot_sentiment_distribution(
    results_df,
    sentiment_column='sentiment_label',
    title='Sentiment Distribution'
)

# Compare multiple companies
fig = plot_sentiment_comparison(
    company_results_dict,  # Dictionary mapping company names to result dataframes
    title='Company Sentiment Comparison'
)

# Save visualization
fig.savefig('sentiment_plot.png', dpi=300, bbox_inches='tight')
```

## Troubleshooting

- **API rate limits**: Both NewsAPI and Twitter API have rate limits. For NewsAPI free tier, you're limited to 100 requests per day.
- **Twitter API errors**: Ensure your query syntax is correct and that you have proper authentication.
- **Model download issues**: If you have issues downloading models, check your internet connection and ensure you have enough disk space.
- **Memory errors**: Transformer models require significant memory. If you encounter memory issues, try using a smaller model like DistilBERT or VADER.
- **NLTK resource errors**: If you encounter NLTK errors, manually download resources with:
  ```python
  import nltk
  nltk.download('punkt')
  nltk.download('stopwords')
  nltk.download('wordnet')
  ``` 