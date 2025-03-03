"""
News scraper for financial news articles.
"""
import re
import time
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse
import requests
from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm
import os

import config


class NewsScraperError(Exception):
    """Exception raised for errors in the news scraper."""
    pass


def clean_text(text: str) -> str:
    """Clean text by removing extra whitespace and unwanted characters.
    
    Args:
        text: Text to clean
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
        
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove newlines and tabs
    text = re.sub(r'[\n\t\r]', ' ', text)
    # Remove multiple spaces
    text = re.sub(r' +', ' ', text)
    # Strip whitespace
    text = text.strip()
    
    return text


def extract_article_text(html: str) -> str:
    """Extract the main article text from HTML.
    
    Args:
        html: HTML content of the page
        
    Returns:
        Extracted article text
    """
    if not html:
        return ""
        
    soup = BeautifulSoup(html, 'html.parser')
    
    # Remove unwanted elements
    for element in soup.find_all(['script', 'style', 'nav', 'header', 'footer', 'iframe']):
        element.decompose()
    
    # Try to find the article content using common patterns
    article_selectors = [
        'article', 
        '.article-body', 
        '.article-content',
        '.story-body',
        '.story-content',
        '.post-content',
        '.entry-content',
        '#content-body',
        '[itemprop="articleBody"]'
    ]
    
    for selector in article_selectors:
        article = soup.select_one(selector)
        if article:
            return clean_text(article.get_text())
    
    # If no article content found, try to get the main content
    main = soup.find('main')
    if main:
        return clean_text(main.get_text())
    
    # Fallback to body text
    body = soup.find('body')
    if body:
        return clean_text(body.get_text())
    
    return ""


def fetch_from_newsapi(company: str, days: int = 7, api_key: Optional[str] = None) -> List[Dict[str, Any]]:
    """Fetch news articles about a company from NewsAPI.
    
    Args:
        company: Company name or ticker symbol
        days: Number of days to look back
        api_key: NewsAPI key (if None, uses NEWS_API_KEY from environment)
        
    Returns:
        List of article data dictionaries
        
    Raises:
        NewsScraperError: If the API request fails
    """
    # Try to get API key from environment or parameter
    api_key = api_key or os.getenv("NEWS_API_KEY") or config.NEWS_API_KEY
    
    # If no key found, use the known working key for this app
    if not api_key or api_key == "your_newsapi_key_goes_here":
        # Fallback to hardcoded key for this application
        api_key = "f009f5a85c50406bbfe084c43590ead1"
        print("Using hardcoded API key")
    
    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    # Format dates for API
    from_date = start_date.strftime("%Y-%m-%d")
    to_date = end_date.strftime("%Y-%m-%d")
    
    # Build request URL
    base_url = "https://newsapi.org/v2/everything"
    
    # Parameters for the API request
    params = {
        "q": f"{company} OR ${company}",  # Search for company name or ticker
        "language": "en",
        "sortBy": "relevancy",
        "from": from_date,
        "to": to_date,
        "pageSize": 25,  # Maximum articles per request
        "apiKey": api_key
    }
    
    try:
        # Make the request
        response = requests.get(base_url, params=params)
        response.raise_for_status()  # Raise exception for HTTP errors
        
        # Parse the JSON response
        data = response.json()
        
        # Check for API errors
        if data["status"] != "ok":
            error_msg = data.get("message", "Unknown API error")
            raise NewsScraperError(f"NewsAPI error: {error_msg}")
        
        return data.get("articles", [])
    
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 401:
            raise NewsScraperError(f"Authentication error: Your API key may be invalid or expired. Status code: {e.response.status_code}")
        elif e.response.status_code == 429:
            raise NewsScraperError(f"Rate limit exceeded: You've made too many requests. Status code: {e.response.status_code}")
        else:
            raise NewsScraperError(f"HTTP error: {str(e)}")
    
    except requests.exceptions.RequestException as e:
        raise NewsScraperError(f"Request failed: {str(e)}")
    
    except json.JSONDecodeError:
        raise NewsScraperError("Failed to parse API response as JSON")
    
    except Exception as e:
        raise NewsScraperError(f"Unexpected error: {str(e)}")


def scrape_article(url: str, timeout: int = 10, headers: Optional[Dict[str, str]] = None,
                  retry_count: int = 3, retry_delay: int = 2) -> Dict[str, Any]:
    """Scrape a single news article.
    
    Args:
        url: URL of the article to scrape
        timeout: Request timeout in seconds
        headers: Request headers
        retry_count: Number of times to retry on failure
        retry_delay: Delay between retries in seconds
        
    Returns:
        Dictionary containing article data:
        {
            'url': URL of the article,
            'domain': Domain of the article,
            'title': Title of the article,
            'text': Text content of the article,
            'html': Raw HTML of the article
        }
        
    Raises:
        NewsScraperError: If the article cannot be scraped
    """
    # Default headers to mimic a browser
    if headers is None:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Cache-Control': 'max-age=0',
        }
    
    # Extract domain from URL
    domain = urlparse(url).netloc
    
    # Initialize result dictionary
    result = {
        'url': url,
        'domain': domain,
        'title': "",
        'text': "",
        'html': ""
    }
    
    # Try to scrape the article with retries
    for attempt in range(retry_count):
        try:
            response = requests.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()
            
            # Store HTML
            result['html'] = response.text
            
            # Parse HTML
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract title
            title_tag = soup.find('title')
            if title_tag:
                result['title'] = clean_text(title_tag.get_text())
            
            # Extract text
            result['text'] = extract_article_text(response.text)
            
            return result
            
        except (requests.RequestException, Exception) as e:
            if attempt < retry_count - 1:
                time.sleep(retry_delay)
            else:
                raise NewsScraperError(f"Failed to scrape article at {url}: {str(e)}")
    
    return result


def scrape_articles(urls: List[str], progress_bar: bool = True) -> pd.DataFrame:
    """Scrape multiple news articles.
    
    Args:
        urls: List of URLs to scrape
        progress_bar: Whether to show a progress bar
        
    Returns:
        DataFrame containing article data
    """
    results = []
    
    # Use tqdm for progress bar if requested
    iterator = tqdm(urls, desc="Scraping articles") if progress_bar else urls
    
    for url in iterator:
        try:
            article = scrape_article(url)
            results.append(article)
        except NewsScraperError as e:
            print(f"Error: {str(e)}")
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    return df


def search_news_for_company(company: str, num_results: int = 10, days: int = 7) -> List[str]:
    """Search for news articles about a company using NewsAPI.
    
    Args:
        company: Company name or ticker symbol
        num_results: Maximum number of results to return
        days: Number of days to look back for news
        
    Returns:
        List of article URLs
    """
    try:
        # Get articles from NewsAPI
        articles = fetch_from_newsapi(company, days=days)
        
        # Extract URLs (with fallback for missing URLs)
        urls = [article.get("url") for article in articles if article.get("url")]
        
        # Limit to requested number
        return urls[:num_results]
        
    except NewsScraperError as e:
        print(f"Warning: {str(e)}")
        print("Falling back to alternative sources...")
        
        # Fallback to some financial news sites for the company
        fallback_urls = [
            f"https://seekingalpha.com/symbol/{company}",
            f"https://finance.yahoo.com/quote/{company}",
            f"https://www.marketwatch.com/investing/stock/{company}"
        ]
        
        return fallback_urls[:num_results]


def get_company_news_from_newsapi(company: str, num_results: int = 10, days: int = 7) -> pd.DataFrame:
    """Get news articles about a company from NewsAPI without scraping.
    
    This is a more reliable alternative to search_news_for_company + scrape_articles
    as it doesn't depend on web scraping which might be blocked.
    
    Args:
        company: Company name or ticker symbol
        num_results: Maximum number of results to return
        days: Number of days to look back for news
        
    Returns:
        DataFrame containing article data
    """
    try:
        # Get articles from NewsAPI
        articles = fetch_from_newsapi(company, days=days)
        
        # Limit to requested number
        articles = articles[:num_results]
        
        if not articles:
            print(f"No news found for {company}")
            return pd.DataFrame()
        
        # Convert to DataFrame with consistent column names
        df = pd.DataFrame(articles)
        
        # Rename columns to match our expected format
        if 'title' in df.columns and 'description' in df.columns:
            # Combine title and description for the text field
            df['text'] = df['title'] + ". " + df['description'].fillna("")
            
        if 'url' in df.columns:
            df['domain'] = df['url'].apply(lambda url: urlparse(url).netloc if url else "")
            
        # Add date column in consistent format
        if 'publishedAt' in df.columns:
            df['date'] = pd.to_datetime(df['publishedAt']).dt.strftime('%Y-%m-%d')
        else:
            df['date'] = datetime.now().strftime('%Y-%m-%d')
            
        # Ensure we have the expected columns
        expected_cols = ['url', 'domain', 'title', 'text', 'date']
        for col in expected_cols:
            if col not in df.columns:
                df[col] = ""
                
        return df
        
    except NewsScraperError as e:
        print(f"Error: {str(e)}")
        return pd.DataFrame() 