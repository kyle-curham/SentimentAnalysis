"""
Twitter scraper for fetching tweets about companies for sentiment analysis.
"""
import os
import time
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import pandas as pd
import tweepy
from tqdm import tqdm

import config

class TwitterScraperError(Exception):
    """Exception raised for errors in the Twitter scraper."""
    pass


def authenticate_twitter() -> tweepy.Client:
    """Authenticate with Twitter API v2 using bearer token.
    
    Returns:
        Authenticated tweepy Client
        
    Raises:
        TwitterScraperError: If authentication fails
    """
    # Get the bearer token (preferred for search queries)
    bearer_token = os.getenv("TWITTER_BEARER_TOKEN") or config.TWITTER_BEARER_TOKEN
    
    if not bearer_token:
        # Fallback to API key/secret if bearer token isn't available
        api_key = os.getenv("TWITTER_API_KEY") or config.TWITTER_API_KEY
        api_secret = os.getenv("TWITTER_API_SECRET") or config.TWITTER_API_SECRET
        access_token = os.getenv("TWITTER_ACCESS_TOKEN") or config.TWITTER_ACCESS_TOKEN
        access_secret = os.getenv("TWITTER_ACCESS_SECRET") or config.TWITTER_ACCESS_SECRET
        
        if not all([api_key, api_secret, access_token, access_secret]):
            raise TwitterScraperError("Twitter API credentials not found. Please check your .env file.")
        
        try:
            client = tweepy.Client(
                consumer_key=api_key,
                consumer_secret=api_secret,
                access_token=access_token,
                access_token_secret=access_secret
            )
            return client
        except Exception as e:
            raise TwitterScraperError(f"Twitter authentication failed: {str(e)}")
    
    try:
        client = tweepy.Client(bearer_token=bearer_token)
        return client
    except Exception as e:
        raise TwitterScraperError(f"Twitter authentication failed: {str(e)}")


def search_recent_tweets(query: str, max_results: int = 100, days: int = 7) -> List[Dict[str, Any]]:
    """Search for recent tweets matching the query.
    
    Args:
        query: Search query string (e.g., "AAPL" or "Apple")
        max_results: Maximum number of tweets to return
        days: Number of days to look back (limited by Twitter API)
        
    Returns:
        List of tweet data dictionaries
        
    Raises:
        TwitterScraperError: If the API request fails
    """
    # Twitter API v2 only allows 7 days of search with recent search endpoint
    days = min(days, 7)
    
    client = authenticate_twitter()
    
    # Set up search parameters
    tweet_fields = ['created_at', 'text', 'public_metrics', 'lang']
    
    # Fix query syntax for Twitter API v2
    # The query syntax is different from what we're using in the NewsAPI
    query = f"{query} lang:en -is:retweet"
    
    try:
        # Search for tweets
        response = client.search_recent_tweets(
            query=query,
            tweet_fields=tweet_fields,
            max_results=min(max_results, 100),  # Twitter API limit is 100 per request
            expansions=['author_id']
        )
        
        if not response.data:
            # No tweets found
            return []
        
        # Process tweets
        tweets = []
        for tweet in response.data:
            # Get the author info
            author = next((user for user in response.includes['users'] if user.id == tweet.author_id), None)
            
            tweets.append({
                'id': tweet.id,
                'text': tweet.text,
                'created_at': tweet.created_at,
                'author_id': tweet.author_id,
                'username': author.username if author else None,
                'like_count': tweet.public_metrics['like_count'],
                'retweet_count': tweet.public_metrics['retweet_count'],
                'reply_count': tweet.public_metrics['reply_count'],
                'lang': tweet.lang
            })
        
        return tweets
        
    except tweepy.TweepyException as e:
        error_msg = str(e)
        if "401" in error_msg:
            raise TwitterScraperError("Twitter authentication failed. Check your API credentials.")
        elif "429" in error_msg:
            raise TwitterScraperError("Twitter API rate limit exceeded. Try again later.")
        else:
            raise TwitterScraperError(f"Twitter API error: {error_msg}")
    
    except Exception as e:
        raise TwitterScraperError(f"Unexpected error: {str(e)}")


def get_company_tweets(company: str, max_results: int = 50, days: int = 7) -> pd.DataFrame:
    """Get tweets about a company and return as a DataFrame.
    
    Args:
        company: Company name or ticker symbol
        max_results: Maximum number of tweets to fetch
        days: Number of days to look back
        
    Returns:
        DataFrame containing tweets with processed content for sentiment analysis
    """
    # Use proper query syntax for Twitter API v2
    query = company
    
    try:
        # Get tweets from Twitter API
        tweets = search_recent_tweets(query, max_results=max_results, days=days)
        
        if not tweets:
            # No tweets found
            print(f"No tweets found for {company}")
            return pd.DataFrame()
        
        # Create dataframe
        df = pd.DataFrame(tweets)
        
        # Convert created_at to datetime
        df['date'] = pd.to_datetime(df['created_at']).dt.date
        
        # Add source and company columns
        df['source'] = 'twitter'
        df['company'] = company
        
        # Rename columns to match news dataframe format for compatibility
        df = df.rename(columns={
            'text': 'content',
            'id': 'id',
            'created_at': 'published_at'
        })
        
        # Select columns that are necessary for sentiment analysis
        columns = ['id', 'content', 'date', 'source', 'company', 'published_at']
        df = df[columns]
        
        return df
        
    except TwitterScraperError as e:
        print(f"Error: {str(e)}")
        return pd.DataFrame()
        
    except Exception as e:
        print(f"Unexpected error fetching tweets: {str(e)}")
        return pd.DataFrame()


def get_multiple_companies_tweets(companies: List[str], max_results_per_company: int = 50, 
                                days: int = 7) -> pd.DataFrame:
    """Get tweets for multiple companies and combine into one DataFrame.
    
    Args:
        companies: List of company names or ticker symbols
        max_results_per_company: Maximum number of tweets per company
        days: Number of days to look back
        
    Returns:
        Combined DataFrame of tweets for all companies
    """
    all_tweets = []
    
    for company in companies:
        df = get_company_tweets(company, max_results=max_results_per_company, days=days)
        if not df.empty:
            all_tweets.append(df)
            
    if not all_tweets:
        return pd.DataFrame()
        
    return pd.concat(all_tweets, ignore_index=True) 