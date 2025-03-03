"""
Data collection module for gathering financial text data.
"""
from src.data_collection.news_scraper import scrape_article, scrape_articles, search_news_for_company

__all__ = [
    'scrape_article',
    'scrape_articles',
    'search_news_for_company',
] 