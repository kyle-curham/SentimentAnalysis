#!/usr/bin/env python
"""
Script to analyze sentiment and compare predictions with original text content.
"""
import sys
import argparse
import pandas as pd
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.visualization.sentiment_comparison import (
    create_sentiment_dashboard,
    find_controversial_examples,
    compare_models_on_text
)
from src.data_collection.news_scraper import get_company_news_from_newsapi
from src.data_collection.twitter_scraper import get_company_tweets
import config


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze sentiment and compare with original text."
    )
    
    # Input source options
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--company",
        type=str,
        help="Company ticker symbol to analyze (e.g., AAPL)"
    )
    source_group.add_argument(
        "--file",
        type=str,
        help="Path to CSV file with text to analyze (must include 'text' column)"
    )
    source_group.add_argument(
        "--single-text",
        type=str,
        help="Directly provide text to analyze"
    )
    
    # Data source options
    parser.add_argument(
        "--data-source",
        choices=["news", "twitter", "both"],
        default="news",
        help="Source of data to analyze (for company analysis)"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="Number of days to look back for news/tweets"
    )
    
    # Output options
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save results (default: data/sentiment_reports)"
    )
    parser.add_argument(
        "--compare-models",
        action="store_true",
        help="Compare results from multiple sentiment models"
    )
    parser.add_argument(
        "--show-disagreements",
        action="store_true",
        help="Show examples where models disagree (requires --compare-models)"
    )
    
    return parser.parse_args()


def analyze_single_text(text, output_dir=None):
    """Analyze a single text with multiple models."""
    print(f"Analyzing text: '{text[:100]}...'")
    
    # Compare results from all models
    results = compare_models_on_text(text)
    
    # Print results
    print("\nSentiment Analysis Results:")
    print("-" * 50)
    
    for model_name, scores in results.items():
        print(f"\n{model_name.upper()} MODEL:")
        print(f"  Positive: {scores['positive']:.4f}")
        print(f"  Negative: {scores['negative']:.4f}")
        print(f"  Neutral:  {scores['neutral']:.4f}")
        print(f"  Label:    {scores['label']}")
    
    print("\nOriginal Text:")
    print("-" * 50)
    print(text)
    print("-" * 50)
    
    return results


def analyze_from_file(file_path, output_dir=None, compare_models=True, show_disagreements=False):
    """Analyze text from a CSV file."""
    print(f"Reading data from: {file_path}")
    
    # Read CSV file
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)
    
    # Check if text column exists
    if 'text' not in df.columns:
        print("Error: CSV file must contain a 'text' column.")
        sys.exit(1)
    
    print(f"Loaded {len(df)} text samples.")
    
    # Create sentiment dashboard
    result_df, dashboard_path = create_sentiment_dashboard(
        df,
        text_column='text',
        output_dir=output_dir,
        compare_models=compare_models
    )
    
    # Show disagreements if requested
    if show_disagreements and compare_models:
        print("\nFinding examples where models disagree...")
        disagreements = find_controversial_examples(
            result_df,
            model1_label='vader_label',
            model2_label='finbert_label',
            text_column='text'
        )
        
        if not disagreements.empty:
            print(f"\nFound {len(disagreements)} disagreements between models.")
            print("\nTop 3 most significant disagreements:")
            for i, (_, row) in enumerate(disagreements.head(3).iterrows()):
                print(f"\n{i+1}. Text: {row['text'][:100]}...")
                print(f"   VADER:   {row['vader_label']} (pos={row['vader_positive']:.3f}, neg={row['vader_negative']:.3f}, neu={row['vader_neutral']:.3f})")
                print(f"   FinBERT: {row['finbert_label']} (pos={row['finbert_positive']:.3f}, neg={row['finbert_negative']:.3f}, neu={row['finbert_neutral']:.3f})")
    
    print(f"\nResults saved to: {dashboard_path}")
    return result_df, dashboard_path


def analyze_company(company, data_source="news", days=7, output_dir=None, 
                   compare_models=True, show_disagreements=False):
    """Analyze sentiment for a company."""
    print(f"Analyzing sentiment for {company} from {data_source} sources over the past {days} days.")
    
    # Initialize empty DataFrame
    all_data = pd.DataFrame()
    
    # Fetch news if requested
    if data_source in ["news", "both"]:
        print(f"Fetching news about {company}...")
        news_df = get_company_news_from_newsapi(company, days=days)
        
        if news_df.empty or len(news_df) == 0:
            print(f"No news found for {company}")
        else:
            print(f"Found {len(news_df)} articles for {company}")
            all_data = pd.concat([all_data, news_df], ignore_index=True)
    
    # Fetch tweets if requested
    if data_source in ["twitter", "both"]:
        print(f"Fetching tweets about {company}...")
        tweets_df = get_company_tweets(company, days=days)
        
        if tweets_df.empty or len(tweets_df) == 0:
            print(f"No tweets found for {company}")
        else:
            print(f"Found {len(tweets_df)} tweets for {company}")
            all_data = pd.concat([all_data, tweets_df], ignore_index=True)
    
    # Check if we found any data
    if all_data.empty:
        print(f"No data found for {company} from specified sources.")
        sys.exit(1)
    
    print(f"Analyzing {len(all_data)} texts...")
    
    # Create sentiment dashboard
    result_df, dashboard_path = create_sentiment_dashboard(
        all_data,
        text_column='text',
        output_dir=output_dir,
        compare_models=compare_models
    )
    
    # Show disagreements if requested
    if show_disagreements and compare_models:
        print("\nFinding examples where models disagree...")
        disagreements = find_controversial_examples(
            result_df,
            model1_label='vader_label',
            model2_label='finbert_label',
            text_column='text'
        )
        
        if not disagreements.empty:
            print(f"\nFound {len(disagreements)} disagreements between models.")
            print("\nTop 3 most significant disagreements:")
            for i, (_, row) in enumerate(disagreements.head(3).iterrows()):
                print(f"\n{i+1}. Text: {row['text'][:100]}...")
                print(f"   VADER:   {row['vader_label']} (pos={row['vader_positive']:.3f}, neg={row['vader_negative']:.3f}, neu={row['vader_neutral']:.3f})")
                print(f"   FinBERT: {row['finbert_label']} (pos={row['finbert_positive']:.3f}, neg={row['finbert_negative']:.3f}, neu={row['finbert_neutral']:.3f})")
    
    print(f"\nResults saved to: {dashboard_path}")
    return result_df, dashboard_path


def main():
    """Main entry point."""
    args = parse_args()
    
    # Configure output directory
    output_dir = args.output_dir
    
    # Handle different input sources
    if args.single_text:
        analyze_single_text(args.single_text, output_dir)
    elif args.file:
        analyze_from_file(
            args.file,
            output_dir=output_dir,
            compare_models=args.compare_models,
            show_disagreements=args.show_disagreements
        )
    elif args.company:
        analyze_company(
            args.company,
            data_source=args.data_source,
            days=args.days,
            output_dir=output_dir,
            compare_models=args.compare_models,
            show_disagreements=args.show_disagreements
        )


if __name__ == "__main__":
    main() 