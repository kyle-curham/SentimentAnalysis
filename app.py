"""
Main application entry point for the Sentiment Analysis project.
"""
import argparse
import sys
import logging
from pathlib import Path
import pandas as pd
from datetime import datetime

from src.models import create_model
from src.data_collection.news_scraper import get_company_news_from_newsapi
from src.data_collection.twitter_scraper import get_company_tweets
from src.preprocessing.text_processor import preprocess_dataframe
from src.visualization.sentiment_plots import plot_sentiment_distribution, plot_sentiment_comparison
import config


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def analyze_company_sentiment(company: str, model_name: str = None, preprocess: bool = True,
                             output_dir: str = None, days: int = 7, sources: str = "news") -> pd.DataFrame:
    """Analyze sentiment for a company.
    
    Args:
        company: Company ticker symbol (e.g., AAPL for Apple)
        model_name: Name of the sentiment model to use (default: config.DEFAULT_MODEL)
        preprocess: Whether to preprocess the text before analysis
        output_dir: Directory to save results and plots
        days: Number of days to look back for news/tweets
        sources: Data sources to analyze ("news", "twitter", or "both")
        
    Returns:
        DataFrame with sentiment analysis results
    """
    model_name = model_name or config.DEFAULT_MODEL
    output_dir = output_dir or config.DATA_DIR / "results"
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    
    # Initialize empty DataFrame to store all data
    all_data = pd.DataFrame()
    
    # Fetch and analyze news articles
    if sources in ["news", "both"]:
        logger.info(f"Fetching news about {company} for the last {days} days")
        news_df = get_company_news_from_newsapi(company, days=days)
        
        if news_df.empty or len(news_df) == 0:
            logger.warning(f"No news found for {company}")
        else:
            logger.info(f"Found {len(news_df)} articles for {company}")
            
            # Add to our full dataset
            all_data = pd.concat([all_data, news_df], ignore_index=True)
            
    # Fetch and analyze tweets
    if sources in ["twitter", "both"]:
        logger.info(f"Fetching tweets about {company} for the last {days} days")
        tweets_df = get_company_tweets(company, days=days)
        
        if tweets_df.empty or len(tweets_df) == 0:
            logger.warning(f"No tweets found for {company}")
        else:
            logger.info(f"Found {len(tweets_df)} tweets for {company}")
            
            # Add to our full dataset
            all_data = pd.concat([all_data, tweets_df], ignore_index=True)
    
    # If no data found from any source, return empty DataFrame
    if all_data.empty:
        logger.warning(f"No data found for {company} from specified sources")
        return pd.DataFrame()
    
    # Preprocess text if requested
    if preprocess:
        logger.info("Preprocessing article text")
        all_data = preprocess_dataframe(
            all_data,
            text_column='text',
            new_column_name='processed_text',
            clean=True,
            remove_stops=True,
            lemmatize=False
        )
        text_column = 'processed_text'
    else:
        text_column = 'text'
    
    # Initialize sentiment model
    logger.info(f"Initializing sentiment model: {model_name}")
    model = create_model(model_name)
    
    # Analyze sentiment
    logger.info("Analyzing sentiment")
    result_df = model.analyze_dataframe(all_data, text_column=text_column)
    
    # Add company information
    result_df['company'] = company
    
    # Make sure we have a date column
    if 'date' not in result_df.columns:
        result_df['date'] = datetime.now().strftime('%Y-%m-%d')
    
    # Save results if output directory is specified
    if output_dir:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        result_path = Path(output_dir) / f"{company}_sentiment_{timestamp}.csv"
        logger.info(f"Saving results to {result_path}")
        result_df.to_csv(result_path, index=False)
        
        # Generate and save visualization
        logger.info("Generating visualizations")
        fig = plot_sentiment_distribution(
            result_df,
            sentiment_column='sentiment_label',
            title=f"Sentiment Distribution for {company}"
        )
        fig_path = Path(output_dir) / f"{company}_sentiment_dist_{timestamp}.png"
        fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    
    return result_df


def analyze_multiple_companies(companies: list, model_name: str = None, preprocess: bool = True,
                              output_dir: str = None, days: int = 7, sources: str = "news") -> dict:
    """Analyze sentiment for multiple companies and compare them.
    
    Args:
        companies: List of company names or ticker symbols
        model_name: Name of the model to use
        preprocess: Whether to preprocess the text
        output_dir: Directory to save results
        days: Number of days to look back for news
        sources: Data sources to analyze ("news", "twitter", or "both")
        
    Returns:
        Dictionary mapping company names to sentiment results
    """
    results = {}
    all_results_df = pd.DataFrame()
    
    # Create output directory if specified
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
    else:
        output_path = None
    
    # Analyze each company
    for company in companies:
        logger.info(f"Analyzing sentiment for {company}")
        company_df = analyze_company_sentiment(
            company,
            model_name=model_name,
            preprocess=preprocess,
            output_dir=output_dir,
            days=days,
            sources=sources
        )
        
        if not company_df.empty:
            results[company] = {
                'positive': company_df['sentiment_positive'].mean(),
                'negative': company_df['sentiment_negative'].mean(),
                'neutral': company_df['sentiment_neutral'].mean()
            }
            all_results_df = pd.concat([all_results_df, company_df])
    
    # Generate comparison visualization if we have results and an output directory
    if results and output_path:
        logger.info("Generating comparison visualization")
        fig = plot_sentiment_comparison(
            results,
            title="Sentiment Comparison Across Companies"
        )
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        fig_path = output_path / f"company_sentiment_comparison_{timestamp}.png"
        fig.savefig(fig_path, dpi=300, bbox_inches='tight')
        
        # Save all results to a single file
        all_results_path = output_path / f"all_companies_sentiment_{timestamp}.csv"
        all_results_df.to_csv(all_results_path, index=False)
    
    return results


def main() -> None:
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(description="Financial Sentiment Analysis")
    parser.add_argument("--companies", nargs="+", required=True, help="Company ticker symbols to analyze")
    parser.add_argument("--model", default=None, choices=["finbert", "vader"], help="Sentiment model to use")
    parser.add_argument("--preprocess", action="store_true", help="Preprocess text before analysis")
    parser.add_argument("--output", default=None, help="Output directory for results and plots")
    parser.add_argument("--days", type=int, default=7, help="Number of days to look back for news/tweets")
    parser.add_argument("--sources", default="news", choices=["news", "twitter", "both"], 
                       help="Data sources to analyze (news, twitter, or both)")
    
    args = parser.parse_args()
    
    # Create output directory if specified
    output_dir = args.output
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
    else:
        output_dir = config.DATA_DIR / "results"
        output_dir.mkdir(exist_ok=True, parents=True)
    
    # If only one company is provided, analyze it
    if len(args.companies) == 1:
        company = args.companies[0]
        analyze_company_sentiment(
            company, 
            model_name=args.model, 
            preprocess=args.preprocess, 
            output_dir=output_dir,
            days=args.days,
            sources=args.sources
        )
    
    # If multiple companies are provided, analyze them all and generate a comparison
    else:
        results = analyze_multiple_companies(
            args.companies, 
            model_name=args.model, 
            preprocess=args.preprocess, 
            output_dir=output_dir,
            days=args.days,
            sources=args.sources
        )
        
        # Save combined results
        if results:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            all_results_df = pd.concat(list(results.values()), ignore_index=True)
            output_file = output_dir / f"all_companies_sentiment_{timestamp}.csv"
            all_results_df.to_csv(output_file, index=False)
            
            # Generate comparison plot
            logger.info("Generating comparison visualization")
            fig = plot_sentiment_comparison(results)
            fig_path = output_dir / f"sentiment_comparison_{timestamp}.png"
            fig.savefig(fig_path, dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    main() 