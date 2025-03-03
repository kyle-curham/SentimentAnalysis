#!/usr/bin/env python
"""
Interactive script for exploring sentiment analysis results.
This script provides a simple text-based interface for analyzing sentiment
and comparing results from different models without requiring Jupyter notebooks.
"""
import sys
import os
from pathlib import Path
import pandas as pd
from io import StringIO

# Add parent directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.models import create_model, get_finbert_model, get_vader_model
from src.data_collection.news_scraper import get_company_news_from_newsapi
from src.data_collection.twitter_scraper import get_company_tweets
from src.visualization.sentiment_comparison import (
    compare_models_on_text,
    create_sentiment_dashboard,
    find_controversial_examples
)
import config


def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')


def print_header(title):
    """Print a formatted header."""
    clear_screen()
    print("=" * 80)
    print(f"{title:^80}")
    print("=" * 80)
    print()


def print_menu(options):
    """Print a menu of options."""
    for idx, option in enumerate(options, 1):
        print(f"{idx}. {option}")
    print()


def get_user_choice(prompt, options=None, allow_empty=False):
    """Get user input with validation."""
    while True:
        choice = input(f"{prompt}: ").strip()
        
        if allow_empty and not choice:
            return choice
            
        if options is not None:
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(options):
                    return options[idx]
                else:
                    print(f"Please enter a number between 1 and {len(options)}")
            except ValueError:
                print("Please enter a valid number")
        else:
            return choice


def analyze_direct_text():
    """Analyze text entered directly by the user."""
    print_header("Analyze Direct Text")
    
    # Get text from user
    print("Enter or paste the text you want to analyze:")
    text = input("> ").strip()
    
    if not text:
        print("No text entered. Returning to main menu.")
        input("Press Enter to continue...")
        return
    
    print("\nAnalyzing text with all available models...\n")
    
    # Analyze with all models
    results = compare_models_on_text(text)
    
    # Display results
    print("\nResults:")
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
    
    input("\nPress Enter to return to the main menu...")


def analyze_company():
    """Analyze sentiment for a company."""
    print_header("Analyze Company Sentiment")
    
    # Get company ticker
    company = input("Enter company ticker symbol (e.g., AAPL for Apple): ").strip().upper()
    if not company:
        print("No company entered. Returning to main menu.")
        input("Press Enter to continue...")
        return
    
    # Get data source
    source_options = ["News articles", "Twitter posts", "Both news and Twitter"]
    source_choice = get_user_choice("Select data source", source_options)
    
    # Map choice to API parameter
    source_map = {
        "News articles": "news",
        "Twitter posts": "twitter",
        "Both news and Twitter": "both"
    }
    data_source = source_map[source_choice]
    
    # Get time range
    days = input("Enter number of days to look back (default: 7): ").strip()
    days = int(days) if days and days.isdigit() else 7
    
    # Confirm
    print(f"\nAnalyzing {company} sentiment from {data_source} over the past {days} days.")
    confirm = input("Continue? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Operation cancelled. Returning to main menu.")
        input("Press Enter to continue...")
        return
    
    print(f"\nFetching and analyzing data for {company}...")
    
    # Initialize empty DataFrame
    all_data = pd.DataFrame()
    
    # Fetch data based on source
    if data_source in ["news", "both"]:
        print(f"Fetching news about {company}...")
        try:
            news_df = get_company_news_from_newsapi(company, days=days)
            if news_df.empty or len(news_df) == 0:
                print(f"No news found for {company}")
            else:
                print(f"Found {len(news_df)} articles for {company}")
                all_data = pd.concat([all_data, news_df], ignore_index=True)
        except Exception as e:
            print(f"Error fetching news: {e}")
    
    if data_source in ["twitter", "both"]:
        print(f"Fetching tweets about {company}...")
        try:
            tweets_df = get_company_tweets(company, days=days)
            if tweets_df.empty or len(tweets_df) == 0:
                print(f"No tweets found for {company}")
            else:
                print(f"Found {len(tweets_df)} tweets for {company}")
                all_data = pd.concat([all_data, tweets_df], ignore_index=True)
        except Exception as e:
            print(f"Error fetching tweets: {e}")
    
    # Check if we found any data
    if all_data.empty:
        print(f"No data found for {company} from specified sources.")
        input("Press Enter to return to the main menu...")
        return
    
    # Ask if user wants to compare models
    compare = input("\nCompare multiple sentiment models? (y/n, default: y): ").strip().lower()
    compare_models = compare != 'n'
    
    # Create output directory
    output_dir = Path(config.DATA_DIR) / "sentiment_reports" / company
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nAnalyzing {len(all_data)} texts...")
    
    # Create dashboard
    result_df, dashboard_path = create_sentiment_dashboard(
        all_data,
        text_column='text',
        output_dir=str(output_dir),
        compare_models=compare_models
    )
    
    # Show summary
    if compare_models:
        # Calculate agreement percentage
        agreement = (result_df['vader_label'] == result_df['finbert_label']).mean() * 100
        print(f"\nModel agreement: {agreement:.1f}%")
        
        # Find disagreements
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
    
    # Overall sentiment distribution
    print("\nOverall sentiment distribution:")
    for label in result_df['vader_label'].unique():
        count = len(result_df[result_df['vader_label'] == label])
        percentage = (count / len(result_df)) * 100
        print(f"  {label}: {count} texts ({percentage:.1f}%)")
    
    print(f"\nResults saved to: {dashboard_path}")
    input("\nPress Enter to return to the main menu...")


def analyze_from_file():
    """Analyze sentiment from a CSV file."""
    print_header("Analyze Text from CSV File")
    
    # Get file path
    file_path = input("Enter path to CSV file (must contain a 'text' column): ").strip()
    if not file_path:
        print("No file path entered. Returning to main menu.")
        input("Press Enter to continue...")
        return
    
    # Check if file exists
    if not Path(file_path).exists():
        print(f"Error: File not found: {file_path}")
        input("Press Enter to return to the main menu...")
        return
    
    # Try to read the file
    try:
        df = pd.read_csv(file_path)
        if 'text' not in df.columns:
            print("Error: CSV file must contain a 'text' column.")
            input("Press Enter to return to the main menu...")
            return
    except Exception as e:
        print(f"Error reading file: {e}")
        input("Press Enter to return to the main menu...")
        return
    
    print(f"Successfully loaded {len(df)} texts from {file_path}.")
    
    # Ask if user wants to compare models
    compare = input("\nCompare multiple sentiment models? (y/n, default: y): ").strip().lower()
    compare_models = compare != 'n'
    
    # Create output directory
    file_name = Path(file_path).stem
    output_dir = Path(config.DATA_DIR) / "sentiment_reports" / file_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nAnalyzing {len(df)} texts...")
    
    # Create dashboard
    result_df, dashboard_path = create_sentiment_dashboard(
        df,
        text_column='text',
        output_dir=str(output_dir),
        compare_models=compare_models
    )
    
    # Show summary
    if compare_models:
        # Calculate agreement percentage
        agreement = (result_df['vader_label'] == result_df['finbert_label']).mean() * 100
        print(f"\nModel agreement: {agreement:.1f}%")
        
        # Find disagreements
        print("\nFinding examples where models disagree...")
        disagreements = find_controversial_examples(
            result_df,
            model1_label='vader_label',
            model2_label='finbert_label',
            text_column='text'
        )
        
        if not disagreements.empty:
            print(f"\nFound {len(disagreements)} disagreements between models.")
            
            # Ask if user wants to see examples
            show_disagreements = input("Show disagreement examples? (y/n): ").strip().lower() == 'y'
            
            if show_disagreements:
                print("\nTop 5 most significant disagreements:")
                for i, (_, row) in enumerate(disagreements.head(5).iterrows()):
                    print(f"\n{i+1}. Text: {row['text'][:100]}...")
                    print(f"   VADER:   {row['vader_label']} (pos={row['vader_positive']:.3f}, neg={row['vader_negative']:.3f}, neu={row['vader_neutral']:.3f})")
                    print(f"   FinBERT: {row['finbert_label']} (pos={row['finbert_positive']:.3f}, neg={row['finbert_negative']:.3f}, neu={row['finbert_neutral']:.3f})")
    
    # Overall sentiment distribution
    print("\nOverall sentiment distribution:")
    for label in result_df['vader_label'].unique():
        count = len(result_df[result_df['vader_label'] == label])
        percentage = (count / len(result_df)) * 100
        print(f"  {label}: {count} texts ({percentage:.1f}%)")
    
    print(f"\nResults saved to: {dashboard_path}")
    input("\nPress Enter to return to the main menu...")


def view_sample_texts():
    """View sample texts with their sentiment analysis."""
    print_header("View Sample Texts with Sentiment")
    
    # Create sample texts
    sample_texts = [
        "The company reported strong earnings, beating analyst expectations.",
        "Investors are concerned about the company's declining market share.",
        "The stock price remained unchanged after the announcement.",
        "The company announced major layoffs following disappointing quarterly results.",
        "Despite missing revenue targets, the company managed to increase profits."
    ]
    
    df = pd.DataFrame({
        'id': range(len(sample_texts)),
        'text': sample_texts
    })
    
    # Analyze with both models
    print("Analyzing sample texts with VADER and FinBERT models...\n")
    
    # Analyze with VADER
    vader_model = get_vader_model()
    vader_results = vader_model.predict(df['text'].tolist())
    df['vader_positive'] = [r['positive'] for r in vader_results]
    df['vader_negative'] = [r['negative'] for r in vader_results]
    df['vader_neutral'] = [r['neutral'] for r in vader_results]
    df['vader_label'] = [r['label'] for r in vader_results]
    
    # Analyze with FinBERT
    finbert_model = get_finbert_model()
    finbert_results = finbert_model.predict(df['text'].tolist())
    df['finbert_positive'] = [r['positive'] for r in finbert_results]
    df['finbert_negative'] = [r['negative'] for r in finbert_results]
    df['finbert_neutral'] = [r['neutral'] for r in finbert_results]
    df['finbert_label'] = [r['label'] for r in finbert_results]
    
    # Display results
    for i, row in df.iterrows():
        print(f"Sample {i+1}:")
        print(f"Text: {row['text']}")
        print(f"VADER:   {row['vader_label']} (pos={row['vader_positive']:.3f}, neg={row['vader_negative']:.3f}, neu={row['vader_neutral']:.3f})")
        print(f"FinBERT: {row['finbert_label']} (pos={row['finbert_positive']:.3f}, neg={row['finbert_negative']:.3f}, neu={row['finbert_neutral']:.3f})")
        print()
    
    input("Press Enter to return to the main menu...")


def run_tests():
    """Run sentiment comparison tests."""
    print_header("Run Sentiment Comparison Tests")
    
    print("This will run all tests from tests/test_sentiment_comparison.py")
    confirm = input("Continue? (y/n): ").strip().lower()
    
    if confirm != 'y':
        print("Operation cancelled. Returning to main menu.")
        input("Press Enter to continue...")
        return
    
    print("\nRunning tests...\n")
    
    try:
        # Run the tests using unittest directly
        import unittest
        
        # Import the test case using a more robust approach
        import importlib.util
        import os
        
        # Get the absolute path to the test file
        project_root = Path(__file__).resolve().parent.parent
        test_file_path = os.path.join(project_root, "tests", "test_sentiment_comparison.py")
        
        # Check if the test file exists
        if not os.path.exists(test_file_path):
            print(f"Error: Test file not found at {test_file_path}")
            input("\nPress Enter to return to the main menu...")
            return
            
        # Import the module dynamically
        spec = importlib.util.spec_from_file_location("test_module", test_file_path)
        test_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(test_module)
        
        # Get the test case class from the module
        TestSentimentComparison = getattr(test_module, "TestSentimentComparison")
        
        # Redirect stdout during tests to capture output
        original_stdout = sys.stdout
        test_output = StringIO()
        sys.stdout = test_output
        
        # Run the tests
        suite = unittest.TestLoader().loadTestsFromTestCase(TestSentimentComparison)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        
        # Restore stdout
        sys.stdout = original_stdout
        
        # Print results
        print(test_output.getvalue())
        
        # Print summary
        print("\nTest Results:")
        print(f"Run: {result.testsRun}")
        print(f"Errors: {len(result.errors)}")
        print(f"Failures: {len(result.failures)}")
        
    except Exception as e:
        print(f"\nError running tests: {e}")
        import traceback
        traceback.print_exc()
    
    input("\nPress Enter to return to the main menu...")


def main_menu():
    """Display the main menu and handle user choices."""
    
    while True:
        print_header("Sentiment Analysis Explorer")
        
        options = [
            "Analyze text (enter/paste directly)",
            "Analyze company (news/tweets)",
            "Analyze from CSV file",
            "View sample texts with sentiment",
            "Run sentiment comparison tests",
            "Exit"
        ]
        
        print_menu(options)
        choice = get_user_choice("Enter your choice", options)
        
        if choice == "Analyze text (enter/paste directly)":
            analyze_direct_text()
        elif choice == "Analyze company (news/tweets)":
            analyze_company()
        elif choice == "Analyze from CSV file":
            analyze_from_file()
        elif choice == "View sample texts with sentiment":
            view_sample_texts()
        elif choice == "Run sentiment comparison tests":
            run_tests()
        elif choice == "Exit":
            print("\nThank you for using the Sentiment Analysis Explorer!")
            break


if __name__ == "__main__":
    try:
        main_menu()
    except KeyboardInterrupt:
        print("\n\nProgram interrupted. Exiting...")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback
        traceback.print_exc() 