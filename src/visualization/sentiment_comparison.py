"""
Utilities for comparing and visualizing sentiment analysis results with original text content.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Union, Tuple
import os
from pathlib import Path
import numpy as np
from IPython.display import display, HTML

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.models import create_model, get_finbert_model, get_vader_model, get_finetuned_finbert_model
import config


def compare_sentiment_with_text(df: pd.DataFrame, 
                              text_column: str = 'text',
                              sentiment_columns: Optional[List[str]] = None,
                              label_column: Optional[str] = None,
                              model_name: Optional[str] = None,
                              output_dir: Optional[str] = None) -> pd.DataFrame:
    """
    Compare sentiment analysis results with original text.
    
    Args:
        df: DataFrame containing text data and optionally sentiment columns
        text_column: Name of the column containing text content
        sentiment_columns: List of column names containing sentiment scores 
                         (if None, will analyze text with specified model)
        label_column: Column name for sentiment label (if None, will analyze with model)
        model_name: Sentiment model to use if sentiment columns not provided
        output_dir: Directory to save visualization output (if None, won't save)
        
    Returns:
        DataFrame with text and sentiment information
    """
    # Create a copy to avoid modifying the original
    result_df = df.copy()
    
    # If no sentiment info provided, analyze text with the specified model
    if sentiment_columns is None or label_column is None:
        if model_name is None:
            model_name = config.DEFAULT_MODEL
            
        print(f"Analyzing sentiment using {model_name} model...")
        model = create_model(model_name)
        result_df = model.analyze_dataframe(result_df, text_column)
        
        # Set the sentiment columns based on model output
        sentiment_columns = ['sentiment_positive', 'sentiment_negative', 'sentiment_neutral']
        label_column = 'sentiment_label'
    
    # Validate columns
    required_columns = [text_column] + sentiment_columns + [label_column]
    for col in required_columns:
        if col not in result_df.columns:
            raise ValueError(f"Required column '{col}' not found in DataFrame")
    
    # Create a visualization directory if specified
    if output_dir:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
    
    # Return the DataFrame
    return result_df


def plot_sentiment_distribution(df: pd.DataFrame,
                               sentiment_columns: List[str],
                               label_column: str,
                               output_path: Optional[str] = None) -> None:
    """
    Plot sentiment score distribution.
    
    Args:
        df: DataFrame with sentiment scores
        sentiment_columns: List of columns containing sentiment scores
        label_column: Column containing sentiment labels
        output_path: Path to save the plot (if None, won't save)
    """
    # Set up the plot style
    sns.set_style(config.CHART_STYLE)
    plt.figure(figsize=(12, 7))
    
    # Prepare data for plotting
    # Melt the dataframe to get sentiment scores in a format suitable for seaborn
    melted_df = pd.melt(
        df,
        id_vars=[label_column],
        value_vars=sentiment_columns,
        var_name='Sentiment Type', 
        value_name='Score'
    )
    
    # Clean up the sentiment type labels (remove prefixes if any)
    melted_df['Sentiment Type'] = melted_df['Sentiment Type'].str.replace('sentiment_', '')
    melted_df['Sentiment Type'] = melted_df['Sentiment Type'].str.replace('vader_', '')
    melted_df['Sentiment Type'] = melted_df['Sentiment Type'].str.replace('finbert_', '')
    
    # Plot distribution by sentiment type
    sns.boxplot(
        x='Sentiment Type', 
        y='Score', 
        hue=label_column,
        data=melted_df,
        palette=config.DEFAULT_COLORS
    )
    
    plt.title('Distribution of Sentiment Scores by Predicted Label')
    plt.xlabel('Sentiment Type')
    plt.ylabel('Score')
    plt.tight_layout()
    
    # Save if output path provided
    if output_path:
        plt.savefig(output_path)
        print(f"Plot saved to {output_path}")
    
    plt.show()


def create_text_sentiment_report(df: pd.DataFrame, 
                               text_column: str,
                               sentiment_columns: List[str],
                               label_column: str,
                               num_examples: int = 10,
                               output_path: Optional[str] = None) -> Union[str, HTML]:
    """
    Create a report comparing text content with sentiment scores.
    
    Args:
        df: DataFrame with text and sentiment data
        text_column: Column containing the text content
        sentiment_columns: Columns containing sentiment scores
        label_column: Column containing sentiment labels
        num_examples: Number of examples to include in the report
        output_path: Path to save HTML report (if None, won't save)
        
    Returns:
        HTML report or path to saved report
    """
    # Sample rows from each sentiment category
    samples = []
    labels = df[label_column].unique()
    
    for label in labels:
        label_df = df[df[label_column] == label]
        if len(label_df) > 0:
            # Sample (with replacement if needed)
            sample_size = min(num_examples // len(labels), len(label_df))
            if sample_size > 0:
                samples.append(label_df.sample(sample_size))
    
    # Combine samples
    if samples:
        sample_df = pd.concat(samples, ignore_index=True)
    else:
        sample_df = df.sample(min(num_examples, len(df)))
    
    # Sort by label for better readability
    sample_df = sample_df.sort_values(by=label_column)
    
    # Create HTML report
    html = ["<style>",
           "table { border-collapse: collapse; width: 100%; }",
           "th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }",
           "th { background-color: #f2f2f2; }",
           "tr:nth-child(even) { background-color: #f9f9f9; }",
           ".positive { background-color: rgba(102, 194, 165, 0.2); }",
           ".negative { background-color: rgba(255, 107, 107, 0.2); }",
           ".neutral { background-color: rgba(44, 142, 204, 0.2); }",
           "</style>",
           "<h2>Text Sentiment Comparison</h2>"]
    
    # Add each example to the report
    for i, row in sample_df.iterrows():
        text = row[text_column]
        label = row[label_column]
        
        # Format sentiment scores
        scores_html = "<table><tr><th>Sentiment Type</th><th>Score</th></tr>"
        for col in sentiment_columns:
            # Extract sentiment type from column name
            sentiment_type = col.replace('sentiment_', '').replace('vader_', '').replace('finbert_', '')
            scores_html += f"<tr><td>{sentiment_type}</td><td>{row[col]:.4f}</td></tr>"
        scores_html += "</table>"
        
        # Add example to report with appropriate styling
        html.append(f"<div class='{label}'>")
        html.append(f"<h3>Example #{i+1} (Label: {label})</h3>")
        html.append(f"<p>{text}</p>")
        html.append("<h4>Sentiment Scores:</h4>")
        html.append(scores_html)
        html.append("</div><hr>")
    
    # Combine into single HTML document
    html_report = "\n".join(html)
    
    # Save if output path provided
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_report)
        print(f"HTML report saved to {output_path}")
        return output_path
    
    # Otherwise return as HTML for display in notebooks
    return HTML(html_report)


def compare_models_on_text(text: str, 
                         models: Optional[List[str]] = None) -> Dict[str, Dict[str, float]]:
    """
    Compare multiple sentiment models on a single text.
    
    Args:
        text: Text to analyze
        models: List of model names to use (if None, uses all available models)
        
    Returns:
        Dictionary of model results
    """
    if models is None:
        # Make sure the finetuned model is included by default
        models = ["finbert_finetuned", "finbert", "vader"]
    
    results = {}
    
    for model_name in models:
        model = create_model(model_name)
        results[model_name] = model.analyze_text(text)
    
    return results


def create_sentiment_dashboard(df: pd.DataFrame,
                             text_column: str = 'text',
                             output_dir: Optional[str] = None,
                             compare_models: bool = True) -> Tuple[pd.DataFrame, str]:
    """
    Create a comprehensive sentiment analysis dashboard.
    
    Args:
        df: DataFrame containing text data
        text_column: Column containing text content
        output_dir: Directory to save output files
        compare_models: Whether to compare multiple models
        
    Returns:
        Tuple of (result_dataframe, dashboard_path)
    """
    # Set up the output directory
    if output_dir is None:
        output_dir = str(Path("data") / "sentiment_reports" / "dashboard")
    
    os.makedirs(output_dir, exist_ok=True)
    
    result_df = df.copy()
    
    # Analyze text with the finetuned FinBERT model
    finbert_finetuned = get_finetuned_finbert_model()
    print(f"Analyzing {len(df)} texts with finetuned FinBERT model...")
    finbert_result = finbert_finetuned.analyze_dataframe(
        result_df, 
        text_column,
        prefix="finbert_finetuned"
    )
    
    # Merge results
    result_df = pd.concat([result_df, finbert_result], axis=1)
    
    # If comparing models, also analyze with original FinBERT and VADER
    if compare_models:
        # VADER analysis
        vader = get_vader_model()
        print(f"Analyzing {len(df)} texts with VADER model...")
        vader_result = vader.analyze_dataframe(
            result_df, 
            text_column,
            prefix="vader"
        )
        
        # Original FinBERT analysis
        finbert_original = get_finbert_model()
        print(f"Analyzing {len(df)} texts with original FinBERT model...")
        finbert_orig_result = finbert_original.analyze_dataframe(
            result_df, 
            text_column,
            prefix="finbert"
        )
        
        # Merge all results
        result_df = pd.concat([result_df, vader_result, finbert_orig_result], axis=1)
        
        # Set up all sentiment columns for visualization
        all_sentiment_columns = []
        
        for prefix in ["finbert_finetuned", "finbert", "vader"]:
            all_sentiment_columns.extend([
                f"{prefix}_positive", 
                f"{prefix}_negative", 
                f"{prefix}_neutral"
            ])
        
        # Create distribution plot
        plot_path = os.path.join(output_dir, "sentiment_distribution.png")
        plot_sentiment_distribution(
            result_df,
            sentiment_columns=all_sentiment_columns,
            label_column="finbert_finetuned_label",  # Use finetuned model as reference
            output_path=plot_path
        )
        
        # Create text report
        report_path = os.path.join(output_dir, "text_sentiment_report.html")
        create_text_sentiment_report(
            result_df,
            text_column=text_column,
            sentiment_columns=all_sentiment_columns,
            label_column="finbert_finetuned_label",  # Use finetuned model as reference
            output_path=report_path
        )
    else:
        # Just use the finetuned model results for visualization
        sentiment_columns = [
            "finbert_finetuned_positive", 
            "finbert_finetuned_negative", 
            "finbert_finetuned_neutral"
        ]
        
        # Create distribution plot
        plot_path = os.path.join(output_dir, "sentiment_distribution.png")
        plot_sentiment_distribution(
            result_df,
            sentiment_columns=sentiment_columns,
            label_column="finbert_finetuned_label",
            output_path=plot_path
        )
        
        # Create text report
        report_path = os.path.join(output_dir, "text_sentiment_report.html")
        create_text_sentiment_report(
            result_df,
            text_column=text_column,
            sentiment_columns=sentiment_columns,
            label_column="finbert_finetuned_label",
            output_path=report_path
        )
    
    # Save results to CSV
    result_path = os.path.join(output_dir, "sentiment_results.csv")
    result_df.to_csv(result_path, index=False)
    
    print(f"Dashboard created at: {output_dir}")
    print(f"- Distribution plot: {plot_path}")
    print(f"- Text report: {report_path}")
    print(f"- Results CSV: {result_path}")
    
    return result_df, output_dir


def find_controversial_examples(df: pd.DataFrame,
                              model1_label: str,
                              model2_label: str,
                              text_column: str) -> pd.DataFrame:
    """
    Find examples where models disagree significantly.
    
    Args:
        df: DataFrame with sentiment analysis results from multiple models
        model1_label: Column name for first model's labels
        model2_label: Column name for second model's labels
        text_column: Column containing the text content
        
    Returns:
        DataFrame with examples where models disagree
    """
    # Find disagreements
    disagreements = df[df[model1_label] != df[model2_label]].copy()
    
    if disagreements.empty:
        print("No disagreements found between models.")
        return pd.DataFrame()
    
    # Calculate disagreement severity (optional)
    if f"{model1_label.replace('_label', '_positive')}" in df.columns:
        # Extract model prefixes
        prefix1 = model1_label.replace('_label', '')
        prefix2 = model2_label.replace('_label', '')
        
        # Calculate disagreement score as sum of absolute differences
        disagreements['disagreement_score'] = 0
        for sentiment in ['positive', 'negative', 'neutral']:
            col1 = f"{prefix1}_{sentiment}"
            col2 = f"{prefix2}_{sentiment}"
            if col1 in df.columns and col2 in df.columns:
                disagreements['disagreement_score'] += abs(
                    disagreements[col1] - disagreements[col2]
                )
        
        # Sort by disagreement score
        disagreements = disagreements.sort_values('disagreement_score', ascending=False)
    
    # Return disagreements
    return disagreements 