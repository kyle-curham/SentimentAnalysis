"""
Visualization module for sentiment analysis results.
"""
from typing import Dict, List, Any, Optional, Union, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path

import config


def setup_plot_style() -> None:
    """Set up the plot style for consistency."""
    sns.set_style(config.CHART_STYLE)
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 12


def plot_sentiment_distribution(
    data: pd.DataFrame,
    sentiment_column: str = "sentiment_label",
    title: str = "Sentiment Distribution",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
    colors: Optional[List[str]] = None
) -> plt.Figure:
    """Plot the distribution of sentiment labels.
    
    Args:
        data: DataFrame with sentiment data
        sentiment_column: Column containing sentiment labels
        title: Plot title
        save_path: Path to save the plot (if None, plot is not saved)
        figsize: Figure size as (width, height) in inches
        colors: Custom colors for the bars
        
    Returns:
        Matplotlib Figure object
    """
    # Set plot style
    setup_plot_style()
    
    # Count sentiment labels
    sentiment_counts = data[sentiment_column].value_counts()
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Use default colors if not specified
    if colors is None:
        colors = config.DEFAULT_COLORS
    
    # Create bar plot
    sentiment_counts.plot(
        kind='bar',
        ax=ax,
        color=colors[:len(sentiment_counts)],
        edgecolor='black',
        linewidth=1
    )
    
    # Set labels and title
    ax.set_title(title, fontsize=16, pad=20)
    ax.set_xlabel('Sentiment', fontsize=14)
    ax.set_ylabel('Count', fontsize=14)
    
    # Add count labels on top of bars
    for i, count in enumerate(sentiment_counts):
        ax.text(
            i,
            count + (max(sentiment_counts) * 0.02),
            str(count),
            ha='center',
            fontsize=12
        )
    
    # Add percentage labels inside bars
    total = sum(sentiment_counts)
    for i, count in enumerate(sentiment_counts):
        percentage = 100 * count / total
        ax.text(
            i,
            count / 2,
            f"{percentage:.1f}%",
            ha='center',
            fontsize=12,
            color='white',
            fontweight='bold'
        )
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_sentiment_time_series(
    data: pd.DataFrame,
    date_column: str,
    sentiment_columns: List[str],
    title: str = "Sentiment Over Time",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6),
    colors: Optional[List[str]] = None,
    rolling_window: Optional[int] = None
) -> plt.Figure:
    """Plot sentiment scores over time.
    
    Args:
        data: DataFrame with sentiment and date data
        date_column: Column containing dates
        sentiment_columns: Columns containing sentiment scores
        title: Plot title
        save_path: Path to save the plot (if None, plot is not saved)
        figsize: Figure size as (width, height) in inches
        colors: Custom colors for the lines
        rolling_window: Window size for rolling average (if None, no rolling average)
        
    Returns:
        Matplotlib Figure object
    """
    # Set plot style
    setup_plot_style()
    
    # Sort data by date
    if pd.api.types.is_datetime64_any_dtype(data[date_column]):
        plot_data = data.sort_values(date_column)
    else:
        plot_data = data.copy()
        plot_data[date_column] = pd.to_datetime(plot_data[date_column])
        plot_data = plot_data.sort_values(date_column)
    
    # Set index to date for easier plotting
    plot_data = plot_data.set_index(date_column)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Use default colors if not specified
    if colors is None:
        colors = config.DEFAULT_COLORS
    
    # Plot each sentiment score
    for i, column in enumerate(sentiment_columns):
        if column not in plot_data.columns:
            continue
            
        # Apply rolling average if specified
        if rolling_window:
            values = plot_data[column].rolling(rolling_window).mean()
            label = f"{column} ({rolling_window}-day avg)"
        else:
            values = plot_data[column]
            label = column
            
        # Plot the line
        ax.plot(
            plot_data.index,
            values,
            label=label,
            color=colors[i % len(colors)],
            linewidth=2
        )
    
    # Set labels and title
    ax.set_title(title, fontsize=16, pad=20)
    ax.set_xlabel('Date', fontsize=14)
    ax.set_ylabel('Sentiment Score', fontsize=14)
    
    # Add grid and legend
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_sentiment_comparison(
    data: Dict[str, Dict[str, float]],
    title: str = "Sentiment Comparison",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8),
    colors: Optional[List[str]] = None
) -> plt.Figure:
    """Plot a comparison of sentiment between different entities (e.g., companies).
    
    Args:
        data: Dictionary mapping entity names to dictionaries of sentiment scores
              Example: {'Company A': {'positive': 0.6, 'negative': 0.2, 'neutral': 0.2}}
        title: Plot title
        save_path: Path to save the plot (if None, plot is not saved)
        figsize: Figure size as (width, height) in inches
        colors: Custom colors for the bars
        
    Returns:
        Matplotlib Figure object
    """
    # Set plot style
    setup_plot_style()
    
    # Convert dictionary to DataFrame for easier plotting
    entities = list(data.keys())
    sentiments = list(data[entities[0]].keys())
    
    # Create DataFrame from the input dict
    df_data = {sentiment: [data[entity][sentiment] for entity in entities] 
               for sentiment in sentiments}
    df = pd.DataFrame(df_data, index=entities)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Use default colors if not specified
    if colors is None:
        colors = config.DEFAULT_COLORS
    
    # Create bar plot
    df.plot(
        kind='bar',
        ax=ax,
        color=colors[:len(sentiments)],
        edgecolor='black',
        linewidth=1
    )
    
    # Set labels and title
    ax.set_title(title, fontsize=16, pad=20)
    ax.set_xlabel('Entity', fontsize=14)
    ax.set_ylabel('Sentiment Score', fontsize=14)
    
    # Add grid and legend
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(fontsize=12)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def create_sentiment_heatmap(
    data: pd.DataFrame,
    x_column: str,
    y_column: str,
    value_column: str,
    title: str = "Sentiment Heatmap",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8),
    cmap: str = "RdYlGn"
) -> plt.Figure:
    """Create a heatmap of sentiment scores.
    
    Args:
        data: DataFrame with sentiment data
        x_column: Column to use for x-axis labels
        y_column: Column to use for y-axis labels
        value_column: Column containing sentiment values
        title: Plot title
        save_path: Path to save the plot (if None, plot is not saved)
        figsize: Figure size as (width, height) in inches
        cmap: Colormap to use (RdYlGn is red-yellow-green)
        
    Returns:
        Matplotlib Figure object
    """
    # Set plot style
    setup_plot_style()
    
    # Pivot data to create a matrix suitable for a heatmap
    pivot_data = data.pivot_table(
        index=y_column,
        columns=x_column,
        values=value_column,
        aggfunc='mean'
    )
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    sns.heatmap(
        pivot_data,
        ax=ax,
        cmap=cmap,
        linewidths=0.5,
        annot=True,
        fmt=".2f",
        cbar_kws={"shrink": 0.8, "label": "Sentiment Score"}
    )
    
    # Set labels and title
    ax.set_title(title, fontsize=16, pad=20)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig 