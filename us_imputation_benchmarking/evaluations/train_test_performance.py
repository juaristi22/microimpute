import numpy as np
import pandas as pd
import plotly.graph_objects as go
from typing import List, Dict, Type, Union, Optional, Tuple
from us_imputation_benchmarking.config import PLOT_CONFIG


def plot_train_test_performance(
    results: pd.DataFrame,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (PLOT_CONFIG["width"], PLOT_CONFIG["height"]),
) -> go.Figure:
    """
    Plot the performance comparison between training and testing sets across quantiles.

    :param results: DataFrame with train and test rows, quantiles as columns, and loss values.
    :type results: pd.DataFrame
    :param title: Custom title for the plot. If None, a default title is used.
    :type title: Optional[str]
    :param save_path: Path to save the plot. If None, the plot is displayed.
    :type save_path: Optional[str]
    :param figsize: Figure size as (width, height) in pixels.
    :type figsize: Tuple[int, int]
    :returns: Plotly figure object
    :rtype: go.Figure
    """
    # Convert column names to strings if they are not already
    results.columns = [str(col) for col in results.columns]
    
    # Default title if none provided
    if title is None:
        title = "Average Quantile Loss: Train vs Test"
        
    # Create a new figure
    fig = go.Figure()
    
    # Add bars for training data if present
    if "train" in results.index:
        fig.add_trace(go.Bar(
            x=results.columns,
            y=results.loc["train"],
            name="Train",
            marker_color='rgba(0, 128, 0, 0.7)'  # green with transparency
        ))
    
    # Add bars for test data if present
    if "test" in results.index:
        fig.add_trace(go.Bar(
            x=results.columns,
            y=results.loc["test"],
            name="Test",
            marker_color='rgba(255, 0, 0, 0.7)'  # red with transparency
        ))
    
    # Update layout with title, axis labels, etc.
    fig.update_layout(
        title=title,
        xaxis_title="Quantile",
        yaxis_title="Average Quantile Loss",
        barmode='group',
        width=figsize[0],
        height=figsize[1],
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        ),
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    # Add grid lines
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)')
    
    # Save or show the plot
    if save_path:
        try:
            fig.write_image(save_path)
            # Also save HTML version for interactive viewing
            html_path = save_path.replace(".png", ".html").replace(".jpg", ".html")
            fig.write_html(html_path)
            print(f"Plot saved to {save_path} and {html_path}")
        except Exception as e:
            print(f"Error saving train-test plot: {e}")
    
    return fig
