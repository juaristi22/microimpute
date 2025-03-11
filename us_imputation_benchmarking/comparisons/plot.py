import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from typing import List, Dict, Type, Union, Optional, Tuple
from us_imputation_benchmarking.config import QUANTILES, PLOT_CONFIG


def plot_loss_comparison(
    loss_comparison_df: pd.DataFrame,
    quantiles: List[float] = QUANTILES,
    save_path: Optional[str] = None,
) -> go.Figure:
    """
    Plot a bar chart comparing quantile losses across different methods.

    :param loss_comparison_df: DataFrame containing loss comparison data.
    :type loss_comparison_df: pd.DataFrame
    :param quantiles: List of quantile values (e.g. [0.05, 0.1, ...]).
    :type quantiles: List[float]
    :param save_path: Path to save the plot. If None, the plot is displayed.
    :type save_path: Optional[str]
    :returns: Plotly figure object
    :rtype: go.Figure
    """
    fig = px.bar(
        loss_comparison_df,
        x="Percentile",
        y="Loss",
        color="Method",
        barmode="group",
        title="Test Loss Across Quantiles for Different Imputation Methods",
        labels={"Percentile": "Percentiles", "Loss": "Average Test Quantile Loss"},
    )
    
    # Update layout for better appearance
    fig.update_layout(
        title_font_size=14,
        xaxis_title_font_size=12,
        yaxis_title_font_size=12,
        legend_title="Method",
        height=PLOT_CONFIG["height"],
        width=PLOT_CONFIG["width"],
    )
    
    # Add grid lines on y-axis
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)')
    
    # Save or show the plot
    if save_path:
        try:
            fig.write_image(save_path)
            html_path = save_path.replace(".png", ".html").replace(".jpg", ".html")
            fig.write_html(html_path)
            print(f"Plot saved to {save_path} and {html_path}")
        except Exception as e:
            print(f"Error saving plot: {e}")
    
    return fig
