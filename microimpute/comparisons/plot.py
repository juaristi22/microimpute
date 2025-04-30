"""Visualization utilities for imputation results comparison.

It supports creating bar charts for comparing quantile loss
across different imputation methods and quantiles.
"""

import logging
import os
from typing import List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pydantic import validate_call

from microimpute.config import PLOT_CONFIG, QUANTILES, VALIDATE_CONFIG

logger = logging.getLogger(__name__)


@validate_call(config=VALIDATE_CONFIG)
def plot_loss_comparison(
    loss_comparison_df: pd.DataFrame,
    quantiles: List[float] = QUANTILES,
    save_path: Optional[str] = None,
) -> go.Figure:
    """Plot a bar chart comparing quantile losses across different methods.

    Args:
        loss_comparison_df: DataFrame containing loss comparison data.
        quantiles: List of quantile values (e.g. [0.05, 0.1, ...]).
        save_path: Path to save the plot. If None, the plot is displayed.

    Returns:
        Plotly figure object

    Raises:
        ValueError: If input DataFrame is invalid or missing required columns
        RuntimeError: If plot creation or saving fails
    """
    logger.debug(
        f"Creating loss comparison plot with DataFrame of shape "
        f"{loss_comparison_df.shape}"
    )

    # Validate inputs
    required_columns = ["Method", "Imputed Variable", "Percentile", "Loss"]
    missing_columns = [
        col
        for col in required_columns
        if col not in loss_comparison_df.columns
    ]
    if missing_columns:
        logger.error(
            f"Missing required columns for plotting: {missing_columns}"
        )
        raise ValueError(f"DataFrame must contain columns: {required_columns}")

    if quantiles is None or len(quantiles) == 0:
        logger.warning(
            "Empty quantiles list provided, using default QUANTILES"
        )
        quantiles = QUANTILES

    try:
        logger.debug("Creating bar chart with plotly express")
        df_avg = loss_comparison_df[
            loss_comparison_df["Imputed Variable"] == "average"
        ]
        df_regular = df_avg[df_avg["Percentile"] != "average"]
        df_average = df_avg[df_avg["Percentile"] == "average"]
        fig = px.bar(
            df_regular,
            x="Percentile",
            y="Loss",
            color="Method",
            barmode="group",
            title="Test Loss Across Quantiles for Different Imputation Methods",
            labels={
                "Percentile": "Percentiles",
                "Loss": "Average Test Quantile Loss",
            },
        )

        # Add a horizontal line for each method using their average value
        if not df_average.empty:
            logger.debug("Adding average loss lines to plot")

            # Create a mapping for method names to their corresponding color in the plot
            method_colors = {
                method: px.colors.qualitative.Plotly[
                    i % len(px.colors.qualitative.Plotly)
                ]
                for i, method in enumerate(df_regular["Method"].unique())
            }

            for method, data in df_average.groupby("Method"):
                avg_loss = data["Loss"].values[0]

                fig.add_shape(
                    type="line",
                    x0=-0.5,  # Start slightly before the first bar
                    y0=avg_loss,
                    x1=len(df_regular["Percentile"].unique())
                    - 0.5,  # End slightly after the last bar
                    y1=avg_loss,
                    line=dict(
                        color=method_colors.get(method),
                        width=2,
                        dash="dot",
                    ),
                    name=f"{method} Mean",
                )

        logger.debug("Updating plot layout")
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
        fig.update_yaxes(
            showgrid=True, gridwidth=1, gridcolor="rgba(0,0,0,0.1)"
        )

        # Save or show the plot
        if save_path:
            try:
                logger.info(f"Saving plot to {save_path}")

                # Ensure directory exists
                save_dir = os.path.dirname(save_path)
                if save_dir and not os.path.exists(save_dir):
                    logger.debug(f"Creating directory: {save_dir}")
                    os.makedirs(save_dir, exist_ok=True)

                # Save as image
                fig.write_image(save_path)

                # Also save as HTML for interactive viewing
                html_path = save_path.replace(".jpg", ".html")
                fig.write_html(html_path)

                logger.info(f"Plot saved to {save_path} and {html_path}")
            except Exception as e:
                logger.error(f"Error saving plot: {str(e)}")
                raise RuntimeError(
                    f"Failed to save plot to {save_path}"
                ) from e

        logger.debug("Plot creation completed successfully")
        return fig

    except Exception as e:
        logger.error(f"Error creating loss comparison plot: {str(e)}")
        raise RuntimeError(
            f"Failed to create loss comparison plot: {str(e)}"
        ) from e


@validate_call(config=VALIDATE_CONFIG)
def plot_autoimpute_method_comparison(
    method_results_df: pd.DataFrame,
    quantiles: List[float] = QUANTILES,
    metric_name: str = "Quantile Loss",
    save_path: Optional[str] = None,
) -> go.Figure:
    """Plot a bar chart comparing losses across different imputation methods.

    Args:
        method_results_df: DataFrame with methods as index and quantiles as
            columns, plus an optional 'mean_loss' column.
        quantiles: List of quantile values (e.g., [0.05, 0.1, ...]).
        metric_name: Name of the metric being compared (default: "Quantile
            Loss").
        save_path: Path to save the plot. If None, the plot is displayed.

    Returns:
        Plotly figure object

    Raises:
        ValueError: If input DataFrame is invalid or missing required columns
        RuntimeError: If plot creation or saving fails
    """
    logger.debug(
        f"Creating method comparison plot with DataFrame of shape {method_results_df.shape}"
    )

    # Validate inputs
    if method_results_df.empty:
        logger.error("Empty DataFrame provided for plotting")
        raise ValueError("DataFrame cannot be empty")

    # Check if the DataFrame structure is valid (quantiles as columns)
    expected_columns = [str(q) for q in quantiles]
    if not all(
        str(q) in method_results_df.columns or q in method_results_df.columns
        for q in quantiles
    ):
        logger.warning(
            f"Some quantiles not found in DataFrame columns. "
            f"Expected: {expected_columns}, Found: {list(method_results_df.columns)}"
        )

    try:
        # Convert DataFrame to long format for plotting
        plot_df = method_results_df.reset_index().rename(
            columns={"index": "Method"}
        )

        # Melt the DataFrame to get it in the right format for plotting
        id_vars = ["Method"]
        value_vars = [
            col
            for col in plot_df.columns
            if col not in id_vars and col != "mean_loss"
        ]

        melted_df = pd.melt(
            plot_df,
            id_vars=id_vars,
            value_vars=value_vars,
            var_name="Percentile",
            value_name=metric_name,
        )

        # Convert percentile column to string if it's not already
        melted_df["Percentile"] = melted_df["Percentile"].astype(str)

        # Create the bar chart
        logger.debug("Creating bar chart with plotly express")
        fig = px.bar(
            melted_df,
            x="Percentile",
            y=metric_name,
            color="Method",
            barmode="group",
            title=f"{metric_name} Across Quantiles for Different Imputation Methods",
            labels={
                "Percentile": "Quantiles",
                metric_name: f"Test {metric_name}",
            },
        )

        # Add a horizontal line for the mean loss if present
        if "mean_loss" in method_results_df.columns:
            logger.debug("Adding mean loss markers to plot")
            for i, method in enumerate(method_results_df.index):
                mean_loss = method_results_df.loc[method, "mean_loss"]
                fig.add_shape(
                    type="line",
                    x0=-0.5,
                    y0=mean_loss,
                    x1=len(quantiles) - 0.5,
                    y1=mean_loss,
                    line=dict(
                        color=px.colors.qualitative.Plotly[
                            i % len(px.colors.qualitative.Plotly)
                        ],
                        width=2,
                        dash="dot",
                    ),
                    name=f"{method} Mean",
                )

        logger.debug("Updating plot layout")
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
        fig.update_yaxes(
            showgrid=True, gridwidth=1, gridcolor="rgba(0,0,0,0.1)"
        )

        # Save or show the plot
        if save_path:
            try:
                logger.info(f"Saving plot to {save_path}")

                # Ensure directory exists
                save_dir = os.path.dirname(save_path)
                if save_dir and not os.path.exists(save_dir):
                    logger.debug(f"Creating directory: {save_dir}")
                    os.makedirs(save_dir, exist_ok=True)

                # Save as image
                fig.write_image(save_path)

                # Also save as HTML for interactive viewing
                html_path = save_path.replace(".jpg", ".html").replace(
                    ".png", ".html"
                )
                fig.write_html(html_path)

                logger.info(f"Plot saved to {save_path} and {html_path}")
            except Exception as e:
                logger.error(f"Error saving plot: {str(e)}")
                raise RuntimeError(
                    f"Failed to save plot to {save_path}"
                ) from e

        logger.debug("Plot creation completed successfully")
        return fig

    except Exception as e:
        logger.error(f"Error creating method comparison plot: {str(e)}")
        raise RuntimeError(
            f"Failed to create method comparison plot: {str(e)}"
        ) from e
