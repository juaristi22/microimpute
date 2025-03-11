import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Dict, Type, Union, Optional, Tuple


def plot_train_test_performance(
    results: pd.DataFrame,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
) -> None:
    """
    Plot the performance comparison between training and testing sets across quantiles.

    :param results: DataFrame with train and test rows, quantiles as columns, and loss values.
    :type results: pd.DataFrame
    :param title: Custom title for the plot. If None, a default title is used.
    :type title: Optional[str]
    :param save_path: Path to save the plot. If None, the plot is displayed.
    :type save_path: Optional[str]
    :param figsize: Figure size as (width, height) in inches.
    :type figsize: Tuple[int, int]
    :returns: None
    :rtype: None
    """
    plt.figure(figsize=figsize)

    # Convert column names to strings if they are not already
    results.columns = [str(col) for col in results.columns]

    # Plot as bars
    width = 0.35  # width of the bars
    x = np.arange(len(results.columns))
    if "train" in results.index:
        plt.bar(
            x - width / 2,
            results.loc["train"],
            width,
            label="Train",
            color="green",
            alpha=0.7,
        )
    if "test" in results.index:
        plt.bar(
            x + width / 2,
            results.loc["test"],
            width,
            label="Test",
            color="red",
            alpha=0.7,
        )

    # Add labels, title, and legend
    plt.xlabel("Quantile")
    plt.ylabel("Average Quantile Loss")
    if title is None:
        title = "Average Quantile Loss: Train vs Test"
    plt.title(title)
    plt.xticks(x, results.columns, rotation=45)
    plt.legend()
    plt.tight_layout()

    # Save or show the plot
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
