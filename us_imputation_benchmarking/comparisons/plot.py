import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import List


def plot_loss_comparison(loss_comparison_df: pd.DataFrame, quantiles: List[float]) -> None:
    """
    Plot a bar chart comparing quantile losses across different methods.
    
    :param loss_comparison_df: DataFrame containing loss comparison data.
    :type loss_comparison_df: pd.DataFrame
    :param quantiles: List of quantile values (e.g. [0.05, 0.1, ...]).
    :type quantiles: List[float]
    :returns: None
    :rtype: None
    """
    percentiles: List[str] = [str(int(q * 100)) for q in quantiles]
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(data=loss_comparison_df, x="Percentile", y="Loss", hue="Method", dodge=True)
    plt.xlabel("Percentiles", fontsize=12)
    plt.ylabel("Average Test Quantile Loss", fontsize=12)
    plt.title("Test Loss Across Quantiles for Different Imputation Methods", fontsize=14)
    plt.legend(title="Method")
    ax.set_xticklabels(percentiles)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()