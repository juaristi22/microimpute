"""
Test the autoimpute function.
"""

import pandas as pd
from sklearn.datasets import load_diabetes

from microimpute.comparisons.autoimpute import autoimpute
from microimpute.comparisons.plot import plot_loss_comparison

diabetes = load_diabetes()
diabetes_donor = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
diabetes_receiver = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)

predictors = ["age", "sex", "bmi", "bp"]
imputed_variables = ["s1", "s4"]

imputations, fitted_model, method_results_df = autoimpute(
    donor_data=diabetes_donor,
    receiver_data=diabetes_receiver,
    predictors=predictors,
    imputed_variables=imputed_variables,
)

# plot_loss_comparison(
#    method_results_df, save_path="autoimpute_model_comparison.jpg"
# )

# imputations.to_csv("autoimpute_best_imputations.csv")
