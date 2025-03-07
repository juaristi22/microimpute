from us_imputation_benchmarking.comparisons.data import preprocess_data
from us_imputation_benchmarking.comparisons.imputations import get_imputations
from us_imputation_benchmarking.comparisons.quantile_loss import compare_quantile_loss
from us_imputation_benchmarking.comparisons.plot import plot_loss_comparison

X, X_test, PREDICTORS, IMPUTED_VARIABLES = preprocess_data(full_data=False)
Y_test = X_test[IMPUTED_VARIABLES]
#data, PREDICTORS, IMPUTED_VARIABLES = preprocess_data(full_data=True)

methods = ["QRF", "OLS", "QuantReg"] # "Matching" still not working
method_imputations = get_imputations(methods, X, X_test, PREDICTORS, IMPUTED_VARIABLES)

loss_comparison_df, quantiles = compare_quantile_loss(Y_test, method_imputations)

print(loss_comparison_df)

plot_loss_comparison(loss_comparison_df, quantiles)