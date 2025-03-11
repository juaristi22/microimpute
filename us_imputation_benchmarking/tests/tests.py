from us_imputation_benchmarking.comparisons.data import preprocess_data
from us_imputation_benchmarking.comparisons.imputations import get_imputations
from us_imputation_benchmarking.comparisons.quantile_loss import (
    compare_quantile_loss,
)
from us_imputation_benchmarking.comparisons.plot import plot_loss_comparison
from us_imputation_benchmarking.models.qrf import QRF
from us_imputation_benchmarking.models.ols import OLS
from us_imputation_benchmarking.models.quantreg import QuantReg

# from us_imputation_benchmarking.models.matching import Matching  # Not working yet
from us_imputation_benchmarking.evaluations.cross_validation import (
    cross_validate_model,
)
from us_imputation_benchmarking.evaluations.train_test_performance import (
    plot_train_test_performance,
)


X, X_test, PREDICTORS, IMPUTED_VARIABLES = preprocess_data(
    full_data=False, years=2019
)
Y_test = X_test[IMPUTED_VARIABLES]
data, PREDICTORS, IMPUTED_VARIABLES = preprocess_data(full_data=True)

model_classes = [QRF, OLS, QuantReg]  # Matching still not working
method_imputations = get_imputations(
    model_classes, X, X_test, PREDICTORS, IMPUTED_VARIABLES
)

loss_comparison_df = compare_quantile_loss(Y_test, method_imputations)

print(loss_comparison_df)

plot_loss_comparison(loss_comparison_df)

qrf_results = cross_validate_model(QRF, data, PREDICTORS, IMPUTED_VARIABLES)

plot_train_test_performance(
    qrf_results, title="QRF: Train vs Test Performance"
)
