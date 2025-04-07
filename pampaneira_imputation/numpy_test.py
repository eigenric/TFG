# numpy_test.py
import numpy as np
from pampaneira_imputation import imputation_methods as im # Adjust path if needed

print(f"NumPy version: {np.__version__}")

n_samples = 10
n_steps = 24
n_features = 4
data = np.ones((n_samples, n_steps, n_features))
data[1, :, 1] = np.nan

print(f"Initial: Is data[1, :, 1] all NaN? {np.isnan(data[1, :, 1]).all()}")

imputed_median_data = im.impute_median(data.copy()) # Use copy here
print("\n--- impute_median results ---")
print(f"Is imputed_median_data[1, :, 1] all zero? {np.all(imputed_median_data[1, :, 1] == 0.0)}")
print(f"Feature 1 values (median imputed): {imputed_median_data[1, :, 1]}")

imputed_mean_data = im.impute_mean(data.copy()) # Use copy here
print("\n--- impute_mean results ---")
print(f"Is imputed_mean_data[1, :, 1] all zero? {np.all(imputed_mean_data[1, :, 1] == 0.0)}")
print(f"Feature 1 values (mean imputed): {imputed_mean_data[1, :, 1]}")