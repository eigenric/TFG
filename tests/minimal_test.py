# minimal_test.py
import numpy as np
# Assuming imputation_methods is accessible (might need path adjustment or running from root)
from pampaneira_imputation import imputation_methods as im

print(f"NumPy version: {np.__version__}")

# Create data similar to fixture
n_samples = 10
n_steps = 24 # Or use config.N_STEPS
n_features = 4
rng = np.random.default_rng(42)
data = rng.random((n_samples, n_steps, n_features))
data[1, :, 1] = np.nan # Feature 1 all NaN

print(f"Initial: Is data[1, :, 1] all NaN? {np.isnan(data[1, :, 1]).all()}")

# Run the specific imputation function
imputed_data = im.impute_median(data) # Or impute_mean

print(f"After imputation: Is imputed_data[1, :, 1] all zero? {np.all(imputed_data[1, :, 1] == 0.0)}")
if not np.all(imputed_data[1, :, 1] == 0.0):
     print(f"Values are: {imputed_data[1, :, 1]}")