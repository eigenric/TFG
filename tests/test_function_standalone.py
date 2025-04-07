# test_function_standalone.py
import numpy as np
from pampaneira_imputation import imputation_methods as im # Adjust path if needed
import warnings

def run_test_impute_median(): # Just the content of test_impute_median
    print("\n--- DEBUG: test_impute_median (Standalone) ---")
    n_samples = 10
    n_steps = 24
    n_features = 4
    rng = np.random.default_rng(42)
    data = rng.random((n_samples, n_steps, n_features))
    data[1, :, 1] = np.nan

    print(f"Data shape created in standalone test: {data.shape}")
    is_feat1_all_nan_input = np.isnan(data[1, :, 1]).all()
    print(f"Standalone test: Is feature 1 all NaN in input data? {is_feat1_all_nan_input}")

    imputed = im.impute_median(data)
    assert data.shape == imputed.shape
    assert not np.isnan(imputed).any()
    print(f"Feature 1 values *after* imputation (standalone): {imputed[1, :, 1]}")
    assert np.all(imputed[:, :, 1] == 0)
    print("Standalone test_impute_median PASSED (locally in script)") # Mark pass if it reaches here

if __name__ == "__main__":
    run_test_impute_median() # Run the function directly
    print("\nStandalone script finished.") # Mark end