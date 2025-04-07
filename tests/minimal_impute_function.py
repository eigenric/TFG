# minimal_impute_function.py
import numpy as np
import warnings

def minimal_impute_median_test(data):
    feature_slice = data[:, :, 1] # Feature 1 slice
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        median_val = np.nanmedian(feature_slice)

    if np.isnan(median_val):
        value_to_impute = 0.0
    else:
        value_to_impute = median_val

    nan_mask = np.isnan(data[:, :, 1])
    data[:, :, 1][nan_mask] = value_to_impute # Update in place

    return data

if __name__ == "__main__":
    n_samples = 5
    n_steps = 10
    n_features = 3
    data = np.random.rand(n_samples, n_steps, n_features)
    data[:, :, 1] = np.nan # Make feature 1 all NaN

    print(f"Initial data:\n{data[0, :, 1]}") # Show initial slice

    imputed_data = minimal_impute_median_test(data.copy()) # Use copy

    print(f"\nImputed data feature 1 slice:\n{imputed_data[0, :, 1]}") # Show imputed slice
    print(f"\nIs imputed_data[1, :, 1] all zero? {np.all(imputed_data[:, :, 1] == 0.0)}") # Check if all zero

    # Check for NaNs in the whole array to confirm no general failure
    print(f"\nAre there still NaNs in the whole array? {np.isnan(imputed_data).any()}")