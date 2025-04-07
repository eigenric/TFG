# tests/test_imputation_methods.py
import numpy as np
import pytest
from pampaneira_imputation import imputation_methods as im
from typing import Optional, Tuple

# Existing tests ...

def test_impute_median_sample_wise(sample_3d_data_with_nans):
    """Tests if impute_median_sample_wise correctly imputes using sample-wise median."""
    data = sample_3d_data_with_nans.copy() # Use copy

    # --- DEBUG: Check input data for this test ---
    print("\n--- DEBUG: test_impute_median_sample_wise ---")
    print(f"Data shape received: {data.shape}")
    is_feat1_all_nan_input = np.isnan(data[1, :, 1]).all()
    print(f"Is feature 1 all NaN in input data? {is_feat1_all_nan_input}")
    # --- END DEBUG ---

    imputed = im.impute_median_sample_wise(data)

    assert data.shape == imputed.shape
    assert not np.isnan(imputed).any()

    # --- Specific Assertion: Check imputation for a sample and feature ---
    sample_idx_check = 2  # Example sample index
    feature_idx_check = 0 # Example feature index

    original_sample_feature_slice = data[sample_idx_check, :, feature_idx_check]
    expected_median = np.nanmedian(original_sample_feature_slice)

    # Find indices where NaNs were originally in this slice
    nan_step_indices = np.argwhere(np.isnan(original_sample_feature_slice)).flatten()

    # Check if imputed values at these positions match the expected median
    imputed_values_at_nan_positions = imputed[sample_idx_check, nan_step_indices, feature_idx_check]

    # Assert that imputed values are approximately equal to the expected median
    if not np.isnan(expected_median):
        assert np.allclose(imputed_values_at_nan_positions, expected_median, atol=1e-6), (
            f"Imputation for sample {sample_idx_check}, feature {feature_idx_check} incorrect. "
            f"Expected median: {expected_median}, Imputed values: {imputed_values_at_nan_positions}"
        )
    else:
        assert np.all(imputed_values_at_nan_positions == 0.0), (
            "Expected 0 imputation for all NaN sample feature slice, but got: "
            f"{imputed_values_at_nan_positions}"
        )

    # --- DEBUG: Check feature 1 values after imputation ---
    print(f"Feature 1 values *after* imputation (sample-wise): {imputed[1, :, 1]}")
    # --- END DEBUG ---

def test_impute_mean_sample_wise(sample_3d_data_with_nans):
    """Tests if impute_mean_sample_wise correctly imputes using sample-wise mean."""
    data = sample_3d_data_with_nans.copy() # Use copy

    # --- DEBUG: Check input data for this test ---
    print("\n--- DEBUG: test_impute_mean_sample_wise ---")
    print(f"Data shape received: {data.shape}")
    is_feat1_all_nan_input = np.isnan(data[1, :, 1]).all()
    print(f"Is feature 1 all NaN in input data? {is_feat1_all_nan_input}")
    # --- END DEBUG ---

    imputed = im.impute_mean_sample_wise(data)

    assert data.shape == imputed.shape
    assert not np.isnan(imputed).any()

    # --- Specific Assertion: Check imputation for a sample and feature ---
    sample_idx_check = 2  # Example sample index
    feature_idx_check = 0 # Example feature index

    original_sample_feature_slice = data[sample_idx_check, :, feature_idx_check]
    expected_mean = np.nanmean(original_sample_feature_slice)

    # Find indices where NaNs were originally in this slice
    nan_step_indices = np.argwhere(np.isnan(original_sample_feature_slice)).flatten()

    # Check if imputed values at these positions match the expected mean
    imputed_values_at_nan_positions = imputed[sample_idx_check, nan_step_indices, feature_idx_check]

    # Assert that imputed values are approximately equal to the expected mean
    if not np.isnan(expected_mean):
        assert np.allclose(imputed_values_at_nan_positions, expected_mean, atol=1e-6), (
            f"Imputation for sample {sample_idx_check}, feature {feature_idx_check} incorrect. "
            f"Expected mean: {expected_mean}, Imputed values: {imputed_values_at_nan_positions}"
        )
    else:
        assert np.all(imputed_values_at_nan_positions == 0.0), (
            "Expected 0 imputation for all NaN sample feature slice, but got: "
            f"{imputed_values_at_nan_positions}"
        )

    # --- DEBUG: Check feature 1 values after imputation ---
    print(f"Feature 1 values *after* imputation (sample-wise mean): {imputed[1, :, 1]}")
    # --- END DEBUG ---


def test_impute_linear(sample_3d_data_with_nans):
    data = sample_3d_data_with_nans.copy()
    imputed = im.impute_linear(data)
    assert data.shape == imputed.shape
    assert not np.isnan(imputed).any()
    # Check interpolation: Sample 0, feature 0, step 7 should be between step 4 and step 10 values
    val_step4 = data[0, 4, 0]
    val_step10 = data[0, 10, 0]
    if not np.isnan(val_step4) and not np.isnan(val_step10):
         # Linear interpolation for step 7 (index 7) between indices 4 and 10
         expected_val = val_step4 + (val_step10 - val_step4) * (7 - 4) / (10 - 4)
         assert imputed[0, 7, 0] == pytest.approx(expected_val)
    # Check if start/end NaNs were filled (e.g., sample 3, step 0)
    assert not np.isnan(imputed[3, 0, 0])

def test_impute_forward_backward(sample_3d_data_with_nans):
    data = sample_3d_data_with_nans.copy()
    imputed_fb, imputed_bf = im.impute_forward_backward(data)
    assert data.shape == imputed_fb.shape
    assert data.shape == imputed_bf.shape
    assert not np.isnan(imputed_fb).any()
    assert not np.isnan(imputed_bf).any()

    # Check ffill: Sample 0, feature 0, step 5 should take value from step 4
    if not np.isnan(data[0, 4, 0]):
        assert imputed_fb[0, 5, 0] == data[0, 4, 0]
        # Check bfill for the same method after ffill
        # Sample 3, step 0, feature 0 should take value from step 1 after bfill
        if not np.isnan(imputed_fb[3, 1, 0]): # Check value *after* ffill/bfill pass
             assert imputed_fb[3, 0, 0] == imputed_fb[3, 1, 0]

    # Check bfill first: Sample 0, feature 0, step 9 should take value from step 10
    if not np.isnan(data[0, 10, 0]):
         assert imputed_bf[0, 9, 0] == data[0, 10, 0]

def test_impute_median_all_nan_feature():
    """Tests if impute_median correctly handles an all-NaN feature."""
    test_data = np.random.rand(5, 10, 3) # samples, steps, features
    test_data[:, :, 1] = np.nan # Make feature 1 all NaN
    imputed = im.impute_median(test_data)
    assert not np.isnan(imputed).any() # Check all NaNs are gone
    assert np.all(imputed[:, :, 1] == 0.0) # Explicitly check feature 1 is 0.0

def test_impute_mean_all_nan_feature():
    """Tests if impute_mean correctly handles an all-NaN feature."""
    test_data = np.random.rand(5, 10, 3) # samples, steps, features
    test_data[:, :, 1] = np.nan # Make feature 1 all NaN
    imputed = im.impute_mean(test_data)
    assert not np.isnan(imputed).any() # Check all NaNs are gone
    assert np.all(imputed[:, :, 1] == 0.0) # Explicitly check feature 1 is 0.0