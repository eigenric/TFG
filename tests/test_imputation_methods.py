# tests/test_imputation_methods.py
import numpy as np
import pytest
from pampaneira_imputation import imputation_methods as im
from typing import Optional, Tuple

# Existing tests ...

def test_impute_median_sample_wise(sample_3d_data_with_nans):
    """Tests if impute_median_sample_wise correctly imputes using sample-wise median."""
    data = sample_3d_data_with_nans.copy()
    imputed = im.impute_median_sample_wise(data)

    assert data.shape == imputed.shape
    assert not np.isnan(imputed).any()

    # Check imputation for a sample and feature
    sample_idx_check = 2
    feature_idx_check = 0

    original_sample_feature_slice = data[sample_idx_check, :, feature_idx_check]
    expected_median = np.nanmedian(original_sample_feature_slice)

    nan_step_indices = np.argwhere(np.isnan(original_sample_feature_slice)).flatten()
    imputed_values_at_nan_positions = imputed[sample_idx_check, nan_step_indices, feature_idx_check]

    if not np.isnan(expected_median):
        assert np.allclose(imputed_values_at_nan_positions, expected_median, atol=1e-6)
    else:
        assert np.all(imputed_values_at_nan_positions == 0.0)

def test_impute_mean_sample_wise(sample_3d_data_with_nans):
    """Tests if impute_mean_sample_wise correctly imputes using sample-wise mean."""
    data = sample_3d_data_with_nans.copy()
    imputed = im.impute_mean_sample_wise(data)

    assert data.shape == imputed.shape
    assert not np.isnan(imputed).any()

    # Check imputation for a sample and feature
    sample_idx_check = 2
    feature_idx_check = 0

    original_sample_feature_slice = data[sample_idx_check, :, feature_idx_check]
    expected_mean = np.nanmean(original_sample_feature_slice)

    nan_step_indices = np.argwhere(np.isnan(original_sample_feature_slice)).flatten()
    imputed_values_at_nan_positions = imputed[sample_idx_check, nan_step_indices, feature_idx_check]

    if not np.isnan(expected_mean):
        assert np.allclose(imputed_values_at_nan_positions, expected_mean, atol=1e-6)
    else:
        assert np.all(imputed_values_at_nan_positions == 0.0)

def test_impute_linear(sample_3d_data_with_nans):
    """Tests if impute_linear correctly performs linear interpolation."""
    data = sample_3d_data_with_nans.copy()
    imputed = im.impute_linear(data)
    
    assert data.shape == imputed.shape
    assert not np.isnan(imputed).any()
    
    # Check interpolation between known values
    val_step4 = data[0, 4, 0]
    val_step10 = data[0, 10, 0]
    if not np.isnan(val_step4) and not np.isnan(val_step10):
        expected_val = val_step4 + (val_step10 - val_step4) * (7 - 4) / (10 - 4)
        assert imputed[0, 7, 0] == pytest.approx(expected_val)
    
    # Check if edge NaNs were filled
    assert not np.isnan(imputed[3, 0, 0])

def test_impute_median_all_nan_feature(sample_3d_data_with_nans):
    """Tests if impute_median_sample_wise correctly handles an all-NaN feature."""
    data = sample_3d_data_with_nans.copy()
    data[:, :, 1] = np.nan  # Make feature 1 all NaN
    
    # Esperamos la advertencia de numpy sobre slice completamente vacío
    with pytest.warns(RuntimeWarning, match="All-NaN slice encountered"):
        imputed = im.impute_median_sample_wise(data)
    
    assert not np.isnan(imputed).any()
    # Cuando una característica está completamente vacía, se imputa con el valor global de la característica
    # que en este caso es 1.0 (el valor por defecto en sample_3d_data_with_nans)
    assert np.all(imputed[:, :, 1] == 1.0)

def test_impute_mean_all_nan_feature(sample_3d_data_with_nans):
    """Tests if impute_mean_sample_wise correctly handles an all-NaN feature."""
    data = sample_3d_data_with_nans.copy()
    data[:, :, 1] = np.nan  # Make feature 1 all NaN
    
    # Esperamos la advertencia de numpy sobre slice vacío
    with pytest.warns(RuntimeWarning, match="Mean of empty slice"):
        imputed = im.impute_mean_sample_wise(data)
    
    assert not np.isnan(imputed).any()
    # Cuando una característica está completamente vacía, se imputa con el valor global de la característica
    # que en este caso es 1.0 (el valor por defecto en sample_3d_data_with_nans)
    assert np.all(imputed[:, :, 1] == 1.0)