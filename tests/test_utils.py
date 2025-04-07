# tests/test_utils.py
import numpy as np
import pandas as pd
import pytest
from pampaneira_imputation import utils, config

def test_sliding_window():
    data = np.arange(20).reshape(10, 2) # 10 steps, 2 features
    n_steps = 3
    result = utils.sliding_window(data, n_steps)
    assert result.shape == (8, 3, 2) # 10 - 3 + 1 = 8 samples
    np.testing.assert_array_equal(result[0], data[0:3])
    np.testing.assert_array_equal(result[-1], data[7:10])

def test_create_missingness_point(sample_3d_data_with_nans):
    original_data = sample_3d_data_with_nans.copy()
    rate = 0.2
    missing_data = utils.create_missingness(original_data.copy(), rate=rate, pattern="point")

    assert original_data.shape == missing_data.shape
    original_nans = np.isnan(original_data).sum()
    new_nans = np.isnan(missing_data).sum()
    assert new_nans > original_nans

    # Check approximate rate (can be slightly off due to existing NaNs and randomness)
    total_elements = np.prod(original_data.shape)
    expected_missing = int(total_elements * rate)
    # Allow some tolerance
    assert abs(new_nans - original_nans - expected_missing) < total_elements * 0.05

def test_create_missingness_subseq(sample_3d_data_with_nans):
    original_data = sample_3d_data_with_nans.copy()
    rate = 0.1 # Lower rate as subsequences remove more data
    min_len = 3
    max_len = 6
    missing_data = utils.create_missingness(original_data.copy(), rate=rate, pattern="subseq", min_missing_len=min_len, max_missing_len=max_len)

    assert original_data.shape == missing_data.shape
    original_nans = np.isnan(original_data).sum()
    new_nans = np.isnan(missing_data).sum()
    assert new_nans > original_nans
    # Check if consecutive NaNs were introduced (harder to test precisely)
    # Look for at least one sequence of length >= min_len
    found_subseq = False
    for i in range(missing_data.shape[0]):
        for k in range(missing_data.shape[2]):
            is_nan = np.isnan(missing_data[i, :, k])
            if np.any(np.convolve(is_nan, np.ones(min_len), mode='valid') >= min_len):
                found_subseq = True
                break
        if found_subseq:
            break
    assert found_subseq or new_nans == original_nans # Allow case where no subseq was added by chance


def test_reshape_imputed_to_df(sample_3d_data_with_nans):
    imputed_data = sample_3d_data_with_nans.copy()
    imputed_data[np.isnan(imputed_data)] = 0 # Simulate imputation
    n_samples, n_steps, n_features = imputed_data.shape
    columns = [f"f_{i}" for i in range(n_features)]

    # Create a dummy index corresponding to flattened data
    total_rows = n_samples * n_steps
    original_flat_index = pd.date_range("2023-01-01", periods=total_rows, freq="h", tz=config.TIMEZONE)

    df = utils.reshape_imputed_to_df(imputed_data, original_flat_index, columns, n_steps)

    assert isinstance(df, pd.DataFrame)
    assert df.shape == (total_rows, n_features)
    assert list(df.columns) == columns
    assert isinstance(df.index, pd.DatetimeIndex)
    assert not df.isnull().any().any() # Should be no NaNs after imputation
    # Check if first/last values match roughly (depends on reshape logic)
    np.testing.assert_array_almost_equal(df.iloc[0].values, imputed_data[0, 0, :])
    np.testing.assert_array_almost_equal(df.iloc[-1].values, imputed_data[-1, -1, :])