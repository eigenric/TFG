# tests/test_data_preprocessor.py
import pytest
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from pampaneira_imputation import data_preprocessor as dp
from pampaneira_imputation import config

def test_fill_missing_timestamps(sample_feature_data):
    df = sample_feature_data.iloc[10:50].copy()
    start = sample_feature_data.index.min()
    end = sample_feature_data.index.max()
    df_filled = dp.fill_missing_timestamps(df.reset_index(), start, end, date_col='index')
    
    expected_range = pd.date_range(start, end, freq='h', tz=config.TIMEZONE)
    pd.testing.assert_index_equal(df_filled.index, expected_range, exact=False)

    assert df_filled.isnull().any().any()
    assert df_filled.shape[0] == len(expected_range)


def test_preprocess_for_imputation(sample_feature_data):
    df = sample_feature_data.copy()
    feature_cols = ['feature1', 'feature2', 'feature3', 'feature4']
    n_steps = 12
    rate = 0.1

    # Define split ranges within the sample data
    train_start = df.index[0].strftime('%Y-%m-%d %H:%M:%S')
    train_end = df.index[50].strftime('%Y-%m-%d %H:%M:%S')
    val_start = df.index[50].strftime('%Y-%m-%d %H:%M:%S')
    val_end = df.index[75].strftime('%Y-%m-%d %H:%M:%S')
    test_start = df.index[75].strftime('%Y-%m-%d %H:%M:%S')
    test_end = df.index[-1].strftime('%Y-%m-%d %H:%M:%S')

    result = dp.preprocess_for_imputation(
        df,
        feature_cols=feature_cols,
        n_steps=n_steps,
        missing_rate=rate,
        missing_pattern='point',
        train_start=train_start, train_end=train_end,
        val_start=val_start, val_end=val_end,
        test_start=test_start, test_end=test_end
    )

    assert isinstance(result, dict)
    assert result['n_steps'] == n_steps
    assert result['n_features'] == len(feature_cols)
    assert isinstance(result['scaler'], StandardScaler)

    # Check shapes considering windowing effect
    # For train: 51 points (0 to 50 inclusive) - n_steps + 1
    expected_train_samples = 51 - n_steps + 1
    # For val: 26 points (50 to 75 inclusive) - n_steps + 1
    expected_val_samples = 26 - n_steps + 1
    # For test: remaining points - n_steps + 1
    expected_test_samples = len(df.loc[test_start:]) - n_steps + 1

    assert result['train_X'].shape == (expected_train_samples, n_steps, len(feature_cols))
    assert result['val_X'].shape == (expected_val_samples, n_steps, len(feature_cols))
    assert result['test_X'].shape[0] >= expected_test_samples - 1
    assert result['test_X'].shape[1:] == (n_steps, len(feature_cols))

    # Check if missingness was introduced
    assert np.isnan(result['train_X']).sum() > np.isnan(result['train_X_ori']).sum()
    assert np.isnan(result['test_X']).sum() > np.isnan(result['test_X_ori']).sum()

    # Check masks
    assert result['train_indicating_mask'].shape == result['train_X'].shape
    assert result['test_indicating_mask'].shape == result['test_X'].shape
    assert result['train_indicating_mask'].sum() == np.isnan(result['train_X']).sum()