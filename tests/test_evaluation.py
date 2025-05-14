# tests/test_evaluation.py
import numpy as np
import pandas as pd
import pytest
from pampaneira_imputation import evaluation as ev
from pampaneira_imputation import config
from typing import Optional, Tuple


def test_calculate_imputation_metrics(sample_preprocessed_data):
    y_true = sample_preprocessed_data['test_X_ori']
    mask = sample_preprocessed_data['test_indicating_mask']

    # Ensure NaN-free ground truth
    y_true_no_nan = np.nan_to_num(y_true.copy(), nan=0.0)
    y_pred = np.nan_to_num(sample_preprocessed_data['test_X_ori'].copy(), nan=0.0)

    # Simulate perfect imputation for testing metrics calculation
    y_pred[mask == 1] = y_true_no_nan[mask == 1]
    
    metrics = ev.calculate_imputation_metrics(y_true_no_nan, y_pred, mask)

    assert isinstance(metrics, dict)
    assert 'mae' in metrics
    assert 'rmse' in metrics

    if mask.sum() > 0:
        assert metrics['mae'] == pytest.approx(0.0, abs=1e-6)
        assert metrics['mse'] == pytest.approx(0.0, abs=1e-6)
        assert metrics['rmse'] == pytest.approx(0.0, abs=1e-6)
    else:
        assert np.isnan(metrics['mae'])

def test_calculate_imputation_metrics_imperfect(sample_preprocessed_data):
    y_true = sample_preprocessed_data['test_X_ori']
    mask = sample_preprocessed_data['test_indicating_mask']
    y_pred = sample_preprocessed_data['test_X'].copy()

    # Simulate imperfect imputation
    noise = np.random.randn(*y_pred.shape) * 0.1
    y_true_filled = np.nan_to_num(y_true, nan=0.0)
    y_pred_filled = y_pred.copy()
    y_pred_filled[mask == 1] = y_true_filled[mask == 1] + noise[mask == 1]
    y_pred_filled = np.nan_to_num(y_pred_filled, nan=0.0)

    metrics = ev.calculate_imputation_metrics(y_true, y_pred_filled, mask)

    if mask.sum() > 0:
        assert metrics['mae'] > 0.0
        assert metrics['rmse'] > 0.0
    else:
        assert np.isnan(metrics['mae'])

def test_evaluate_all_methods(sample_preprocessed_data):
    # Simulate imputed results for different methods
    imputed_results = {}
    y_test = sample_preprocessed_data['test_X']
    mask = sample_preprocessed_data['test_indicating_mask']
    y_true = sample_preprocessed_data['test_X_ori']

    # Method 1: Perfect imputation
    imputed_results['perfect'] = np.where(mask, y_true, y_test)
    imputed_results['perfect'] = np.nan_to_num(imputed_results['perfect'], nan=0.0)

    # Method 2: Mean imputation
    imputed_results['mean'] = y_test.copy()
    mean_val = np.nanmean(y_test)
    imputed_results['mean'][mask == 1] = mean_val
    imputed_results['mean'] = np.nan_to_num(imputed_results['mean'], nan=0.0)

    # Add artificial mask to preprocessed data
    sample_preprocessed_data['test_artificial_mask'] = mask

    df_results = ev.evaluate_all_methods(sample_preprocessed_data, imputed_results, methods_to_evaluate=['perfect', 'mean'])

    assert isinstance(df_results, pd.DataFrame)
    assert df_results.shape[0] == 2  # Two methods evaluated
    assert 'MAE' in df_results.columns
    assert 'RMSE' in df_results.columns
    assert df_results.loc['Perfect', 'MAE'] == pytest.approx(0.0, abs=1e-6)
    if mask.sum() > 0:
        assert df_results.loc['Mean', 'MAE'] > 0.0
    else:
        assert np.isnan(df_results.loc['Mean', 'MAE'])