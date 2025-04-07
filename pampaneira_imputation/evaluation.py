# pampaneira_imputation/evaluation.py
import numpy as np
import pandas as pd
from pypots.nn.functional import calc_mae, calc_mse, calc_rmse, calc_mre
from typing import Dict, Tuple, List 
from . import config # <-- Added config import

def calculate_imputation_metrics(y_true: np.ndarray,
                                 y_pred: np.ndarray,
                                 indicating_mask: np.ndarray) -> Dict[str, float]:
    """
    Calculates MAE, MSE, RMSE, MRE for imputed values where mask is 1.

    Args:
        y_true: Ground truth data (potentially with NaNs where originally missing).
        y_pred: Imputed data.
        indicating_mask: Mask where 1 indicates a missing value that was imputed,
                         0 indicates an observed value.

    Returns:
        Dictionary containing 'mae', 'mse', 'rmse', 'mre'.
    """
    # Ensure inputs are numpy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    indicating_mask = np.asarray(indicating_mask)

    # Replace NaNs in ground truth with 0 for calculation where mask is 1
    # This is necessary because pypots functions expect non-NaN ground truth
    # We only evaluate where indicating_mask is 1, so this replacement is safe.
    y_true_filled = np.nan_to_num(y_true, nan=0.0)

    if y_true.shape != y_pred.shape or y_true.shape != indicating_mask.shape:
        raise ValueError(f"Shape mismatch: y_true={y_true.shape}, "
                         f"y_pred={y_pred.shape}, mask={indicating_mask.shape}")

    # Check if mask sum is zero (no values to evaluate)
    if indicating_mask.sum() == 0:
        print("Warning: Indicating mask sum is 0. No imputed values to evaluate.")
        return {'mae': np.nan, 'mse': np.nan, 'rmse': np.nan, 'mre': np.nan}

    try:
        mae = calc_mae(y_pred, y_true_filled, indicating_mask)
        mse = calc_mse(y_pred, y_true_filled, indicating_mask)
        rmse = calc_rmse(y_pred, y_true_filled, indicating_mask)
        mre = calc_mre(y_pred, y_true_filled, indicating_mask) # Be cautious with MRE if true values are near zero

        return {'mae': mae, 'mse': mse, 'rmse': rmse, 'mre': mre}
    except Exception as e:
         print(f"Error during metric calculation: {e}")
         # Add more debug info if needed
         print(f"Shapes: y_pred={y_pred.shape}, y_true_filled={y_true_filled.shape}, mask={indicating_mask.shape}")
         print(f"Mask sum: {indicating_mask.sum()}")
         print(f"NaN counts: pred={np.isnan(y_pred).sum()}, true_filled={np.isnan(y_true_filled).sum()}, mask={np.isnan(indicating_mask).sum()}")
         # Consider checking for infinities as well
         return {'mae': np.nan, 'mse': np.nan, 'rmse': np.nan, 'mre': np.nan}


def evaluate_all_methods(preprocessed_data: Dict,
                         imputed_results: Dict[str, np.ndarray],
                         methods_to_evaluate: list = ['median', 'mean', 'linear', 'ffill_bfill', 'bfill_ffill', 'saits']) -> pd.DataFrame:
    """
    Evaluates multiple imputation methods using the test set results.

    Args:
        preprocessed_data: Dictionary from preprocess_for_imputation.
        imputed_results: Dictionary mapping method names to imputed numpy arrays (test set).
        methods_to_evaluate: List of keys in imputed_results to evaluate.

    Returns:
        Pandas DataFrame summarizing MAE, MSE, RMSE, MRE for each method.
    """
    results = []
    y_true = preprocessed_data['test_X_ori']
    indicating_mask = preprocessed_data['test_indicating_mask']

    # Handle potential dropping of columns for ffill/bfill if necessary
    # This logic assumes WS/WD were dropped *before* imputation for ffill/bfill
    cols_to_drop_indices = [config.FEATURE_COLUMNS.index(col) for col in config.COLS_TO_DROP_FOR_BASELINE if col in config.FEATURE_COLUMNS]

    for method_name in methods_to_evaluate:
        if method_name not in imputed_results:
            print(f"Warning: Imputed results for method '{method_name}' not found. Skipping.")
            continue

        y_pred = imputed_results[method_name]
        current_y_true = y_true
        current_mask = indicating_mask

        # Specific handling for methods where columns might have been dropped
        if method_name in ['ffill_bfill', 'bfill_ffill'] and cols_to_drop_indices:
             print(f"Adjusting true data and mask for {method_name} due to dropped columns.")
             current_y_true = np.delete(y_true, cols_to_drop_indices, axis=2)
             current_mask = np.delete(indicating_mask, cols_to_drop_indices, axis=2)
             # y_pred for these methods should already have columns dropped

        print(f"\nCalculating metrics for: {method_name}")
        metrics = calculate_imputation_metrics(current_y_true, y_pred, current_mask)

        results.append({
            "Method": method_name.replace('_', ' ').title(), # Nicer name
            "RMSE": metrics.get('rmse', np.nan),
            "MSE": metrics.get('mse', np.nan),
            "MAE": metrics.get('mae', np.nan),
            "MRE": metrics.get('mre', np.nan)
        })

    error_table = pd.DataFrame.from_records(results)
    error_table = error_table.set_index("Method").round(4)
    return error_table