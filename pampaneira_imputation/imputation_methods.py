# pampaneira_imputation/imputation_methods.py
import numpy as np
import pandas as pd
import warnings
from typing import Optional, Tuple


def impute_median(X_missing: np.ndarray) -> np.ndarray:
    X_imputed = X_missing.copy()
    n_samples, n_steps, n_features = X_imputed.shape

    for k in range(n_features):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            median_val = np.nanmedian(X_missing[:, :, k]) # Calculate on ORIGINAL

        # --- Explicit Handling for All-NaN ---
        if np.isnan(median_val):
            print(f"Warning: Feature {k} is all NaN. Imputing with 0.")
            # Directly impute 0 for this specific feature where it's NaN
            current_nan_mask = np.isnan(X_imputed[:, :, k])
            X_imputed[:, :, k][current_nan_mask] = 0.0
        else:
            # Impute the calculated median for non-all-NaN features
            current_nan_mask = np.isnan(X_imputed[:, :, k])
            X_imputed[:, :, k][current_nan_mask] = median_val
        # --- End Explicit Handling ---

    return X_imputed

def impute_mean(X_missing: np.ndarray) -> np.ndarray:
    X_imputed = X_missing.copy()
    n_samples, n_steps, n_features = X_imputed.shape

    for k in range(n_features):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mean_val = np.nanmean(X_missing[:, :, k]) # Calculate on ORIGINAL

        # --- Explicit Handling for All-NaN ---
        if np.isnan(mean_val):
            print(f"Warning: Feature {k} is all NaN. Imputing with 0.")
            current_nan_mask = np.isnan(X_imputed[:, :, k])
            X_imputed[:, :, k][current_nan_mask] = 0.0
        else:
            current_nan_mask = np.isnan(X_imputed[:, :, k])
            X_imputed[:, :, k][current_nan_mask] = mean_val
        # --- End Explicit Handling ---

    return X_imputed

def impute_median_sample_wise(data):
    """
    Impute missing values using the median of each sample (across time).
    If an entire feature for a sample is NaN, use a fallback value.
    
    Parameters:
    -----------
    data : ndarray of shape (n_samples, n_timestamps, n_features)
        Input data with NaN values to be imputed.
        
    Returns:
    --------
    ndarray of shape (n_samples, n_timestamps, n_features)
        Data with NaN values imputed.
    """
    imputed_data = data.copy()
    n_samples, n_timestamps, n_features = data.shape
    
    for feat_idx in range(n_features):
        for sample_idx in range(n_samples):
            # Extract the feature slice for this sample
            feature_slice = data[sample_idx, :, feat_idx]
            
            # Check if the slice contains any non-NaN values
            if not np.isnan(feature_slice).all():
                # Calculate median only if there are non-NaN values
                median_val = np.nanmedian(feature_slice)
                
                # Apply the median to NaN values in this sample's feature
                nan_indices = np.isnan(imputed_data[sample_idx, :, feat_idx])
                imputed_data[sample_idx, nan_indices, feat_idx] = median_val
            else:
                # If all values are NaN, apply fallback strategy
                # Option 1: Use median of this feature across all other samples
                global_feature_median = np.nanmedian(data[:, :, feat_idx])
                if not np.isnan(global_feature_median):
                    imputed_data[sample_idx, :, feat_idx] = global_feature_median
                else:
                    # Option 2: Use a default value (e.g., 0 or the global median of all data)
                    imputed_data[sample_idx, :, feat_idx] = np.nanmedian(data) if not np.isnan(data).all() else 0
    
    return imputed_data
    
def impute_mean_sample_wise(data):
    """
    Impute missing values using the mean of each sample (across time).
    If an entire feature for a sample is NaN, use a fallback value.
    
    Parameters:
    -----------
    data : ndarray of shape (n_samples, n_timestamps, n_features)
        Input data with NaN values to be imputed.
        
    Returns:
    --------
    ndarray of shape (n_samples, n_timestamps, n_features)
        Data with NaN values imputed.
    """
    imputed_data = data.copy()
    n_samples, n_timestamps, n_features = data.shape
    
    for feat_idx in range(n_features):
        for sample_idx in range(n_samples):
            # Extract the feature slice for this sample
            feature_slice = data[sample_idx, :, feat_idx]
            
            # Check if the slice contains any non-NaN values
            if not np.isnan(feature_slice).all():
                # Calculate mean only if there are non-NaN values
                mean_val = np.nanmean(feature_slice)
                
                # Apply the mean to NaN values in this sample's feature
                nan_indices = np.isnan(imputed_data[sample_idx, :, feat_idx])
                imputed_data[sample_idx, nan_indices, feat_idx] = mean_val
            else:
                # If all values are NaN, apply fallback strategy
                # Option 1: Use mean of this feature across all other samples
                global_feature_mean = np.nanmean(data[:, :, feat_idx])
                if not np.isnan(global_feature_mean):
                    imputed_data[sample_idx, :, feat_idx] = global_feature_mean
                else:
                    # Option 2: Use a default value (e.g., 0 or the global mean of all data)
                    imputed_data[sample_idx, :, feat_idx] = np.nanmean(data) if not np.isnan(data).all() else 0
    
    return imputed_data   

def impute_forward_backward(X_missing: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    X_filled_forward_then_backward = X_missing.copy()
    X_filled_backward_then_forward = X_missing.copy()
    n_samples, n_steps, n_features = X_missing.shape

    for i in range(n_samples):
        sample_df = pd.DataFrame(X_missing[i, :, :]) # Convert step x feature slice to DataFrame

        # Forward fill then backward fill
        filled_fb = sample_df.ffill(axis=0).bfill(axis=0)
        # Fallback for remaining NaNs (all-NaN columns)
        if filled_fb.isnull().any().any():
            filled_fb.fillna(0.0, inplace=True)
        X_filled_forward_then_backward[i, :, :] = filled_fb.to_numpy()


        # Backward fill then forward fill
        filled_bf = sample_df.bfill(axis=0).ffill(axis=0)
        # Fallback for remaining NaNs (all-NaN columns)
        if filled_bf.isnull().any().any():
            filled_bf.fillna(0.0, inplace=True)
        X_filled_backward_then_forward[i, :, :] = filled_bf.to_numpy()


    return X_filled_forward_then_backward, X_filled_backward_then_forward

# pampaneira_imputation/imputation_methods.py
def impute_linear(X_missing: np.ndarray) -> np.ndarray:
    X_interpolated = X_missing.copy()
    n_samples, n_steps, n_features = X_missing.shape

    for i in range(n_samples):
        sample_df = pd.DataFrame(X_missing[i, :, :])
        interpolated_df = sample_df.interpolate(method='linear', axis=0, limit_direction='both')
        # Check if any NaNs remain (e.g., if start/end are NaN OR if whole col was NaN)
        # Fallback: fill remaining NaNs (e.g., with 0 or ffill/bfill)
        if interpolated_df.isnull().any().any():
             # Option 1: Use ffill/bfill (might still fail if whole column is NaN)
             # interpolated_df = interpolated_df.ffill(axis=0).bfill(axis=0)
             # Option 2: Fill with a constant (like 0)
             interpolated_df.fillna(0.0, inplace=True)

        X_interpolated[i, :, :] = interpolated_df.to_numpy()

    return X_interpolated

