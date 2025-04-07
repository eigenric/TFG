# pampaneira_imputation/data_preprocessor.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Dict, Tuple, Optional, List 
from . import config
from .utils import sliding_window, create_missingness

def fill_missing_timestamps(df: pd.DataFrame,
                            start_date: pd.Timestamp,
                            end_date: pd.Timestamp,
                            freq: str = 'h',
                            date_col: str = config.DATE_COL) -> pd.DataFrame:
    """Fills missing hourly timestamps in a DataFrame with NaNs."""
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df[date_col] = pd.to_datetime(df[date_col])
    if df[date_col].dt.tz is None:
         df[date_col] = df[date_col].dt.tz_localize(config.TIMEZONE) # Ensure timezone
    df = df.set_index(date_col)
    full_date_range = pd.date_range(start=start_date, end=end_date, freq=freq, tz=config.TIMEZONE)
    df_reindexed = df.reindex(full_date_range)
    # Don't reset the index if you want to preserve the DatetimeIndex
    return df_reindexed

def split_by_period(df: pd.DataFrame,
                    period_1_start: pd.Timestamp = config.PERIOD_1_START,
                    period_1_end: pd.Timestamp = config.PERIOD_1_END,
                    period_2_start: pd.Timestamp = config.PERIOD_2_START,
                    period_2_end: pd.Timestamp = config.PERIOD_2_END,
                    date_col: str = config.DATE_COL) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Splits the DataFrame into two periods based on predefined dates."""
    if not pd.api.types.is_datetime64_any_dtype(df.index):
         # Assume date_col is the index if not already set
         if date_col in df.columns:
             df = df.set_index(date_col)
         else:
             raise ValueError(f"'{date_col}' not found as index or column for splitting.")

    periodo_1 = df.loc[period_1_start:period_1_end].copy()
    periodo_2 = df.loc[period_2_start:period_2_end].copy()
    return periodo_1, periodo_2

def add_period1_padding(df_period1: pd.DataFrame,
                        start_pad: pd.Timestamp = config.PERIOD_1_PADDING_START,
                        end_pad: pd.Timestamp = config.PERIOD_1_PADDING_END,
                        freq: str = 'h') -> pd.DataFrame:
    """Adds NaN padding to the start and end of Period 1 data."""
    if not isinstance(df_period1.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex for padding.")

    # Create padding dataframes
    start_range = pd.date_range(start=start_pad, end=df_period1.index.min() - pd.Timedelta(hours=1), freq=freq, tz=config.TIMEZONE)
    end_range = pd.date_range(start=df_period1.index.max() + pd.Timedelta(hours=1), end=end_pad, freq=freq, tz=config.TIMEZONE)

    df_start_pad = pd.DataFrame(np.nan, index=start_range, columns=df_period1.columns)
    df_end_pad = pd.DataFrame(np.nan, index=end_range, columns=df_period1.columns)

    # Concatenate
    df_padded = pd.concat([df_start_pad, df_period1, df_end_pad], axis=0)
    # Ensure index is sorted if concatenation messes order (unlikely with date ranges)
    df_padded = df_padded.sort_index()

    return df_padded


def preprocess_for_imputation(
    df: pd.DataFrame,
    feature_cols: list = config.FEATURE_COLUMNS,
    train_start: str = config.TRAIN_START_DATE,
    train_end: str = config.TRAIN_END_DATE,
    val_start: str = config.VAL_START_DATE,
    val_end: str = config.VAL_END_DATE,
    test_start: str = config.TEST_START_DATE,
    test_end: str = config.TEST_END_DATE,
    n_steps: int = config.N_STEPS,
    missing_rate: float = config.MISSING_RATE,
    missing_pattern: str = config.MISSING_PATTERN,
    **missingness_kwargs
) -> Dict:
    """
    Prepares data for imputation models: splits, scales, windows, adds missingness.

    Args:
        df: DataFrame with DatetimeIndex and feature columns.
        feature_cols: List of columns to use as features.
        train_start/end, val_start/end, test_start/end: Date strings for splitting.
        n_steps: Size of the sliding window.
        missing_rate: Proportion of values to make NaN (0 to disable).
        missing_pattern: 'point', 'subseq', or 'block'.
        **missingness_kwargs: Additional arguments for create_missingness (e.g., min/max length).

    Returns:
        A dictionary containing processed data splits (train_X, val_X, test_X),
        original (unmasked) data (train_X_ori, etc.), the scaler, number of features,
        and number of steps. Also includes indicating masks.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex.")

    # 1. Split Data
    mask_train = (df.index >= train_start) & (df.index < train_end)
    train_set = df.loc[mask_train, feature_cols]

    mask_val = (df.index >= val_start) & (df.index < val_end)
    val_set = df.loc[mask_val, feature_cols]

    mask_test = (df.index >= test_start) & (df.index < test_end)
    test_set = df.loc[mask_test, feature_cols]

    if train_set.empty or val_set.empty or test_set.empty:
        raise ValueError("One or more data splits are empty. Check date ranges and input data.")

    # Store original indices for later reshaping if needed
    train_index = train_set.index
    val_index = val_set.index
    test_index = test_set.index

    # 2. Scale Data
    scaler = StandardScaler()
    train_X_scaled = scaler.fit_transform(train_set)
    val_X_scaled = scaler.transform(val_set)
    test_X_scaled = scaler.transform(test_set)

    # Convert back to DataFrame to preserve index for windowing alignment (optional but safer)
    # train_X_scaled_df = pd.DataFrame(train_X_scaled, index=train_index, columns=feature_cols)
    # val_X_scaled_df = pd.DataFrame(val_X_scaled, index=val_index, columns=feature_cols)
    # test_X_scaled_df = pd.DataFrame(test_X_scaled, index=test_index, columns=feature_cols)
    # Note: Using sliding_window directly on numpy arrays is more standard

    # 3. Apply Sliding Window
    # Important: Sliding window reduces the number of samples.
    # The resulting windows correspond to sequences ending at time t, t+1, ...
    # Keep track of the index corresponding to the *start* of each window if needed.
    train_X_win = sliding_window(train_X_scaled, n_steps)
    val_X_win = sliding_window(val_X_scaled, n_steps)
    test_X_win = sliding_window(test_X_scaled, n_steps)

    # Adjust original indices to match window starts (if needed downstream)
    # train_win_index = train_index[n_steps-1:]
    # val_win_index = val_index[n_steps-1:]
    # test_win_index = test_index[n_steps-1:]

    n_features = train_X_win.shape[-1]

    processed_data = {
        "n_steps": n_steps,
        "n_features": n_features,
        "scaler": scaler,
        "train_index": train_index, # Index before windowing
        "val_index": val_index,     # Index before windowing
        "test_index": test_index,   # Index before windowing
        # "train_win_index": train_win_index, # Index of window starts
        # "val_win_index": val_win_index,
        # "test_win_index": test_win_index,
        # Store original scaled+windowed data before masking
        "train_X_ori": train_X_win.copy(),
        "val_X_ori": val_X_win.copy(),
        "test_X_ori": test_X_win.copy(),
        # These will be overwritten if missingness is added
        "train_X": train_X_win,
        "val_X": val_X_win,
        "test_X": test_X_win,
    }

    # 4. Introduce Missingness (Optional)
    if missing_rate > 0:
        print(f"Introducing {missing_rate*100:.1f}% missingness with pattern '{missing_pattern}'...")
        # Create masks based on the *original* windowed data before introducing NaNs
        processed_data["train_missing_mask"] = np.isnan(processed_data["train_X_ori"])
        processed_data["val_missing_mask"] = np.isnan(processed_data["val_X_ori"])
        processed_data["test_missing_mask"] = np.isnan(processed_data["test_X_ori"])


        # Apply missingness
        train_X_missing = create_missingness(processed_data["train_X_ori"], missing_rate, missing_pattern, **missingness_kwargs)
        val_X_missing = create_missingness(processed_data["val_X_ori"], missing_rate, missing_pattern, **missingness_kwargs)
        test_X_missing = create_missingness(processed_data["test_X_ori"], missing_rate, missing_pattern, **missingness_kwargs)

        processed_data["train_X"] = train_X_missing
        processed_data["val_X"] = val_X_missing
        processed_data["test_X"] = test_X_missing

        # Create indicating masks (1 where value is missing, 0 otherwise) *after* introduction
        processed_data["train_indicating_mask"] = np.isnan(train_X_missing).astype(int)
        processed_data["val_indicating_mask"] = np.isnan(val_X_missing).astype(int)
        processed_data["test_indicating_mask"] = np.isnan(test_X_missing).astype(int)

    else:
        print("Missing rate is 0. No artificial missingness introduced.")
        # If no missingness added, masks reflect original NaNs (if any after scaling/windowing)
        processed_data["train_missing_mask"] = np.isnan(processed_data["train_X"])
        processed_data["val_missing_mask"] = np.isnan(processed_data["val_X"])
        processed_data["test_missing_mask"] = np.isnan(processed_data["test_X"])
        processed_data["train_indicating_mask"] = processed_data["train_missing_mask"].astype(int)
        processed_data["val_indicating_mask"] = processed_data["val_missing_mask"].astype(int)
        processed_data["test_indicating_mask"] = processed_data["test_missing_mask"].astype(int)


    # Add number of samples
    processed_data["n_train_samples"] = processed_data["train_X"].shape[0]
    processed_data["n_val_samples"] = processed_data["val_X"].shape[0]
    processed_data["n_test_samples"] = processed_data["test_X"].shape[0]


    return processed_data