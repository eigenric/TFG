# pampaneira_imputation/data_loader.py
import pandas as pd
from typing import List
from . import config

pd.options.mode.chained_assignment = None

def load_traffic_data(filepath: str = config.TRAFFIC_FILE,
                      columns_to_use: List[str] = config.PAM_BUB_TRAFFIC_COLS,
                      date_col: str = config.DATE_COL,
                      timezone: str = config.TIMEZONE) -> pd.DataFrame:
    """Loads general traffic data, selects relevant columns, converts date."""
    try:
        df = pd.read_csv(filepath)
        df[date_col] = pd.to_datetime(df[date_col])
        # Ensure timezone-aware UTC if not already
        if df[date_col].dt.tz is None:
             df[date_col] = df[date_col].dt.tz_localize(timezone)
        elif df[date_col].dt.tz.zone != timezone:
             df[date_col] = df[date_col].dt.tz_convert(timezone)

        # Select only required columns plus the date column
        df_filtered = df[[date_col] + columns_to_use]

        # Convert integer columns to float64 as in original script
        int_cols = df_filtered.select_dtypes(include='int64').columns
        df_filtered[int_cols] = df_filtered[int_cols].astype('float64')

        return df_filtered

    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        raise
    except KeyError as e:
        print(f"Error: Column {e} not found in {filepath}.")
        raise


def load_intersection_data(filepath: str = config.INTERSECTION_FILE,
                           date_col_original: str = "Date", # Original name in CSV
                           date_col_target: str = config.DATE_COL,
                           truck_pos_col: str = config.TRUCK_POS_COL,
                           target_truck_pos: str = config.TARGET_TRUCK_POS,
                           timezone: str = config.TIMEZONE) -> pd.DataFrame:
    """Loads intersection data, filters by truck position, converts date."""
    try:
        df = pd.read_csv(filepath)
        df.rename(columns={date_col_original: date_col_target}, inplace=True)
        df[date_col_target] = pd.to_datetime(df[date_col_target])

        # Ensure timezone-aware UTC if not already
        if df[date_col_target].dt.tz is None:
             df[date_col_target] = df[date_col_target].dt.tz_localize(timezone)
        elif df[date_col_target].dt.tz.zone != timezone:
             df[date_col_target] = df[date_col_target].dt.tz_convert(timezone)

        # Filter by truck position
        df_filtered = df[df[truck_pos_col] == target_truck_pos].copy() # Use .copy()

        # Convert integer columns to float64
        int_cols = df_filtered.select_dtypes(include='int64').columns
        df_filtered[int_cols] = df_filtered[int_cols].astype('float64')

        # Select only the final feature columns + date (defined in config)
        # This assumes the intersection file contains all FEATURE_COLUMNS
        # If not, adjust config.FEATURE_COLUMNS or this selection logic
        cols_to_keep = [date_col_target] + config.FEATURE_COLUMNS
        # Ensure we only keep columns present in the dataframe
        cols_present = [col for col in cols_to_keep if col in df_filtered.columns]
        df_final = df_filtered[cols_present]


        return df_final

    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        raise
    except KeyError as e:
        print(f"Error: Column {e} not found or renaming failed in {filepath}.")
        raise