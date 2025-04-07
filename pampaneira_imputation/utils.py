# pampaneira_imputation/utils.py
import numpy as np
import pandas as pd
from typing import List

def sliding_window(data: np.ndarray, n_steps: int) -> np.ndarray:
    """Creates sliding windows from sequential data."""
    X = []
    for i in range(len(data) - n_steps + 1):
        X.append(data[i : i + n_steps])
    return np.array(X)

def create_missingness(data: np.ndarray, rate: float, pattern: str, **kwargs) -> np.ndarray:
    """
    Introduces missing values (NaN) into a 3D numpy array (samples, steps, features).
    Modifies the array in place for efficiency, but returns it.
    """
    n_samples, n_steps, n_features = data.shape
    missing_data = data.copy() # Work on a copy

    if pattern == "point":
        total_elements = n_samples * n_steps * n_features
        n_missing = int(total_elements * rate)
        indices = np.random.choice(total_elements, n_missing, replace=False)
        coords = np.unravel_index(indices, (n_samples, n_steps, n_features))
        missing_data[coords] = np.nan

    elif pattern == "subseq":
         # More complex: Introduce missing subsequences per feature per sample
         # This implementation introduces missingness *independently* per feature
         # Adjust if block missingness across features is desired
         min_len = kwargs.get('min_missing_len', 1)
         max_len = kwargs.get('max_missing_len', n_steps // 4) # Example max length

         for i in range(n_samples):
             for k in range(n_features):
                 # Decide if this feature sequence gets missingness
                 if np.random.rand() < rate: # Probability 'rate' to add missingness block
                     length = np.random.randint(min_len, max_len + 1)
                     start_idx = np.random.randint(0, n_steps - length + 1)
                     missing_data[i, start_idx : start_idx + length, k] = np.nan

    elif pattern == "block":
         # Introduce missingness in blocks starting from index 0
         # This interpretation might differ; adjust if needed.
         min_len = kwargs.get('min_missing_len', 1)
         max_len = kwargs.get('max_missing_len', n_steps // 2) # Example max length

         for i in range(n_samples):
             for k in range(n_features):
                if np.random.rand() < rate: # Probability 'rate' to add missingness block
                    block_size = np.random.randint(min_len, max_len + 1)
                    missing_data[i, :block_size, k] = np.nan
    else:
        raise ValueError(f"Unknown missingness pattern: {pattern}")

    return missing_data


def reshape_imputed_to_df(imputed_data: np.ndarray,
                          original_index: pd.DatetimeIndex,
                          columns: List[str],
                          n_steps: int) -> pd.DataFrame:
    """
    Reshapes 3D imputed data (samples, steps, features) back to a 2D DataFrame.
    Assumes the original index corresponds to the *start* of each window.
    Reconstructs the full timeline.
    """
    n_samples, _, n_features = imputed_data.shape
    if len(columns) != n_features:
         raise ValueError(f"Number of columns ({len(columns)}) must match number of features ({n_features})")

    # Naive reconstruction: Use the last value of each window for its corresponding time step
    # A more sophisticated approach might average overlapping windows, but this is simpler.
    # This effectively takes the imputation for the *last* time step in each window.
    # reconstructed_data = imputed_data[:, -1, :]

    # Alternative: Flatten and assume contiguous windows (handle with care for overlap/gaps)
    # This matches the original script's apparent flattening logic
    flat_data = imputed_data.reshape(-1, n_features)

    # Need to reconstruct the index carefully based on the original split
    # The original script implicitly used the index from the test_set *before* windowing
    # Let's assume the provided `original_index` is the index of the *flattened* test data
    # before windowing. We need enough index points for the flattened data.
    num_expected_rows = n_samples * n_steps # or just len(flat_data)
    if len(original_index) < num_expected_rows:
        # This suggests the original index was perhaps from the *start* of windows.
        # Let's reconstruct a full index based on the start and frequency.
        if isinstance(original_index, pd.DatetimeIndex) and original_index.freq:
            full_index = pd.date_range(start=original_index[0], periods=num_expected_rows, freq=original_index.freq)
        else:
            # Cannot reliably reconstruct index without frequency info or correct original index
            print("Warning: Cannot reliably reconstruct DataFrame index. Using range index.")
            full_index = pd.RangeIndex(stop=num_expected_rows)

    else:
        # Assume original_index covers all time steps in the flattened output
        full_index = original_index[:num_expected_rows]


    df = pd.DataFrame(flat_data, index=full_index, columns=columns)
    return df