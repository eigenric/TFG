# tests/conftest.py
import pytest
import pandas as pd
import numpy as np
from pampaneira_imputation import config # Use config for consistency

@pytest.fixture(scope="module")
def sample_datetime_index():
    """Provides a sample DatetimeIndex."""
    return pd.date_range(start="2023-01-01 00:00", periods=100, freq="h", tz=config.TIMEZONE)

@pytest.fixture(scope="module")
def sample_feature_data(sample_datetime_index):
    """Provides sample DataFrame with features and NaNs."""
    n_periods = len(sample_datetime_index)
    data = {
        'feature1': np.linspace(0, 1, n_periods) + np.random.rand(n_periods) * 0.1,
        'feature2': np.sin(np.linspace(0, 4 * np.pi, n_periods)) + np.random.rand(n_periods) * 0.2,
        'feature3': np.random.randn(n_periods)
    }
    df = pd.DataFrame(data, index=sample_datetime_index)
    # Add some NaNs
    df.iloc[5:10, 0] = np.nan
    df.iloc[20:25, 1] = np.nan
    df.iloc[::10, 2] = np.nan
    df['feature4'] = 5.0 # Add a constant feature
    df.iloc[30, 3] = np.nan
    return df

# tests/conftest.py
import pytest
import pandas as pd
import numpy as np
from pampaneira_imputation import config
import warnings

# ... other fixtures ...

# tests/conftest.py
@pytest.fixture(scope="function")
def sample_3d_data_with_nans():
    n_samples = 10
    n_steps = config.N_STEPS
    n_features = 4
    data = np.ones((n_samples, n_steps, n_features)) # Use ones
    data[0, 5:10, 0] = np.nan
    data[1, :, 1] = np.nan    # Feature 1 all NaN
    data[2, 15, 2] = np.nan
    data[3, 0, :] = np.nan
    return data.copy()

@pytest.fixture(scope="function")
def sample_preprocessed_data(sample_3d_data_with_nans):
    """Provides a sample preprocessed data dictionary for evaluation tests."""
    data = sample_3d_data_with_nans
    ori_data = data.copy()
    # Simulate some observed NaNs being filled (e.g., by scaling if mean was NaN)
    ori_data[1, :, 1] = 0.0 # Assume NaN feature was imputed with 0 in ground truth for metrics

    mask = np.isnan(data).astype(int)

    return {
        "test_X_ori": ori_data,
        "test_indicating_mask": mask,
        "test_X": data # The data with NaNs to be imputed
    }