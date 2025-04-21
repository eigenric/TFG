# pampaneira_imputation/imputation_methods.py
import numpy as np
import pandas as pd
import warnings

from typing import Optional, Tuple
from .config import *
from .utils import timeit_factory

from pypots.imputation import SAITS, Transformer
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import SmoothL1Loss

imputation_times = {}
timeit = timeit_factory(imputation_times)

@timeit
def impute_median(X_missing: np.ndarray) -> np.ndarray:
    """
    Imputa los valores faltantes en un array 3D usando la mediana por característica.
s
    Calcula la mediana de cada característica (a través de muestras y pasos de tiempo) y
    reemplaza los valores NaN con la mediana calculada para esa característica.

    Args:
        X_missing (np.ndarray): Array NumPy 3D de entrada con valores faltantes (NaNs).
                                 Forma: (n_samples, n_steps, n_features).

    Returns:
        np.ndarray: Array NumPy 3D con valores faltantes imputados.
    """
    X_imputed = X_missing.copy()
    n_samples, n_steps, n_features = X_imputed.shape

    for k in range(n_features):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            median_val = np.nanmedian(X_missing[:, :, k])  # Calcula sobre el ORIGINAL

        if np.isnan(median_val):
            print(f"Advertencia: La característica {k} es toda NaN. Imputando con 0.")
            # Imputa directamente 0 para esta característica específica donde es NaN
            current_nan_mask = np.isnan(X_imputed[:, :, k])
            X_imputed[:, :, k][current_nan_mask] = 0.0
        else:
            # Imputa la mediana calculada para características que no son todas NaN
            current_nan_mask = np.isnan(X_imputed[:, :, k])
            X_imputed[:, :, k][current_nan_mask] = median_val

    return X_imputed

@timeit
def impute_mean(X_missing: np.ndarray) -> np.ndarray:
    """
    Imputa los valores faltantes en un array 3D usando la media por característica.

    Calcula la media de cada característica (a través de muestras y pasos de tiempo) y
    reemplaza los valores NaN con la media calculada para esa característica.

    Args:
        X_missing (np.ndarray): Array NumPy 3D de entrada con valores faltantes (NaNs).
                                 Forma: (n_samples, n_steps, n_features).

    Returns:
        np.ndarray: Array NumPy 3D con valores faltantes imputados.
    """
    X_imputed = X_missing.copy()
    n_samples, n_steps, n_features = X_imputed.shape

    for k in range(n_features):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mean_val = np.nanmean(X_missing[:, :, k])  # Calcula sobre el ORIGINAL

        # --- Manejo explícito para todos NaN ---
        if np.isnan(mean_val):
            print(f"Advertencia: La característica {k} es toda NaN. Imputando con 0.")
            current_nan_mask = np.isnan(X_imputed[:, :, k])
            X_imputed[:, :, k][current_nan_mask] = 0.0
        else:
            current_nan_mask = np.isnan(X_imputed[:, :, k])
            X_imputed[:, :, k][current_nan_mask] = mean_val
        # --- Fin del manejo explícito ---

    return X_imputed

@timeit
def impute_median_sample_wise(data):
    """
    Imputa los valores faltantes usando la mediana de cada muestra (a través del tiempo).

    Si toda una característica para una muestra es NaN, usa un valor de reserva.

    Args:
        data (ndarray): Array NumPy de forma (n_samples, n_timestamps, n_features)
                        Datos de entrada con valores NaN para ser imputados.

    Returns:
        ndarray: Array NumPy de forma (n_samples, n_timestamps, n_features)
                 Datos con valores NaN imputados.
    """
    imputed_data = data.copy()
    n_samples, n_timestamps, n_features = data.shape

    for feat_idx in range(n_features):
        for sample_idx in range(n_samples):
            # Extrae el slice de característica para esta muestra
            feature_slice = data[sample_idx, :, feat_idx]

            # Verifica si el slice contiene algún valor no NaN
            if not np.isnan(feature_slice).all():
                # Calcula la mediana solo si hay valores no NaN
                median_val = np.nanmedian(feature_slice)

                # Aplica la mediana a los valores NaN en la característica de esta muestra
                nan_indices = np.isnan(imputed_data[sample_idx, :, feat_idx])
                imputed_data[sample_idx, nan_indices, feat_idx] = median_val
            else:
                # Si todos los valores son NaN, aplica la estrategia de reserva
                # Opción 1: Usa la mediana de esta característica a través de todas las otras muestras
                global_feature_median = np.nanmedian(data[:, :, feat_idx])
                if not np.isnan(global_feature_median):
                    imputed_data[sample_idx, :, feat_idx] = global_feature_median
                else:
                    # Opción 2: Usa un valor por defecto (ej., 0 o la mediana global de todos los datos)
                    imputed_data[sample_idx, :, feat_idx] = np.nanmedian(data) if not np.isnan(data).all() else 0

    return imputed_data

@timeit
def impute_mean_sample_wise(data):
    """
    Imputa los valores faltantes usando la media de cada muestra (a través del tiempo).

    Si toda una característica para una muestra es NaN, usa un valor de reserva.

    Args:
        data (ndarray): Array NumPy de forma (n_samples, n_timestamps, n_features)
                        Datos de entrada con valores NaN para ser imputados.

    Returns:
        ndarray: Array NumPy de forma (n_samples, n_timestamps, n_features)
                 Datos con valores NaN imputados.
    """
    imputed_data = data.copy()
    n_samples, n_timestamps, n_features = data.shape

    for feat_idx in range(n_features):
        for sample_idx in range(n_samples):
            # Extrae el slice de característica para esta muestra
            feature_slice = data[sample_idx, :, feat_idx]

            # Verifica si el slice contiene algún valor no NaN
            if not np.isnan(feature_slice).all():
                # Calcula la media solo si hay valores no NaN
                mean_val = np.nanmean(feature_slice)

                # Aplica la media a los valores NaN en la característica de esta muestra
                nan_indices = np.isnan(imputed_data[sample_idx, :, feat_idx])
                imputed_data[sample_idx, nan_indices, feat_idx] = mean_val
            else:
                # Si todos los valores son NaN, aplica la estrategia de reserva
                # Opción 1: Usa la media de esta característica a través de todas las otras muestras
                global_feature_mean = np.nanmean(data[:, :, feat_idx])
                if not np.isnan(global_feature_mean):
                    imputed_data[sample_idx, :, feat_idx] = global_feature_mean
                else:
                    # Opción 2: Usa un valor por defecto (ej., 0 o la media global de todos los datos)
                    imputed_data[sample_idx, :, feat_idx] = np.nanmean(data) if not np.isnan(data).all() else 0

    return imputed_data

@timeit
def impute_forward(X_missing: np.ndarray, fillna_value=0.0) -> np.ndarray:
    """
    Realiza la imputación forward-fill en un array 3D.

    Aplica forward-fill para imputar valores faltantes (NaNs) en cada muestra
    del array de entrada.

    Args:
        X_missing (np.ndarray): Array NumPy 3D de entrada con valores faltantes (NaNs).
                                 Forma: (n_samples, n_steps, n_features).
        fillna_value (float, optional): Valor para rellenar los NaNs restantes después del
                                       forward fill (por ejemplo, columnas completamente NaN).
                                       Por defecto es 0.0.

    Returns:
        np.ndarray: Array NumPy 3D con forward-fill aplicado.

    Raises:
        ValueError: Si X_missing no es un array NumPy 3D.
    """
    if not isinstance(X_missing, np.ndarray) or X_missing.ndim != 3:
        raise ValueError("X_missing debe ser un array NumPy 3D.")

    X_filled_forward = X_missing.copy()
    n_samples, n_steps, n_features = X_missing.shape

    for i in range(n_samples):
        sample_df = pd.DataFrame(X_missing[i, :, :])  # Convierte el slice paso x característica a DataFrame

        # Forward fill
        filled_f = sample_df.ffill(axis=0)
        # Reserva para NaNs restantes (columnas totalmente NaN)
        if filled_f.isnull().any().any():
            filled_f.fillna(fillna_value, inplace=True)
        X_filled_forward[i, :, :] = filled_f.to_numpy()

    return X_filled_forward

@timeit
def impute_backward(X_missing: np.ndarray, fillna_value=0.0) -> np.ndarray:
    """
    Realiza la imputación backward-fill en un array 3D.

    Aplica backward-fill para imputar valores faltantes (NaNs) en cada muestra
    del array de entrada.

    Args:
        X_missing (np.ndarray): Array NumPy 3D de entrada con valores faltantes (NaNs).
                                 Forma: (n_samples, n_steps, n_features).
        fillna_value (float, optional): Valor para rellenar los NaNs restantes después del
                                       backward fill (por ejemplo, columnas completamente NaN).
                                       Por defecto es 0.0.

    Returns:
        np.ndarray: Array NumPy 3D con backward-fill aplicado.

    Raises:
        ValueError: Si X_missing no es un array NumPy 3D.
    """
    if not isinstance(X_missing, np.ndarray) or X_missing.ndim != 3:
        raise ValueError("X_missing debe ser un array NumPy 3D.")

    X_filled_backward = X_missing.copy()
    n_samples, n_steps, n_features = X_missing.shape

    for i in range(n_samples):
        sample_df = pd.DataFrame(X_missing[i, :, :])  # Convierte el slice paso x característica a DataFrame

           # Backward fill
        filled_b = sample_df.bfill(axis=0)
        # Reserva para NaNs restantes (columnas totalmente NaN)
        if filled_b.isnull().any().any():
            filled_b.fillna(fillna_value, inplace=True)
        X_filled_backward[i, :, :] = filled_b.to_numpy()

    return X_filled_backward

@timeit
def impute_linear(X_missing: np.ndarray) -> np.ndarray:
    """
    Realiza la interpolación lineal para imputar valores faltantes en un array 3D.

    Aplica la interpolación lineal a lo largo del eje del tiempo (axis=0) para cada muestra
    en el array de entrada.

    Args:
        X_missing (np.ndarray): Array NumPy 3D de entrada con valores faltantes (NaNs).
                                 Forma: (n_samples, n_steps, n_features).

    Returns:
        np.ndarray: Array NumPy 3D con valores faltantes imputados usando interpolación lineal.
    """
    X_interpolated = X_missing.copy()
    n_samples, n_steps, n_features = X_missing.shape

    for i in range(n_samples):
        sample_df = pd.DataFrame(X_missing[i, :, :])
        interpolated_df = sample_df.interpolate(method='linear', axis=0, limit_direction='both')
        # Verifica si quedan NaNs (ej., si inicio/fin son NaN O si toda la columna era NaN)
        # Reserva: rellena los NaNs restantes (ej., con 0 o ffill/bfill)
        if interpolated_df.isnull().any().any():
             # Opción 1: Usa ffill/bfill (aún podría fallar si toda la columna es NaN)
             # interpolated_df = interpolated_df.ffill(axis=0).bfill(axis=0)
             # Opción 2: Rellena con una constante (como 0)
             interpolated_df.fillna(0.0, inplace=True)

        X_interpolated[i, :, :] = interpolated_df.to_numpy()

    return X_interpolated


@timeit
def impute_transformer(dataset_for_training, dataset_for_validating, dataset_for_testing):
    """
    Configura, entrena y usa el modelo Transformer para imputación.
    Modifica el diccionario global `imputed_results`.
    """
    print("\n---> Ejecutando lógica de imputación con Transformer...")

    transformer_config = TRANSFORMER_PARAMS.copy()
    # Asegúrate de que processed_data está disponible globalmente o pásalo si es necesario
    transformer_config['n_features'] = dataset_for_training['X'].shape[2]

    # Initialize the Transformer model
    print("    Inicializando Transformer model...")
    # Asegúrate de que tu clase Transformer real está disponible/importada
    transformer_model = Transformer(**transformer_config) # Usa tu clase real

    # Set up optimizer, scheduler, and loss function
    print("    Configurando optimizer, scheduler, loss...")
    # Asegúrate de que tus clases reales (Adam, ReduceLROnPlateau, SmoothL1Loss) están disponibles/importadas
    optimizer = Adam(transformer_model.model.parameters(), **TRANSFORMER_OPTIMIZER_PARAMS)
    scheduler = ReduceLROnPlateau(optimizer, **TRANSFORMER_SCHEDULER_PARAMS)
    loss_func = SmoothL1Loss(**TRANSFORMER_LOSS_PARAMS)

    # Train the model
    print("    Training Transformer model...")
    # Asegúrate de que dataset_for_training y dataset_for_validating están disponibles
    transformer_model.fit(
            train_set=dataset_for_training,
            val_set=dataset_for_validating, # Pass validation set for early stopping
    )
    print("    Transformer training complete.")

    # Predecir (imputar) en el conjunto de test
    print("    Imputando con Transformer en el conjunto de test...")
    transformer_prediction = transformer_model.predict(dataset_for_testing)
    print("    Imputación Transformer completa.")

    return transformer_prediction["imputation"]

@timeit
def impute_saits(dataset_for_training, dataset_for_validating, dataset_for_testing):
    """
    Configura, entrena y usa el modelo SAITS para imputación.
    Modifica el diccionario global `imputed_results`.
    """

    print("\nConfigurando y ejecutando imputación con SAITS...")

    # Configurar modelo SAITS
    saits_config = SAITS_PARAMS.copy()
    saits_config['n_features'] = dataset_for_testing['X'].shape[2]

    saits_model = SAITS(**saits_config)

    # Configurar optimizador, planificador y función de pérdida
    optimizer = Adam(saits_model.model.parameters(), **SAITS_OPTIMIZER_PARAMS)
    scheduler = ReduceLROnPlateau(optimizer, **SAITS_SCHEDULER_PARAMS)
    loss_func = SmoothL1Loss(**SAITS_LOSS_PARAMS)

    # Entrenar el modelo
    # Nota: Podrías querer ajustar epochs/patience para ejecuciones más rápidas
    print("Entrenando modelo SAITS...")
    saits_model.fit(
        train_set=dataset_for_training,
        val_set=dataset_for_validating, # Pasar conjunto de validación para early stopping
    )
    print("Entrenamiento de SAITS completo.")

    # Predecir (imputar) en el conjunto de test
    print("Imputando con SAITS en el conjunto de test...")
    saits_prediction = saits_model.predict(dataset_for_testing)
    return saits_prediction["imputation"]