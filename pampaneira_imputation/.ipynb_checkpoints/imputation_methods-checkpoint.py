# pampaneira_imputation/imputation_methods.py
import numpy as np
import pandas as pd
import warnings

from scipy.linalg import hankel, svd
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

# Usaremos una versión de create_hankel_matrix que maneje NaNs
# y que no dependa de scipy.linalg.hankel directamente para flexibilidad con NaNs,
# pero el concepto es el mismo que subyace a scipy.linalg.hankel.
def _create_hankel_from_flattened(X_flat, k_rows):
    """
    Crea una matriz de Hankel a partir de una serie temporal 1D (aplanada).
    La implementación se centra en manejar correctamente los NaNs presentes en X_flat
    y construir la matriz de Hankel esperada para la completación.
    
    Args:
        X_flat (np.ndarray): Serie temporal 1D aplanada, puede contener NaNs.
        k_rows (int): Número de filas de la matriz de Hankel (lag).
    Returns:
        np.ndarray: La matriz de Hankel.
    Raises:
        ValueError: Si k_rows es inválido.
    """
    T_flat = len(X_flat)
    if not (1 <= k_rows <= T_flat):
        raise ValueError(f"El número de filas de Hankel (k_rows) debe estar entre 1 y la longitud aplanada ({T_flat}), pero k_rows={k_rows}")
    
    num_cols_H = T_flat - k_rows + 1
    
    H = np.full((k_rows, num_cols_H), np.nan, dtype=X_flat.dtype) # Asegurar el tipo de dato
    for i in range(k_rows):
        H[i, :] = X_flat[i : i + num_cols_H]
    return H

def _reconstruct_from_hankel_to_flattened(H_imputed, T_flat):
    """
    Reconstruye una serie temporal 1D aplanada a partir de una matriz de Hankel imputada
    promediando los valores superpuestos.
    """
    k_rows, num_cols = H_imputed.shape
    
    X_flat_reconstructed = np.zeros(T_flat, dtype=H_imputed.dtype)
    counts = np.zeros(T_flat, dtype=int) # Usamos int para los contadores

    for i in range(k_rows):
        for j in range(num_cols):
            idx = i + j
            if idx < T_flat: # Asegurar que no se exceda el tamaño original
                X_flat_reconstructed[idx] += H_imputed[i, j]
                counts[idx] += 1
    
    # Evitar divisiones por cero donde no hay datos
    counts_safe = np.where(counts == 0, 1, counts) # Reemplazar 0 por 1 para evitar ZeroDivisionError
    X_flat_reconstructed = X_flat_reconstructed / counts_safe
    
    return X_flat_reconstructed

# --- Métodos de Imputación Existentes (sin cambios) ---

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
def impute_hankel(X_missing: np.ndarray, k: Optional[int] = None, max_iter: int = 1, tol: float = 1e-1, tau_scaling: float = 0.1) -> np.ndarray:
    """
    Imputa valores faltantes en un array 3D usando Hankel Imputation (HI)
    basado en la completación de matrices de bajo rango mediante Singular Value Thresholding (SVT).

    Aplica este método a cada muestra (sample_idx) del array 3D de entrada.

    Args:
        X_missing (np.ndarray): Array NumPy 3D de entrada con valores faltantes (NaNs).
                                 Forma: (n_samples, n_steps, n_features).
        k (int, optional): Número de filas para la construcción de la matriz de Hankel
                           para la serie aplanada de cada muestra. Si es None, se calcula
                           automáticamente: k = floor((T_flat + 1) / 2), donde T_flat es
                           la longitud aplanada de la muestra (n_steps * n_features).
        max_iter (int): Número máximo de iteraciones para el algoritmo SVT.
        tol (float): Tolerancia para la convergencia del algoritmo SVT (cambio relativo en la norma de Frobenius).
        tau_scaling (float): Factor de escalado para el parámetro de umbral suave (tau).
                             tau = tau_scaling * max_singular_value. Un valor más alto
                             fuerza un rango más bajo (más imputación suavizada).

    Returns:
        np.ndarray: Array NumPy 3D con valores NaN imputados.
    """
    print("Iniciando Hankel Imputation (HI) para imputación.")
    
    if not isinstance(X_missing, np.ndarray) or X_missing.ndim != 3:
        print("ERROR: X_missing debe ser un array NumPy 3D. Devolviendo entrada original.")
        return X_missing # Devolver original si el formato es incorrecto

    imputed_data = X_missing.copy()
    n_samples, n_steps, n_features = X_missing.shape

    print(f"Procesando {n_samples} muestras. Cada muestra tiene forma ({n_steps}, {n_features}).")

    for sample_idx in range(n_samples):
        current_sample = X_missing[sample_idx, :, :]
        X_sample_flat = current_sample.flatten()
        T_flat_sample = len(X_sample_flat)

        k_rows_sample = k
        if k_rows_sample is None:
            k_rows_sample = int(np.floor((T_flat_sample + 1) / 2))
            if k_rows_sample < 1:
                k_rows_sample = 1
            print(f"    Muestra {sample_idx}/{n_samples-1}: Lag k no especificado. Calculado como {k_rows_sample} para longitud aplanada {T_flat_sample}.")
        else:
             print(f"    Muestra {sample_idx}/{n_samples-1}: Usando lag k especificado: {k_rows_sample}.")
        
        if not (1 <= k_rows_sample <= T_flat_sample):
             print(f"    ADVERTENCIA: Lag k={k_rows_sample} inválido para muestra {sample_idx} (longitud aplanada {T_flat_sample}). Ajustando k a {T_flat_sample // 2 if T_flat_sample > 1 else 1}.")
             k_rows_sample = T_flat_sample // 2 if T_flat_sample > 1 else 1
             if k_rows_sample == 0: k_rows_sample = 1

        try:
            H_missing_true_nan = _create_hankel_from_flattened(X_sample_flat, k_rows_sample)
        except ValueError as e:
            print(f"    ERROR: No se pudo crear la matriz de Hankel para muestra {sample_idx} (k={k_rows_sample}, T_flat={T_flat_sample}): {e}. Saltando esta muestra.")
            continue 

        H_mask = ~np.isnan(H_missing_true_nan)

        H_imputed_current = H_missing_true_nan.copy()
        initial_fill_value = np.nanmean(X_sample_flat)
        if np.isnan(initial_fill_value):
            initial_fill_value = 0
        H_imputed_current[~H_mask] = initial_fill_value

        print(f"    Muestra {sample_idx}/{n_samples-1}: Iniciando SVT (max_iter={max_iter}, tol={tol:.1e}, tau_scaling={tau_scaling}).")
        
        # Algoritmo SVT iterativo
        for iteration in range(max_iter):
            H_prev = H_imputed_current.copy()

            U, s, Vh = svd(H_imputed_current, full_matrices=False)
            
            tau = tau_scaling * s[0] if len(s) > 0 else 0
            s_thresholded = np.maximum(0, s - tau)
            
            H_new = U @ np.diag(s_thresholded) @ Vh
            
            H_imputed_current = H_new
            H_imputed_current[H_mask] = H_missing_true_nan[H_mask]

            norm_diff = np.linalg.norm(H_imputed_current - H_prev, 'fro')
            norm_prev = np.linalg.norm(H_prev, 'fro')
            
            # Evitar ZeroDivisionError si norm_prev es muy pequeño
            if norm_prev < 1e-10: # Usamos un umbral pequeño en lugar de == 0
                if norm_diff < 1e-10: # Si ambos son casi cero, consideramos convergido
                    print(f"        Iteración {iteration + 1}/{max_iter}: Norma previa muy pequeña. Convergencia asumida.")
                    break
                else: 
                    # Si norm_prev es ~0 pero norm_diff es grande, algo va mal o es el inicio y el cambio es grande.
                    # Continuamos, pero quizás una advertencia aquí podría ser útil.
                    pass # Para evitar el log spam en las primeras iteraciones

            change = norm_diff / norm_prev
            
            if (iteration + 1) % 10 == 0 or iteration == 0 or change < tol: # Log cada 10 iteraciones o al inicio/convergencia
                print(f"        Iteración {iteration + 1}/{max_iter}: Cambio relativo (Frobenius) = {change:.4e}. Tau = {tau:.4e}. Valores singulares: {s_thresholded.round(2) if len(s_thresholded) > 0 else '[]'}")
            
            if change < tol:
                print(f"    Muestra {sample_idx}/{n_samples-1}: SVT convergió en la iteración {iteration + 1}.")
                break
        else: # Este else se ejecuta si el bucle no se rompe con 'break'
            print(f"    ADVERTENCIA: Muestra {sample_idx}/{n_samples-1}: SVT no convergió después de {max_iter} iteraciones (tol={tol:.1e}).")
        
        # Reconstruir y asignar la muestra imputada
        X_sample_imputed_flat = _reconstruct_from_hankel_to_flattened(H_imputed_current, T_flat_sample)
        imputed_data[sample_idx, :, :] = X_sample_imputed_flat.reshape(n_steps, n_features)

    print("Hankel Imputation (HI) completada para todas las muestras.")
    return imputed_data


@timeit
def fit_transformer(dataset_for_training, dataset_for_validating):
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
    return transformer_model

@timeit
def impute_transformer(transformer_model, dataset_for_testing):
    """
    Configura, entrena y usa el modelo Transformer para imputación.
    Modifica el diccionario global `imputed_results`.
    """
    # Predecir (imputar) en el conjunto de test
    print("    Imputando con Transformer en el conjunto de test...")
    transformer_prediction = transformer_model.predict(dataset_for_testing)
    print("    Imputación Transformer completa.")

    return transformer_prediction["imputation"]

@timeit
def fit_saits(dataset_for_training, dataset_for_validating):
    """
    Configura, entrena y usa el modelo SAITS para imputación.
    Modifica el diccionario global `imputed_results`.
    """

    print("\nConfigurando y ejecutando imputación con SAITS...")

    # Configurar modelo SAITS
    saits_config = SAITS_PARAMS.copy()
    saits_config['n_features'] = dataset_for_training['X'].shape[2]

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
    return saits_model

@timeit
def impute_saits(saits_model, dataset_for_testing):
    """
    Usa el modelo SAITS para imputación.
    Modifica el diccionario global `imputed_results`.
    """
    # Predecir (imputar) en el conjunto de test
    print("Imputando con SAITS en el conjunto de test...")
    saits_prediction = saits_model.predict(dataset_for_testing)
    return saits_prediction["imputation"]