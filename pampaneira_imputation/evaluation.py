# pampaneira_imputation/evaluation.py
import numpy as np
import pandas as pd
from pypots.nn.functional import calc_mae, calc_mse, calc_rmse, calc_mre
from typing import Dict, Tuple, List
from . import config

def calculate_imputation_metrics(y_true: np.ndarray,
                                 y_pred: np.ndarray,
                                 indicating_mask: np.ndarray) -> Dict[str, float]:
    """
    Calcula MAE, MSE, RMSE, MRE para valores imputados donde la máscara es 1.

    Args:
        y_true (np.ndarray): Datos verdaderos (potencialmente con NaNs donde faltaban originalmente).
        y_pred (np.ndarray): Datos imputados.
        indicating_mask (np.ndarray): Máscara donde 1 indica un valor faltante que fue imputado,
                                     0 indica un valor observado.

    Returns:
        Dict[str, float]: Diccionario que contiene 'mae', 'mse', 'rmse', 'mre'.
    """
    # Asegura que las entradas sean arrays numpy
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    indicating_mask = np.asarray(indicating_mask)

    # Reemplaza NaNs en la verdad fundamental con 0 para el cálculo donde la máscara es 1
    # Esto es necesario porque las funciones pypots esperan una verdad fundamental sin NaN
    # Solo evaluamos donde indicating_mask es 1, por lo que este reemplazo es seguro.
    y_true_filled = np.nan_to_num(y_true, nan=0.0)

    if y_true.shape != y_pred.shape or y_true.shape != indicating_mask.shape:
        raise ValueError(f"Desajuste de forma: y_true={y_true.shape}, "
                         f"y_pred={y_pred.shape}, mask={indicating_mask.shape}")

    # Verifica si la suma de la máscara es cero (no hay valores para evaluar)
    if indicating_mask.sum() == 0:
        print("Advertencia: La suma de la máscara indicadora es 0. No hay valores imputados para evaluar.")
        return {'mae': np.nan, 'mse': np.nan, 'rmse': np.nan, 'mre': np.nan}

    try:
        mae = calc_mae(y_pred, y_true_filled, indicating_mask)
        mse = calc_mse(y_pred, y_true_filled, indicating_mask)
        rmse = calc_rmse(y_pred, y_true_filled, indicating_mask)
        mre = calc_mre(y_pred, y_true_filled, indicating_mask)  # Precaución con MRE si los valores verdaderos están cerca de cero

        return {'mae': mae, 'mse': mse, 'rmse': rmse, 'mre': mre}
    except Exception as e:
         print(f"Error durante el cálculo de métricas: {e}")
         # Añade más información de depuración si es necesario
         print(f"Formas: y_pred={y_pred.shape}, y_true_filled={y_true_filled.shape}, mask={indicating_mask.shape}")
         print(f"Suma de máscara: {indicating_mask.sum()}")
         print(f"Conteo de NaN: pred={np.isnan(y_pred).sum()}, true_filled={np.isnan(y_true_filled).sum()}, mask={np.isnan(indicating_mask).sum()}")
         # Considera verificar también por infinitos
         return {'mae': np.nan, 'mse': np.nan, 'rmse': np.nan, 'mre': np.nan}

def evaluate_all_methods(preprocessed_data: Dict,
                         imputed_results: Dict[str, np.ndarray],
                         methods_to_evaluate: list = ['median', 'mean', 'linear', 'ffill', 'bfill', 'transformer', 'saits']) -> pd.DataFrame:
    """
    Evalúa múltiples métodos de imputación usando los resultados del conjunto de prueba.

    Args:
        preprocessed_data (Dict): Diccionario de preprocess_for_imputation.
        imputed_results (Dict[str, np.ndarray]): Diccionario que mapea nombres de métodos a arrays NumPy imputados (conjunto de prueba).
        methods_to_evaluate (list, optional): Lista de claves en imputed_results para evaluar.
                                              (por defecto: ['median', 'mean', 'linear', 'ffill', 'bfill', 'saits'])

    Returns:
        pd.DataFrame: DataFrame de Pandas que resume MAE, MSE, RMSE, MRE para cada método.
    """
    results = []
    y_true = preprocessed_data['test_X_ori']
    indicating_mask = preprocessed_data['test_artificial_mask']

    # Maneja la posible eliminación de columnas para ffill/bfill si es necesario
    # Esta lógica asume que WS/WD se eliminaron *antes* de la imputación para ffill/bfill
    cols_to_drop_indices = [config.FEATURE_COLUMNS.index(col) for col in config.COLS_TO_DROP_FOR_BASELINE if col in config.FEATURE_COLUMNS]

    for method_name in methods_to_evaluate:
        if method_name not in imputed_results:
            print(f"Advertencia: No se encontraron resultados imputados para el método '{method_name}'. Saltando.")
            continue

        y_pred = imputed_results[method_name]
        current_y_true = y_true
        current_mask = indicating_mask

        # Manejo específico para métodos donde las columnas podrían haberse eliminado
        if method_name in ['ffill', 'bfill'] and cols_to_drop_indices:
             print(f"Ajustando datos verdaderos y máscara para {method_name} debido a columnas eliminadas para {method_name}.")
             current_y_true = np.delete(y_true, cols_to_drop_indices, axis=2)
             current_mask = np.delete(indicating_mask, cols_to_drop_indices, axis=2)
             # y_pred para estos métodos ya debería tener las columnas eliminadas

        print(f"\nCalculando métricas para: {method_name}")
        metrics = calculate_imputation_metrics(current_y_true, y_pred, current_mask)

        results.append({
            "Method": method_name.replace('_', ' ').title(),  # Nombre más bonito
            "RMSE": metrics.get('rmse', np.nan),
            "MSE": metrics.get('mse', np.nan),
            "MAE": metrics.get('mae', np.nan),
            "MRE": metrics.get('mre', np.nan)
        })

    error_table = pd.DataFrame.from_records(results)
    error_table = error_table.set_index("Method").round(4)
    return error_table