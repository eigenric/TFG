# pampaneira_imputation/utils.py
import numpy as np
import pandas as pd
from typing import List

def sliding_window(data: np.ndarray, n_steps: int) -> np.ndarray:
    """
    Crea ventanas deslizantes a partir de datos secuenciales.

    Args:
        data (np.ndarray): Array NumPy 2D o 3D de datos secuenciales.
                           Si es 2D, se asume forma (n_timesteps, n_features).
                           Si es 3D, se asume forma (n_samples, n_timesteps, n_features).
        n_steps (int): Tamaño de cada ventana deslizante.

    Returns:
        np.ndarray: Array NumPy 3D de ventanas deslizantes.
                    Forma: (n_windows, n_steps, n_features) si la entrada era 2D,
                           o (n_samples, n_windows, n_steps, n_features) si la entrada era 3D.
                           En este caso, siempre devuelve 3D con forma (n_windows, n_steps, n_features).
    """
    X = []
    for i in range(len(data) - n_steps + 1):
        X.append(data[i: i + n_steps])
    return np.array(X)

def create_missingness(data: np.ndarray, rate: float, pattern: str, **kwargs) -> np.ndarray:
    """
    Introduce valores faltantes (NaN) en un array NumPy 3D (muestras, pasos, características).

    Modifica el array in situ por eficiencia, pero también lo devuelve.

    Args:
        data (np.ndarray): Array NumPy 3D de entrada. Forma: (n_samples, n_steps, n_features).
        rate (float): Tasa de datos faltantes a introducir (entre 0 y 1).
        pattern (str): Patrón de datos faltantes: 'point', 'subseq' o 'block'.
        **kwargs: Argumentos adicionales dependientes del patrón:
            - Para 'subseq': min_missing_len, max_missing_len.
            - Para 'block': min_missing_len, max_missing_len.

    Returns:
        np.ndarray: Array NumPy 3D con valores faltantes introducidos.
    """
    n_samples, n_steps, n_features = data.shape
    missing_data = data.copy()  # Trabaja sobre una copia

    if pattern == "point":
        total_elements = n_samples * n_steps * n_features
        n_missing = int(total_elements * rate)
        indices = np.random.choice(total_elements, n_missing, replace=False)
        coords = np.unravel_index(indices, (n_samples, n_steps, n_features))
        missing_data[coords] = np.nan

    elif pattern == "subseq":
         # Más complejo: Introduce subsecuencias faltantes por característica por muestra
         # Esta implementación introduce datos faltantes *independientemente* por característica
         # Ajusta si se desea un bloque de datos faltantes a través de las características
         min_len = kwargs.get('min_missing_len', 1)
         max_len = kwargs.get('max_missing_len', n_steps // 4)  # Longitud máxima de ejemplo

         for i in range(n_samples):
             for k in range(n_features):
                 # Decide si esta secuencia de características tendrá datos faltantes
                 if np.random.rand() < rate:  # Probabilidad 'rate' para añadir bloque de datos faltantes
                     length = np.random.randint(min_len, max_len + 1)
                     start_idx = np.random.randint(0, n_steps - length + 1)
                     missing_data[i, start_idx: start_idx + length, k] = np.nan

    elif pattern == "block":
         # Introduce datos faltantes en bloques comenzando desde el índice 0
         # Esta interpretación podría diferir; ajusta si es necesario.
         min_len = kwargs.get('min_missing_len', 1)
         max_len = kwargs.get('max_missing_len', n_steps // 2)  # Longitud máxima de ejemplo

         for i in range(n_samples):
             for k in range(n_features):
                if np.random.rand() < rate:  # Probabilidad 'rate' para añadir bloque de datos faltantes
                    block_size = np.random.randint(min_len, max_len + 1)
                    missing_data[i, :block_size, k] = np.nan
    else:
        raise ValueError(f"Patrón de datos faltantes desconocido: {pattern}")

    return missing_data


def reshape_imputed_to_df(imputed_data: np.ndarray,
                          original_index: pd.DatetimeIndex,
                          columns: List[str],
                          n_steps: int) -> pd.DataFrame:
    """
    Remodela los datos imputados 3D (muestras, pasos, características) de vuelta a un DataFrame 2D.

    Asume que el índice original corresponde al *inicio* de cada ventana.
    Reconstruye la línea de tiempo completa.

    Args:
        imputed_data (np.ndarray): Array NumPy 3D de datos imputados.
                                   Forma: (n_samples, n_steps, n_features).
        original_index (pd.DatetimeIndex): DatetimeIndex original correspondiente al conjunto de datos a imputar.
        columns (List[str]): Lista de nombres de columnas para el DataFrame reconstruido.
        n_steps (int): Número de pasos de tiempo en cada ventana.

    Returns:
        pd.DataFrame: DataFrame de Pandas reconstruido a partir de los datos imputados.
    """
    n_samples, _, n_features = imputed_data.shape
    if len(columns) != n_features:
         raise ValueError(f"El número de columnas ({len(columns)}) debe coincidir con el número de características ({n_features})")

    # Reconstrucción ingenua: Usa el último valor de cada ventana para su paso de tiempo correspondiente
    # Un enfoque más sofisticado podría promediar ventanas superpuestas, pero esto es más simple.
    # Esto toma efectivamente la imputación para el *último* paso de tiempo en cada ventana.
    # reconstructed_data = imputed_data[:, -1, :]

    # Alternativa: Aplana y asume ventanas contiguas (manejar con cuidado para superposición/huecos)
    # Esto coincide con la lógica de aplanamiento aparente del script original
    flat_data = imputed_data.reshape(-1, n_features)

    # Necesita reconstruir el índice cuidadosamente basado en la división original
    # El script original usaba implícitamente el índice del test_set *antes* del ventaneo
    # Asumamos que el `original_index` proporcionado es el índice de los datos de prueba *aplanados*
    # antes del ventaneo. Necesitamos suficientes puntos de índice para los datos aplanados.
    num_expected_rows = n_samples * n_steps  # o simplemente len(flat_data)
    if len(original_index) < num_expected_rows:
        # Esto sugiere que el índice original era quizás del *inicio* de las ventanas.
        # Reconstruyamos un índice completo basado en el inicio y la frecuencia.
        if isinstance(original_index, pd.DatetimeIndex) and original_index.freq:
            full_index = pd.date_range(start=original_index[0], periods=num_expected_rows, freq=original_index.freq)
        else:
            # No se puede reconstruir el índice de manera fiable sin información de frecuencia o el índice original correcto
            print("Advertencia: No se puede reconstruir el índice del DataFrame de manera fiable. Usando índice de rango.")
            full_index = pd.RangeIndex(stop=num_expected_rows)

    else:
        # Asume que original_index cubre todos los pasos de tiempo en la salida aplanada
        full_index = original_index[:num_expected_rows]


    df = pd.DataFrame(flat_data, index=full_index, columns=columns)
    return df