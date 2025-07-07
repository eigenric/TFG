# pampaneira_imputation/data_preprocessor.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Dict, Tuple, Optional, List
from . import config
from .utils import sliding_window, create_missingness


def fill_missing_timestamps(
    df: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    freq: str = 'h',
    date_col: str = config.DATE_COL
) -> pd.DataFrame:
    """
    Rellena las marcas de tiempo horarias faltantes en un DataFrame con NaNs.
    """
    
    full_date_range = pd.date_range(
        start=start_date, end=end_date, freq=freq, tz=config.TIMEZONE
    )
    df_reindexed = df.set_index(date_col).reindex(full_date_range) # Removed reset_index()
    df_reindexed.index.name = None  # Set index name to None

    return df_reindexed

def split_by_period(
    df: pd.DataFrame,
    period_1_start: pd.Timestamp = config.PERIOD_1_START,
    period_1_end: pd.Timestamp = config.PERIOD_1_END,
    period_2_start: pd.Timestamp = config.PERIOD_2_START,
    period_2_end: pd.Timestamp = config.PERIOD_2_END,
    date_col: str = config.DATE_COL,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Divide el DataFrame en dos periodos basados en fechas predefinidas.

    Args:
        df (pd.DataFrame): DataFrame de entrada con una columna de fecha
                           o DatetimeIndex.
        period_1_start (pd.Timestamp, optional): Fecha de inicio del Periodo 1
                                                 (por defecto: config.PERIOD_1_START).
        period_1_end (pd.Timestamp, optional): Fecha de fin del Periodo 1
                                               (por defecto: config.PERIOD_1_END).
        period_2_start (pd.Timestamp, optional): Fecha de inicio del Periodo 2
                                                 (por defecto: config.PERIOD_2_START).
        period_2_end (pd.Timestamp, optional): Fecha de fin del Periodo 2
                                               (por defecto: config.PERIOD_2_END).
        date_col (str, optional): Nombre de la columna de fecha si no se usa el índice
                                  (por defecto: config.DATE_COL).

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Tupla que contiene el DataFrame del
                                          Periodo 1 y el DataFrame del Periodo 2.
    """
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        # Asume que date_col es el índice si no está ya establecido
        if date_col in df.columns:
            df = df.set_index(date_col)
        else:
            raise ValueError(
                f"'{date_col}' no se encuentra como índice o columna para dividir."
            )

    periodo_1 = df.loc[period_1_start:period_1_end].copy()
    periodo_2 = df.loc[period_2_start:period_2_end].copy()
    return periodo_1, periodo_2


def preprocess_for_imputation(
    df: pd.DataFrame,
    feature_cols: List[str] = config.FEATURE_COLUMNS,
    train_start: str = config.TRAIN_START_DATE,
    train_end: str = config.TRAIN_END_DATE,
    val_start: str = config.VAL_START_DATE,
    val_end: str = config.VAL_END_DATE,
    test_start: str = config.TEST_START_DATE,
    test_end: str = config.TEST_END_DATE,
    n_steps: int = config.N_STEPS,
    missing_rate: float = config.MISSING_RATE,
    missing_pattern: str = config.MISSING_PATTERN,
    random_seed: int = 42, # Semilla para la creación de datos faltantes
    **missingness_kwargs,
) -> Dict:
    """
    Prepara los datos para modelos de imputación: divide, escala, aplica ventanas
    e introduce datos faltantes artificialmente (si missing_rate > 0).

    Calcula máscaras explícitas para:
    - NaNs preexistentes (antes de la introducción artificial).
    - NaNs totales (indicadora) después de la introducción artificial.
    - NaNs *artificiales* (solo los introducidos en este paso).

    Args:
        df (pd.DataFrame): DataFrame con DatetimeIndex y columnas de características.
        feature_cols (List[str]): Columnas a usar como características.
        train_start (str): Fecha inicio entrenamiento.
        train_end (str): Fecha fin entrenamiento.
        val_start (str): Fecha inicio validación.
        val_end (str): Fecha fin validación.
        test_start (str): Fecha inicio prueba.
        test_end (str): Fecha fin prueba.
        n_steps (int): Tamaño de la ventana deslizante (longitud de secuencia).
        missing_rate (float): Proporción de valores a convertir en NaN
                                (0 para desactivar).
        missing_pattern (str): Patrón: 'point', 'subseq' o 'block'.
        random_seed (int): Semilla para la generación de datos faltantes.
        **missingness_kwargs: Argumentos adicionales para create_missingness
                              (ej., min_seq, max_seq).

    Returns:
        Dict: Diccionario con datos procesados:
              - 'train_X', 'val_X', 'test_X': Datos finales con NaNs (originales + artificiales).
              - 'train_X_ori', 'val_X_ori', 'test_X_ori': Datos antes de introducir NaNs artificiales.
              - 'scaler': Objeto StandardScaler ajustado.
              - 'n_steps', 'n_features': Dimensiones de las secuencias.
              - 'train_preexisting_nan_mask', etc.: Máscara de NaNs preexistentes (0/1).
              - 'train_indicating_mask', etc.: Máscara de NaNs totales (0/1).
              - 'train_artificial_mask', etc.: Máscara SOLO de NaNs artificiales (0/1). <- Clave para evaluación.
              - 'n_train_samples', etc.: Número de muestras (ventanas) en cada conjunto.
              (Se eliminaron índices y dataframes intermedios para simplificar).
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("El índice del DataFrame debe ser un DatetimeIndex.")
    if not all(col in df.columns for col in feature_cols):
        missing = [col for col in feature_cols if col not in df.columns]
        raise ValueError(f"Las siguientes feature_cols no están en el DataFrame: {missing}")

    # --- 1. División por Fechas ---
    print(f"Train Start_End: {train_start}:{train_end}")
    train_set = df.loc[train_start:train_end, feature_cols]
    val_set = df.loc[val_start:val_end, feature_cols]
    test_set = df.loc[test_start:test_end, feature_cols]

    if train_set.empty or val_set.empty or test_set.empty:
        raise ValueError(
            "Una o más divisiones de datos están vacías. "
            "Verifica los rangos de fechas y los datos de entrada."
        )

     # Guardar los índices originales ANTES del escalado y ventaneo
    train_index_original = train_set.index
    val_index_original = val_set.index
    test_index_original = test_set.index

    # --- 2. Escalado ---
    # Nota: StandardScaler no maneja bien los NaNs. Idealmente, los NaNs
    # originales se imputan *antes* de escalar o se usa un escalador robusto.
    # Aquí asumimos que se manejan o que el impacto es aceptable para el caso de uso.
    scaler = StandardScaler()
    train_X_scaled = scaler.fit_transform(train_set)
    val_X_scaled = scaler.transform(val_set)
    test_X_scaled = scaler.transform(test_set)

    # --- 3. Ventana Deslizante ---
    train_X_win = sliding_window(train_X_scaled, n_steps)
    val_X_win = sliding_window(val_X_scaled, n_steps)
    test_X_win = sliding_window(test_X_scaled, n_steps)

    if train_X_win.size == 0 or val_X_win.size == 0 or test_X_win.size == 0:
         raise ValueError(
             f"Una o más divisiones resultaron vacías después de aplicar la ventana "
             f"deslizante con n_steps={n_steps}. Asegúrate de que cada conjunto "
             f"original tenga al menos {n_steps} puntos de datos."
         )

    n_features = train_X_win.shape[-1]

    # Prepara el diccionario de resultados inicial
    processed_data = {
        "n_steps": n_steps,
        "n_features": n_features,
        "scaler": scaler,
        "train_X_ori": train_X_win.copy(), # Guarda antes de NaNs artificiales
        "val_X_ori": val_X_win.copy(),
        "test_X_ori": test_X_win.copy(),
        # Estos se sobreescribirán si missing_rate > 0
        "train_X": train_X_win,
        "val_X": val_X_win,
        "test_X": test_X_win,
        "train_index": train_index_original,
        "val_index": val_index_original,
        "test_index": test_index_original,
    }

    # --- 4. Introducción Opcional de Datos Faltantes y Creación de Máscaras ---
    if missing_rate > 0:
        print(
            f"Introduciendo {missing_rate*100:.1f}% de datos faltantes "
            f"con patrón '{missing_pattern}' (seed={random_seed})..."
        )

        # a) Máscara de NaNs PREEXISTENTES (antes de la introducción artificial)
        train_preexisting_nan_mask = np.isnan(processed_data["train_X_ori"])
        val_preexisting_nan_mask = np.isnan(processed_data["val_X_ori"])
        test_preexisting_nan_mask = np.isnan(processed_data["test_X_ori"])
        processed_data["train_preexisting_nan_mask"] = train_preexisting_nan_mask.astype(int)
        processed_data["val_preexisting_nan_mask"] = val_preexisting_nan_mask.astype(int)
        processed_data["test_preexisting_nan_mask"] = test_preexisting_nan_mask.astype(int)

        # b) Introducir NaNs artificiales
        # Pasamos la semilla a create_missingness para controlar su aleatoriedad interna
        train_X_missing = create_missingness(
            processed_data["train_X_ori"], missing_rate, missing_pattern,
            seed=random_seed, **missingness_kwargs
        )
        # Para val y test, podríamos usar semillas diferentes si quisiéramos patrones distintos,
        # pero usar la misma semilla + offset o ninguna semilla interna suele estar bien.
        # Aquí, para simplicidad, no se pasa semilla explícita (usará la global o será aleatorio).
        val_X_missing = create_missingness(
            processed_data["val_X_ori"], missing_rate, missing_pattern,
            seed=random_seed+1, **missingness_kwargs # Podrías añadir seed=random_seed+1 si quieres diferenciar
        )
        test_X_missing = create_missingness(
            processed_data["test_X_ori"], missing_rate, missing_pattern,
            seed=random_seed+1, **missingness_kwargs # Podrías añadir seed=random_seed+2
        )

        # Actualiza los datos en el diccionario
        processed_data["train_X"] = train_X_missing
        processed_data["val_X"] = val_X_missing
        processed_data["test_X"] = test_X_missing

        # c) Máscara indicadora TOTAL (NaNs originales + artificiales)
        train_indicating_mask = np.isnan(train_X_missing)
        val_indicating_mask = np.isnan(val_X_missing)
        test_indicating_mask = np.isnan(test_X_missing)
        processed_data["train_indicating_mask"] = train_indicating_mask.astype(int)
        processed_data["val_indicating_mask"] = val_indicating_mask.astype(int)
        processed_data["test_indicating_mask"] = test_indicating_mask.astype(int)

        # d) Máscara de NaNs ARTIFICIALES (SOLO los introducidos ahora)
        # Es True donde NO era NaN antes (~preexisting) Y SÍ es NaN ahora (indicating)
        train_artificial_nan_mask = ~train_preexisting_nan_mask & train_indicating_mask
        val_artificial_nan_mask = ~val_preexisting_nan_mask & val_indicating_mask
        test_artificial_nan_mask = ~test_preexisting_nan_mask & test_indicating_mask
        processed_data["train_artificial_mask"] = train_artificial_nan_mask.astype(int)
        processed_data["val_artificial_mask"] = val_artificial_nan_mask.astype(int)
        processed_data["test_artificial_mask"] = test_artificial_nan_mask.astype(int)

    else: # Caso sin introducción artificial de NaNs
        print("La tasa de datos faltantes es 0. No se introducen NaNs artificiales.")

        # Las máscaras preexistentes e indicadoras son las mismas
        train_preexisting_nan_mask = np.isnan(processed_data["train_X"])
        val_preexisting_nan_mask = np.isnan(processed_data["val_X"])
        test_preexisting_nan_mask = np.isnan(processed_data["test_X"])
        processed_data["train_preexisting_nan_mask"] = train_preexisting_nan_mask.astype(int)
        processed_data["val_preexisting_nan_mask"] = val_preexisting_nan_mask.astype(int)
        processed_data["test_preexisting_nan_mask"] = test_preexisting_nan_mask.astype(int)

        processed_data["train_indicating_mask"] = processed_data["train_preexisting_nan_mask"]
        processed_data["val_indicating_mask"] = processed_data["val_preexisting_nan_mask"]
        processed_data["test_indicating_mask"] = processed_data["test_preexisting_nan_mask"]

        # La máscara artificial es todo ceros
        shape_train = processed_data["train_X"].shape
        shape_val = processed_data["val_X"].shape
        shape_test = processed_data["test_X"].shape
        processed_data["train_artificial_mask"] = np.zeros(shape_train, dtype=int)
        processed_data["val_artificial_mask"] = np.zeros(shape_val, dtype=int)
        processed_data["test_artificial_mask"] = np.zeros(shape_test, dtype=int)

    # Añade el número de muestras finales
    processed_data["n_train_samples"] = processed_data["train_X"].shape[0]
    processed_data["n_val_samples"] = processed_data["val_X"].shape[0]
    processed_data["n_test_samples"] = processed_data["test_X"].shape[0]

    return processed_data