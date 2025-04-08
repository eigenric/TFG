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

    Args:
        df (pd.DataFrame): DataFrame de entrada con una columna de fecha.
        start_date (pd.Timestamp): Fecha de inicio del rango completo.
        end_date (pd.Timestamp): Fecha de fin del rango completo.
        freq (str, optional): Frecuencia para el rango de fechas (por defecto: 'h').
        date_col (str, optional): Nombre de la columna de fecha (por defecto: config.DATE_COL).

    Returns:
        pd.DataFrame: DataFrame con marcas de tiempo horarias completas y NaNs
                      para los datos faltantes.
    """
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df[date_col] = pd.to_datetime(df[date_col])
    if df[date_col].dt.tz is None:
        df[date_col] = df[date_col].dt.tz_localize(
            config.TIMEZONE
        )  # Asegura la zona horaria
    df = df.set_index(date_col)
    full_date_range = pd.date_range(
        start=start_date, end=end_date, freq=freq, tz=config.TIMEZONE
    )
    df_reindexed = df.reindex(full_date_range)
    # No reinicies el índice si quieres preservar el DatetimeIndex
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


def add_period1_padding(
    df_period1: pd.DataFrame,
    start_pad: pd.Timestamp = config.PERIOD_1_PADDING_START,
    end_pad: pd.Timestamp = config.PERIOD_1_PADDING_END,
    freq: str = 'h',
) -> pd.DataFrame:
    """
    Añade relleno de NaNs al inicio y al final de los datos del Periodo 1.

    Args:
        df_period1 (pd.DataFrame): DataFrame del Periodo 1 con DatetimeIndex.
        start_pad (pd.Timestamp, optional): Fecha de inicio para el relleno inicial
                                           (por defecto: config.PERIOD_1_PADDING_START).
        end_pad (pd.Timestamp, optional): Fecha de fin para el relleno final
                                         (por defecto: config.PERIOD_1_PADDING_END).
        freq (str, optional): Frecuencia para el rango de fechas de relleno
                              (por defecto: 'h').

    Returns:
        pd.DataFrame: DataFrame del Periodo 1 con relleno de NaNs añadido al
                      inicio y al final.
    """
    if not isinstance(df_period1.index, pd.DatetimeIndex):
        raise ValueError("El índice del DataFrame debe ser un DatetimeIndex para el relleno.")

    # Crea DataFrames de relleno
    start_range = pd.date_range(
        start=start_pad,
        end=df_period1.index.min() - pd.Timedelta(hours=1),
        freq=freq,
        tz=config.TIMEZONE,
    )
    end_range = pd.date_range(
        start=df_period1.index.max() + pd.Timedelta(hours=1),
        end=end_pad,
        freq=freq,
        tz=config.TIMEZONE,
    )

    df_start_pad = pd.DataFrame(np.nan, index=start_range, columns=df_period1.columns)
    df_end_pad = pd.DataFrame(np.nan, index=end_range, columns=df_period1.columns)

    # Concatena
    df_padded = pd.concat([df_start_pad, df_period1, df_end_pad], axis=0)
    # Asegura que el índice esté ordenado si la concatenación desordena
    # (poco probable con rangos de fechas)
    df_padded = df_padded.sort_index()

    return df_padded

def add_period2_padding(
    df_period2: pd.DataFrame,
    start_pad: pd.Timestamp = config.PERIOD_2_PADDING_START,
    end_pad: pd.Timestamp = config.PERIOD_2_PADDING_END,
    freq: str = 'h',
) -> pd.DataFrame:
    """
    Añade relleno de NaNs al inicio y al final de los datos del Periodo 2.

    Args:
        df_period2 (pd.DataFrame): DataFrame del Periodo 2 con DatetimeIndex.
        start_pad (pd.Timestamp, optional): Fecha de inicio para el relleno inicial
                                           (por defecto: config.PERIOD_2_PADDING_START).
        end_pad (pd.Timestamp, optional): Fecha de fin para el relleno final
                                         (por defecto: config.PERIOD_2_PADDING_END).
        freq (str, optional): Frecuencia para el rango de fechas de relleno
                              (por defecto: 'h').

    Returns:
        pd.DataFrame: DataFrame del Periodo 2 con relleno de NaNs añadido al
                      inicio y al final.
    """
    if not isinstance(df_period2.index, pd.DatetimeIndex):
        raise ValueError("El índice del DataFrame debe ser un DatetimeIndex para el relleno.")

    # Crea DataFrames de relleno
    start_range = pd.date_range(
        start=start_pad,
        end=df_period2.index.min() - pd.Timedelta(hours=1),
        freq=freq,
        tz=config.TIMEZONE,
    )
    end_range = pd.date_range(
        start=df_period2.index.max() + pd.Timedelta(hours=1),
        end=end_pad,
        freq=freq,
        tz=config.TIMEZONE,
    )

    df_start_pad = pd.DataFrame(np.nan, index=start_range, columns=df_period2.columns)
    df_end_pad = pd.DataFrame(np.nan, index=end_range, columns=df_period2.columns)

    # Concatena
    df_padded = pd.concat([df_start_pad, df_period2, df_end_pad], axis=0)
    # Asegura que el índice esté ordenado si la concatenación desordena
    # (poco probable con rangos de fechas)
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
    **missingness_kwargs,
) -> Dict:
    """
    Prepara los datos para modelos de imputación: divide, escala, ventanas,
    añade datos faltantes.

    Args:
        df (pd.DataFrame): DataFrame con DatetimeIndex y columnas de características.
        feature_cols (list, optional): Lista de columnas a usar como características
                                       (por defecto: config.FEATURE_COLUMNS).
        train_start (str, optional): Fecha de inicio del conjunto de entrenamiento
                                     (por defecto: config.TRAIN_START_DATE).
        train_end (str, optional): Fecha de fin del conjunto de entrenamiento
                                   (por defecto: config.TRAIN_END_DATE).
        val_start (str, optional): Fecha de inicio del conjunto de validación
                                   (por defecto: config.VAL_START_DATE).
        val_end (str, optional): Fecha de fin del conjunto de validación
                                 (por defecto: config.VAL_END_DATE).
        test_start (str, optional): Fecha de inicio del conjunto de prueba
                                    (por defecto: config.TEST_START_DATE).
        test_end (str, optional): Fecha de fin del conjunto de prueba
                                  (por defecto: config.TEST_END_DATE).
        n_steps (int, optional): Tamaño de la ventana deslizante
                                 (por defecto: config.N_STEPS).
        missing_rate (float, optional): Proporción de valores a convertir en NaN
                                        (0 para desactivar) (por defecto: config.MISSING_RATE).
        missing_pattern (str, optional): Patrón de datos faltantes: 'point', 'subseq'
                                          o 'block' (por defecto: config.MISSING_PATTERN).
        **missingness_kwargs: Argumentos adicionales para create_missingness
                              (ej., min/max_length).

    Returns:
        Dict: Un diccionario que contiene divisiones de datos procesados
              (train_X, val_X, test_X), datos originales (sin máscara)
              (train_X_ori, etc.), el escalador, número de características,
              y número de pasos. También incluye máscaras indicadoras.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("El índice del DataFrame debe ser un DatetimeIndex.")

    # 1. Divide los datos
    mask_train = (df.index >= train_start) & (df.index < train_end)
    train_set = df.loc[mask_train, feature_cols]

    mask_val = (df.index >= val_start) & (df.index < val_end)
    val_set = df.loc[mask_val, feature_cols]

    mask_test = (df.index >= test_start) & (df.index < test_end)
    test_set = df.loc[mask_test, feature_cols]

    if train_set.empty or val_set.empty or test_set.empty:
        raise ValueError(
            "Una o más divisiones de datos están vacías. "
            "Verifica los rangos de fechas y los datos de entrada."
        )

    # Guarda los índices originales para remodelar más tarde si es necesario
    train_index = train_set.index
    val_index = val_set.index
    test_index = test_set.index

    # 2. Escala los datos
    scaler = StandardScaler()
    train_X_scaled = scaler.fit_transform(train_set)
    val_X_scaled = scaler.transform(val_set)
    test_X_scaled = scaler.transform(test_set)

    # Convierte de nuevo a DataFrame para preservar el índice para la alineación
    # de ventanas (opcional pero más seguro)
    # train_X_scaled_df = pd.DataFrame(train_X_scaled, index=train_index,
    #                                  columns=feature_cols)
    # val_X_scaled_df = pd.DataFrame(val_X_scaled, index=val_index,
    #                                  columns=feature_cols)
    # test_X_scaled_df = pd.DataFrame(test_X_scaled, index=test_index,
    #                                   columns=feature_cols)
    # Nota: Usar sliding_window directamente en arrays numpy es más estándar

    # 3. Aplica ventana deslizante
    # Importante: La ventana deslizante reduce el número de muestras.
    # Las ventanas resultantes corresponden a secuencias que terminan en el tiempo
    # t, t+1, ...
    # Mantén el seguimiento del índice correspondiente al *inicio* de cada
    # ventana si es necesario.
    train_X_win = sliding_window(train_X_scaled, n_steps)
    val_X_win = sliding_window(val_X_scaled, n_steps)
    test_X_win = sliding_window(test_X_scaled, n_steps)

    # Ajusta los índices originales para que coincidan con los inicios de ventana
    # (si es necesario aguas abajo)
    # train_win_index = train_index[n_steps-1:]
    # val_win_index = val_index[n_steps-1:]
    # test_win_index = test_index[n_steps-1:]

    n_features = train_X_win.shape[-1]

    processed_data = {
        "n_steps": n_steps,
        "n_features": n_features,
        "scaler": scaler,
        "train_index": train_index,  # Índice antes del ventaneo
        "val_index": val_index,  # Índice antes del ventaneo
        "test_index": test_index,  # Índice antes del ventaneo
        # "train_win_index": train_win_index, # Índice de los inicios de ventana
        # "val_win_index": val_win_index,
        # "test_win_index": test_win_index,
        # Guarda los datos escalados+ventaneados originales antes de enmascarar
        "train_X_ori": train_X_win.copy(),
        "val_X_ori": val_X_win.copy(),
        "test_X_ori": test_X_win.copy(),
        # Estos serán sobreescritos si se añaden datos faltantes
        "train_X": train_X_win,
        "val_X": val_X_win,
        "test_X": test_X_win,
    }

    # 4. Introduce datos faltantes (Opcional)
    if missing_rate > 0:
        print(
            f"Introduciendo {missing_rate*100:.1f}% de datos faltantes "
            f"con patrón '{missing_pattern}'..."
        )
        # Crea máscaras basadas en los datos ventaneados *originales* antes de
        # introducir NaNs
        processed_data["train_missing_mask"] = np.isnan(processed_data["train_X_ori"])
        processed_data["val_missing_mask"] = np.isnan(processed_data["val_X_ori"])
        processed_data["test_missing_mask"] = np.isnan(processed_data["test_X_ori"])

        # Aplica datos faltantes
        train_X_missing = create_missingness(
            processed_data["train_X_ori"], missing_rate, missing_pattern, **missingness_kwargs
        )
        val_X_missing = create_missingness(
            processed_data["val_X_ori"], missing_rate, missing_pattern, **missingness_kwargs
        )
        test_X_missing = create_missingness(
            processed_data["test_X_ori"], missing_rate, missing_pattern, **missingness_kwargs
        )

        processed_data["train_X"] = train_X_missing
        processed_data["val_X"] = val_X_missing
        processed_data["test_X"] = test_X_missing

        # Crea máscaras indicadoras (1 donde el valor falta, 0 en otro caso)
        # *después* de la introducción
        processed_data["train_indicating_mask"] = np.isnan(train_X_missing).astype(int)
        processed_data["val_indicating_mask"] = np.isnan(val_X_missing).astype(int)
        processed_data["test_indicating_mask"] = np.isnan(test_X_missing).astype(int)

    else:
        print("La tasa de datos faltantes es 0. No se introducen datos faltantes artificiales.")
        # Si no se añaden datos faltantes, las máscaras reflejan los NaNs
        # originales (si los hay después de escalar/ventanear)
        processed_data["train_missing_mask"] = np.isnan(processed_data["train_X"])
        processed_data["val_missing_mask"] = np.isnan(processed_data["val_X"])
        processed_data["test_missing_mask"] = np.isnan(processed_data["test_X"])
        processed_data["train_indicating_mask"] = processed_data[
            "train_missing_mask"
        ].astype(int)
        processed_data["val_indicating_mask"] = processed_data["val_missing_mask"].astype(
            int
        )
        processed_data["test_indicating_mask"] = processed_data[
            "test_missing_mask"
        ].astype(int)

    # Añade el número de muestras
    processed_data["n_train_samples"] = processed_data["train_X"].shape[0]
    processed_data["n_val_samples"] = processed_data["val_X"].shape[0]
    processed_data["n_test_samples"] = processed_data["test_X"].shape[0]

    return processed_data