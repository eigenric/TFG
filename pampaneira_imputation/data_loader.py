# pampaneira_imputation/data_loader.py
import pandas as pd
from typing import List
from . import config

pd.options.mode.chained_assignment = None

def load_traffic_data(filepath: str = config.TRAFFIC_FILE,
                      columns_to_use: List[str] = config.PAM_BUB_TRAFFIC_COLS,
                      date_col: str = config.DATE_COL,
                      timezone: str = config.TIMEZONE) -> pd.DataFrame:
    """
    Carga datos de tráfico generales, selecciona columnas relevantes, convierte la fecha.

    Args:
        filepath (str, optional): Ruta al archivo CSV de tráfico (por defecto: config.TRAFFIC_FILE).
        columns_to_use (List[str], optional): Lista de columnas de tráfico a usar (por defecto: config.PAM_BUB_TRAFFIC_COLS).
        date_col (str, optional): Nombre de la columna de fecha (por defecto: config.DATE_COL).
        timezone (str, optional): Zona horaria para las fechas (por defecto: config.TIMEZONE).

    Returns:
        pd.DataFrame: DataFrame con datos de tráfico cargados y preprocesados.

    Raises:
        FileNotFoundError: Si el archivo especificado no se encuentra.
        KeyError: Si una columna especificada no se encuentra en el archivo.
    """
    try:
        df = pd.read_csv(filepath)
        df[date_col] = pd.to_datetime(df[date_col])
        # Asegura que la zona horaria UTC sea consciente si aún no lo es
        if df[date_col].dt.tz is None:
             df[date_col] = df[date_col].dt.tz_localize(timezone)
        elif df[date_col].dt.tz.zone != timezone:
             df[date_col] = df[date_col].dt.tz_convert(timezone)

        # Selecciona solo las columnas requeridas más la columna de fecha
        df_filtered = df[[date_col] + columns_to_use]

        # Convierte columnas enteras a float64 como en el script original
        int_cols = df_filtered.select_dtypes(include='int64').columns
        df_filtered[int_cols] = df_filtered[int_cols].astype('float64')

        return df_filtered

    except FileNotFoundError:
        print(f"Error: No se encontró el archivo en {filepath}")
        raise
    except KeyError as e:
        print(f"Error: No se encontró la columna {e} en {filepath}.")
        raise


def load_intersection_data(filepath: str = config.INTERSECTION_FILE,
                           date_col_original: str = "Date",  # Nombre original en CSV
                           date_col_target: str = config.DATE_COL,
                           truck_pos_col: str = config.TRUCK_POS_COL,
                           target_truck_pos: str = config.TARGET_TRUCK_POS,
                           timezone: str = config.TIMEZONE) -> pd.DataFrame:
    """
    Carga datos de intersección, filtra por posición de camión, convierte la fecha.

    Args:
        filepath (str, optional): Ruta al archivo CSV de intersección (por defecto: config.INTERSECTION_FILE).
        date_col_original (str, optional): Nombre original de la columna de fecha en el CSV (por defecto: "Date").
        date_col_target (str, optional): Nombre objetivo de la columna de fecha (por defecto: config.DATE_COL).
        truck_pos_col (str, optional): Nombre de la columna de posición del camión (por defecto: config.TRUCK_POS_COL).
        target_truck_pos (str, optional): Posición objetivo del camión para filtrar (por defecto: config.TARGET_TRUCK_POS).
        timezone (str, optional): Zona horaria para las fechas (por defecto: config.TIMEZONE).

    Returns:
        pd.DataFrame: DataFrame con datos de intersección cargados, filtrados y preprocesados.

    Raises:
        FileNotFoundError: Si el archivo especificado no se encuentra.
        KeyError: Si una columna especificada no se encuentra o falla el cambio de nombre en el archivo.
    """
    try:
        df = pd.read_csv(filepath)
        df.rename(columns={date_col_original: date_col_target}, inplace=True)
        df[date_col_target] = pd.to_datetime(df[date_col_target])

        # Asegura que la zona horaria UTC sea consciente si aún no lo es
        if df[date_col_target].dt.tz is None:
             df[date_col_target] = df[date_col_target].dt.tz_localize(timezone)
        elif df[date_col_target].dt.tz.zone != timezone:
             df[date_col_target] = df[date_col_target].dt.tz_convert(timezone)

        # Filtra por posición de camión
        df_filtered = df[df[truck_pos_col] == target_truck_pos].copy()  # Usa .copy()

        # Convierte columnas enteras a float64
        int_cols = df_filtered.select_dtypes(include='int64').columns
        df_filtered[int_cols] = df_filtered[int_cols].astype('float64')

        # Selecciona solo las columnas de características finales + fecha (definidas en config)
        # Esto asume que el archivo de intersección contiene todas las FEATURE_COLUMNS
        # Si no, ajusta config.FEATURE_COLUMNS o esta lógica de selección
        cols_to_keep = [date_col_target] + config.FEATURE_COLUMNS
        # Asegura que solo mantenemos las columnas presentes en el dataframe
        cols_present = [col for col in cols_to_keep if col in df_filtered.columns]
        df_final = df_filtered[cols_present]


        return df_final

    except FileNotFoundError:
        print(f"Error: No se encontró el archivo en {filepath}")
        raise
    except KeyError as e:
        print(f"Error: No se encontró la columna {e} o falló el renombrado en {filepath}.")
        raise