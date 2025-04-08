# pampaneira_imputation/config.py
import pandas as pd

# --- Rutas de archivos ---
DATA_DIR = "../data"
RESULTS_DIR = "../results"
TRAFFIC_FILE = f"{DATA_DIR}/trafico_feb22_ago23.csv"
INTERSECTION_FILE = f"{DATA_DIR}/trafico_contamina_intersección.csv"
SAITS_MODEL_SAVE_PATH = f"{RESULTS_DIR}/imputation/saits"
TRANSFORMER_MODEL_SAVE_PATH = f"{RESULTS_DIR}/imputation/transformer"

# --- Nombres de columnas ---
PAM_BUB_TRAFFIC_COLS = [
    'vehicles_PAM_1_OUT', 'vehicles_PAM_1_OUT_Zona_Granada', 'vehicles_PAM_1_OUT_Zona_Catalunia_y_Otras',
    'vehicles_PAM_1_OUT_Zona_Andalucia_no_GR', 'vehicles_PAM_1_OUT_Extranjero', 'vehicles_PAM_1_OUT_Zona_Comunidad_de_Madrid',
    'vehicles_PAM_1_OUT_Zona_Extremadura_y_Otras', 'vehicles_PAM_1_OUT_Zona_Otras', 'vehicles_PAM_1_OUT_5_seats',
    'vehicles_PAM_1_OUT_+5_seats', 'vehicles_PAM_1_OUT_-5_seats', 'vehicles_PAM_1_OUT_nan_seats',
    'vehicles_PAM_1_OUT_101-200_CO2', 'vehicles_PAM_1_OUT_0-100_CO2', 'vehicles_PAM_1_OUT_nan_CO2',
    'vehicles_PAM_1_OUT_201-300_CO2', 'vehicles_PAM_1_OUT_+300_CO2', 'vehicles_PAM_1_IN',
    'vehicles_PAM_1_IN_Zona_Granada', 'vehicles_PAM_1_IN_Zona_Catalunia_y_Otras', 'vehicles_PAM_1_IN_Zona_Andalucia_no_GR',
    'vehicles_PAM_1_IN_Extranjero', 'vehicles_PAM_1_IN_Zona_Comunidad_de_Madrid', 'vehicles_PAM_1_IN_Zona_Extremadura_y_Otras',
    'vehicles_PAM_1_IN_Zona_Otras', 'vehicles_PAM_1_IN_5_seats', 'vehicles_PAM_1_IN_+5_seats',
    'vehicles_PAM_1_IN_-5_seats', 'vehicles_PAM_1_IN_nan_seats', 'vehicles_PAM_1_IN_101-200_CO2',
    'vehicles_PAM_1_IN_0-100_CO2', 'vehicles_PAM_1_IN_nan_CO2', 'vehicles_PAM_1_IN_201-300_CO2',
    'vehicles_PAM_1_IN_+300_CO2', 'vehicles_PAM_2_OUT', 'vehicles_PAM_2_OUT_Zona_Granada',
    'vehicles_PAM_2_OUT_Zona_Catalunia_y_Otras', 'vehicles_PAM_2_OUT_Zona_Andalucia_no_GR', 'vehicles_PAM_2_OUT_Extranjero',
    'vehicles_PAM_2_OUT_Zona_Comunidad_de_Madrid', 'vehicles_PAM_2_OUT_Zona_Extremadura_y_Otras',
    'vehicles_PAM_2_OUT_Zona_Otras', 'vehicles_PAM_2_OUT_5_seats', 'vehicles_PAM_2_OUT_+5_seats',
    'vehicles_PAM_2_OUT_-5_seats', 'vehicles_PAM_2_OUT_nan_seats', 'vehicles_PAM_2_OUT_101-200_CO2',
    'vehicles_PAM_2_OUT_0-100_CO2', 'vehicles_PAM_2_OUT_nan_CO2', 'vehicles_PAM_2_OUT_201-300_CO2',
    'vehicles_PAM_2_OUT_+300_CO2', 'vehicles_PAM_2_IN', 'vehicles_PAM_2_IN_Zona_Granada',
    'vehicles_PAM_2_IN_Zona_Catalunia_y_Otras', 'vehicles_PAM_2_IN_Zona_Andalucia_no_GR', 'vehicles_PAM_2_IN_Extranjero',
    'vehicles_PAM_2_IN_Zona_Comunidad_de_Madrid', 'vehicles_PAM_2_IN_Zona_Extremadura_y_Otras',
    'vehicles_PAM_2_IN_Zona_Otras', 'vehicles_PAM_2_IN_5_seats', 'vehicles_PAM_2_IN_+5_seats',
    'vehicles_PAM_2_IN_-5_seats', 'vehicles_PAM_2_IN_nan_seats', 'vehicles_PAM_2_IN_101-200_CO2',
    'vehicles_PAM_2_IN_0-100_CO2', 'vehicles_PAM_2_IN_nan_CO2', 'vehicles_PAM_2_IN_201-300_CO2',
    'vehicles_PAM_2_IN_+300_CO2', 'vehicles_BUB_OUT', 'vehicles_BUB_OUT_Zona_Granada',
    'vehicles_BUB_OUT_Zona_Catalunia_y_Otras', 'vehicles_BUB_OUT_Zona_Andalucia_no_GR', 'vehicles_BUB_OUT_Extranjero',
    'vehicles_BUB_OUT_Zona_Comunidad_de_Madrid', 'vehicles_BUB_OUT_Zona_Extremadura_y_Otras',
    'vehicles_BUB_OUT_Zona_Otras', 'vehicles_BUB_OUT_5_seats', 'vehicles_BUB_OUT_+5_seats',
    'vehicles_BUB_OUT_-5_seats', 'vehicles_BUB_OUT_nan_seats', 'vehicles_BUB_OUT_101-200_CO2',
    'vehicles_BUB_OUT_0-100_CO2', 'vehicles_BUB_OUT_nan_CO2', 'vehicles_BUB_OUT_201-300_CO2',
    'vehicles_BUB_OUT_+300_CO2', 'vehicles_BUB_IN', 'vehicles_BUB_IN_Zona_Granada',
    'vehicles_BUB_IN_Zona_Catalunia_y_Otras', 'vehicles_BUB_IN_Zona_Andalucia_no_GR', 'vehicles_BUB_IN_Extranjero',
    'vehicles_BUB_IN_Zona_Comunidad_de_Madrid', 'vehicles_BUB_IN_Zona_Extremadura_y_Otras',
    'vehicles_BUB_IN_Zona_Otras', 'vehicles_BUB_IN_5_seats', 'vehicles_BUB_IN_+5_seats',
    'vehicles_BUB_IN_-5_seats', 'vehicles_BUB_IN_nan_seats', 'vehicles_BUB_IN_101-200_CO2',
    'vehicles_BUB_IN_0-100_CO2', 'vehicles_BUB_IN_nan_CO2', 'vehicles_BUB_IN_201-300_CO2',
    'vehicles_BUB_IN_+300_CO2'
]

INTERSECTION_POLLUTION_COLS = [
    "CO", "NO2", "NO", "O3", "PM10", "eBC_ff", "eBC_bb",
    "TEMP", "RH", "WS", "WD", "PRES"
]

# Combina columnas de tráfico y contaminación para el conjunto de características final
FEATURE_COLUMNS = PAM_BUB_TRAFFIC_COLS + INTERSECTION_POLLUTION_COLS

# Columnas potencialmente eliminadas para algunos métodos base o análisis
COLS_TO_DROP_FOR_BASELINE = ["WS", "WD"]

# --- Rangos de fechas y periodos ---
DATE_COL = "date"
TRUCK_POS_COL = "truck_pos"
TARGET_TRUCK_POS = "PAM_2"
TIMEZONE = "UTC"

# Fechas del Periodo 1 para los datos de camiones
PERIOD_1_START = pd.to_datetime("2023-01-17 17:00:00+00:00", utc=True)
PERIOD_1_END = pd.to_datetime("2023-03-14 11:00:00+00:00", utc=True)
PERIOD_1_PADDING_START = pd.to_datetime("2023-01-17 00:00:00", utc=True)
PERIOD_1_PADDING_END = pd.to_datetime("2023-03-14 23:00:00", utc=True)


# Fechas del Periodo 2 para los datos de camiones
PERIOD_2_START = pd.to_datetime("2023-06-06 13:00:00+00:00", utc=True)
PERIOD_2_END = pd.to_datetime("2023-06-27 00:00:00+00:00", utc=True)
PERIOD_2_PADDING_START = pd.to_datetime("2023-06-06 00:00:00", utc=True)
PERIOD_2_PADDING_END = pd.to_datetime("2023-06-27 23:00:00", utc=True)


# Fechas de división Entrenamiento/Validación/Prueba para el Periodo 1
TRAIN_START_DATE = "2023-01-17"
TRAIN_END_DATE = "2023-02-22"  # Exclusivo
VAL_START_DATE = "2023-02-22"
VAL_END_DATE = "2023-03-03"    # Exclusivo
TEST_START_DATE = "2023-03-03"
TEST_END_DATE = "2023-03-15"    # Exclusivo (cubre hasta 2023-03-14 23:00)

# Fechas de división Entrenamiento/Validación/Prueba para el Periodo 2
TRAIN_2_START_DATE = "2023-06-06"
TRAIN_2_END_DATE = "2023-06-20"  # Exclusivo
VAL_2_START_DATE = "2023-06-20"
VAL_2_END_DATE = "2023-06-24"    # Exclusivo
TEST_2_START_DATE = "2023-06-24"
TEST_2_END_DATE = "2023-06-28"    # Exclusivo (cubre hasta 2023-06-27 23:00)


# --- Parámetros de preprocesamiento ---
N_STEPS = 24  # Tamaño de la ventana deslizante
MISSING_RATE = 0.1
MISSING_PATTERN = "point"  # o "subseq", "block"

# --- Parámetros del modelo SAITS ---
SAITS_PARAMS = {
    "n_steps": N_STEPS,
    # n_features se establecerá dinámicamente
    "n_layers": 3,
    "d_model": 128,
    "d_ffn": 256,
    "n_heads": 4,
    "d_k": 32,
    "d_v": 32,
    "dropout": 0.3,
    "attn_dropout": 0.2,
    "diagonal_attention_mask": True,
    "ORT_weight": 1,
    "MIT_weight": 1,
    "batch_size": 64,
    "epochs": 10,  # Considera reducir para pruebas/depuración más rápidas
    "patience": 5,
    "num_workers": 0,
    "device": None,  # Autodetecta (CPU o GPU si disponible)
    "saving_path": SAITS_MODEL_SAVE_PATH,
    "model_saving_strategy": "best",
}

# --- Parámetros de entrenamiento SAITS ---
SAITS_OPTIMIZER_PARAMS = {"lr": 5e-5, "weight_decay": 1e-4}
SAITS_SCHEDULER_PARAMS = {"mode": "min", "factor": 0.5, "patience": 5}
SAITS_LOSS_PARAMS = {"beta": 0.1}  # Para SmoothL1Loss


from pypots.nn.modules.loss import MAE, MSE
from pypots.optim import Adam

TRANSFORMER_PARAMS = {
    'n_steps': 24,               # Number of time steps in the time series
    'n_features': None,          # Will be set dynamically
    'n_layers': 3,               # Number of transformer layers
    'd_model': 128,              # Dimension of model
    'n_heads': 4,                # Number of attention heads
    'd_k': 32,                   # Dimension of key
    'd_v': 32,                   # Dimension of value
    'd_ffn': 256,                # Dimension of feed-forward network
    'dropout': 0.3,              # Dropout rate
    'attn_dropout': 0.2,         # Attention dropout rate
    'ORT_weight': 1.0,           # Weight for ORT (Observed Reconstruction Term)
    'MIT_weight': 1.0,           # Weight for MIT (Missing Imputation Term)
    'batch_size': 64,            # Batch size for training
    'epochs': 10,               # Maximum epochs for training
    'patience': 5,              # Early stopping patience
    'num_workers': 0,            # Number of workers for data loading
    'device': None,              # Device to use (None for auto-detection)
    'saving_path': TRANSFORMER_MODEL_SAVE_PATH,   # Path to save the model
    'model_saving_strategy': 'best', # Model saving strategy
    'verbose': True              # Verbose output
}

# --- Parámetros de entrenamiento TRANSFORMER igualados a SAITS ---
TRANSFORMER_OPTIMIZER_PARAMS = {
    'lr': 5e-5,
    'weight_decay': 1e-4
}

TRANSFORMER_SCHEDULER_PARAMS = {
    'mode': 'min',
    'factor': 0.5,
    'patience': 5,
    'verbose': True
}

TRANSFORMER_LOSS_PARAMS = {
    'beta': 0.1  # Igual que SAITS
}