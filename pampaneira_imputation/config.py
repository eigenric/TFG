# pampaneira_imputation/config.py
import pandas as pd

# --- File Paths ---
DATA_DIR = "../data"
RESULTS_DIR = "../results"
TRAFFIC_FILE = f"{DATA_DIR}/trafico_feb22_ago23.csv"
INTERSECTION_FILE = f"{DATA_DIR}/trafico_contamina_intersección.csv"
SAITS_MODEL_SAVE_PATH = f"{RESULTS_DIR}/imputation/saits"

# --- Column Names ---
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

# Combine traffic and pollution columns for the final feature set
FEATURE_COLUMNS = PAM_BUB_TRAFFIC_COLS + INTERSECTION_POLLUTION_COLS

# Columns potentially dropped for some baseline methods or analyses
COLS_TO_DROP_FOR_BASELINE = ["WS", "WD"]

# --- Date Ranges & Periods ---
DATE_COL = "date"
TRUCK_POS_COL = "truck_pos"
TARGET_TRUCK_POS = "PAM_2"
TIMEZONE = "UTC"

# Period 1 dates for the truck data
PERIOD_1_START = pd.to_datetime("2023-01-17 17:00:00+00:00", utc=True)
PERIOD_1_END = pd.to_datetime("2023-03-14 11:00:00+00:00", utc=True)
PERIOD_1_PADDING_START = pd.to_datetime("2023-01-17 00:00:00", utc=True)
PERIOD_1_PADDING_END = pd.to_datetime("2023-03-14 23:00:00", utc=True)


# Period 2 dates for the truck data
PERIOD_2_START = pd.to_datetime("2023-06-06 13:00:00+00:00", utc=True)
PERIOD_2_END = pd.to_datetime("2023-06-27 00:00:00+00:00", utc=True)

# Train/Val/Test Split Dates for Period 1
TRAIN_START_DATE = "2023-01-17"
TRAIN_END_DATE = "2023-02-22" # Exclusive
VAL_START_DATE = "2023-02-22"
VAL_END_DATE = "2023-03-03"   # Exclusive
TEST_START_DATE = "2023-03-03"
TEST_END_DATE = "2023-03-15"   # Exclusive (covers up to 2023-03-14 23:00)

# --- Preprocessing Parameters ---
N_STEPS = 24 # Sliding window size
MISSING_RATE = 0.1
MISSING_PATTERN = "point" # or "subseq", "block"

# --- SAITS Model Parameters ---
SAITS_PARAMS = {
    "n_steps": N_STEPS,
    # n_features will be set dynamically
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
    "epochs": 200, # Consider reducing for faster testing/debugging
    "patience": 20,
    "num_workers": 0,
    "device": None, # Autodetect (CPU or GPU if available)
    "saving_path": SAITS_MODEL_SAVE_PATH,
    "model_saving_strategy": "best",
}

# --- SAITS Training Parameters ---
SAITS_OPTIMIZER_PARAMS = {"lr": 5e-5, "weight_decay": 1e-4}
SAITS_SCHEDULER_PARAMS = {"mode": "min", "factor": 0.5, "patience": 5}
SAITS_LOSS_PARAMS = {"beta": 0.1} # For SmoothL1Loss