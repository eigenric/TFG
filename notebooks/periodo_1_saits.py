import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from transformers import pipeline
from pypots.optim import Adam
from pypots.imputation import SAITS


pd.options.mode.chained_assignment = None

trafico_feb22_ago23 = pd.read_csv('../data/trafico_feb22_ago23.csv')

# PAM_1 y BUB
columns = ['vehicles_PAM_1_OUT',
 'vehicles_PAM_1_OUT_Zona_Granada',
 'vehicles_PAM_1_OUT_Zona_Catalunia_y_Otras',
 'vehicles_PAM_1_OUT_Zona_Andalucia_no_GR',
 'vehicles_PAM_1_OUT_Extranjero',
 'vehicles_PAM_1_OUT_Zona_Comunidad_de_Madrid',
 'vehicles_PAM_1_OUT_Zona_Extremadura_y_Otras',
 'vehicles_PAM_1_OUT_Zona_Otras',
 'vehicles_PAM_1_OUT_5_seats',
 'vehicles_PAM_1_OUT_+5_seats',
 'vehicles_PAM_1_OUT_-5_seats',
 'vehicles_PAM_1_OUT_nan_seats',
 'vehicles_PAM_1_OUT_101-200_CO2',
 'vehicles_PAM_1_OUT_0-100_CO2',
 'vehicles_PAM_1_OUT_nan_CO2',
 'vehicles_PAM_1_OUT_201-300_CO2',
 'vehicles_PAM_1_OUT_+300_CO2',
 'vehicles_PAM_1_IN',
 'vehicles_PAM_1_IN_Zona_Granada',
 'vehicles_PAM_1_IN_Zona_Catalunia_y_Otras',
 'vehicles_PAM_1_IN_Zona_Andalucia_no_GR',
 'vehicles_PAM_1_IN_Extranjero',
 'vehicles_PAM_1_IN_Zona_Comunidad_de_Madrid',
 'vehicles_PAM_1_IN_Zona_Extremadura_y_Otras',
 'vehicles_PAM_1_IN_Zona_Otras',
 'vehicles_PAM_1_IN_5_seats',
 'vehicles_PAM_1_IN_+5_seats',
 'vehicles_PAM_1_IN_-5_seats',
 'vehicles_PAM_1_IN_nan_seats',
 'vehicles_PAM_1_IN_101-200_CO2',
 'vehicles_PAM_1_IN_0-100_CO2',
 'vehicles_PAM_1_IN_nan_CO2',
 'vehicles_PAM_1_IN_201-300_CO2',
 'vehicles_PAM_1_IN_+300_CO2',
 'vehicles_PAM_2_OUT',
 'vehicles_PAM_2_OUT_Zona_Granada',
 'vehicles_PAM_2_OUT_Zona_Catalunia_y_Otras',
 'vehicles_PAM_2_OUT_Zona_Andalucia_no_GR',
 'vehicles_PAM_2_OUT_Extranjero',
 'vehicles_PAM_2_OUT_Zona_Comunidad_de_Madrid',
 'vehicles_PAM_2_OUT_Zona_Extremadura_y_Otras',
 'vehicles_PAM_2_OUT_Zona_Otras',
 'vehicles_PAM_2_OUT_5_seats',
 'vehicles_PAM_2_OUT_+5_seats',
 'vehicles_PAM_2_OUT_-5_seats',
 'vehicles_PAM_2_OUT_nan_seats',
 'vehicles_PAM_2_OUT_101-200_CO2',
 'vehicles_PAM_2_OUT_0-100_CO2',
 'vehicles_PAM_2_OUT_nan_CO2',
 'vehicles_PAM_2_OUT_201-300_CO2',
 'vehicles_PAM_2_OUT_+300_CO2',
 'vehicles_PAM_2_IN',
 'vehicles_PAM_2_IN_Zona_Granada',
 'vehicles_PAM_2_IN_Zona_Catalunia_y_Otras',
 'vehicles_PAM_2_IN_Zona_Andalucia_no_GR',
 'vehicles_PAM_2_IN_Extranjero',
 'vehicles_PAM_2_IN_Zona_Comunidad_de_Madrid',
 'vehicles_PAM_2_IN_Zona_Extremadura_y_Otras',
 'vehicles_PAM_2_IN_Zona_Otras',
 'vehicles_PAM_2_IN_5_seats',
 'vehicles_PAM_2_IN_+5_seats',
 'vehicles_PAM_2_IN_-5_seats',
 'vehicles_PAM_2_IN_nan_seats',
 'vehicles_PAM_2_IN_101-200_CO2',
 'vehicles_PAM_2_IN_0-100_CO2',
 'vehicles_PAM_2_IN_nan_CO2',
 'vehicles_PAM_2_IN_201-300_CO2',
 'vehicles_PAM_2_IN_+300_CO2',
 'vehicles_BUB_OUT',
 'vehicles_BUB_OUT_Zona_Granada',
 'vehicles_BUB_OUT_Zona_Catalunia_y_Otras',
 'vehicles_BUB_OUT_Zona_Andalucia_no_GR',
 'vehicles_BUB_OUT_Extranjero',
 'vehicles_BUB_OUT_Zona_Comunidad_de_Madrid',
 'vehicles_BUB_OUT_Zona_Extremadura_y_Otras',
 'vehicles_BUB_OUT_Zona_Otras',
 'vehicles_BUB_OUT_5_seats',
 'vehicles_BUB_OUT_+5_seats',
 'vehicles_BUB_OUT_-5_seats',
 'vehicles_BUB_OUT_nan_seats',
 'vehicles_BUB_OUT_101-200_CO2',
 'vehicles_BUB_OUT_0-100_CO2',
 'vehicles_BUB_OUT_nan_CO2',
 'vehicles_BUB_OUT_201-300_CO2',
 'vehicles_BUB_OUT_+300_CO2',
 'vehicles_BUB_IN',
 'vehicles_BUB_IN_Zona_Granada',
 'vehicles_BUB_IN_Zona_Catalunia_y_Otras',
 'vehicles_BUB_IN_Zona_Andalucia_no_GR',
 'vehicles_BUB_IN_Extranjero',
 'vehicles_BUB_IN_Zona_Comunidad_de_Madrid',
 'vehicles_BUB_IN_Zona_Extremadura_y_Otras',
 'vehicles_BUB_IN_Zona_Otras',
 'vehicles_BUB_IN_5_seats',
 'vehicles_BUB_IN_+5_seats',
 'vehicles_BUB_IN_-5_seats',
 'vehicles_BUB_IN_nan_seats',
 'vehicles_BUB_IN_101-200_CO2',
 'vehicles_BUB_IN_0-100_CO2',
 'vehicles_BUB_IN_nan_CO2',
 'vehicles_BUB_IN_201-300_CO2',
 'vehicles_BUB_IN_+300_CO2']

trafico_feb22_ago23["date"] = pd.to_datetime(trafico_feb22_ago23["date"])
trafico_feb22_ago23["date"] = trafico_feb22_ago23["date"].dt.tz_convert('UTC')

df_orig = trafico_feb22_ago23[["date"] + columns]
columnas_int64 = df_orig.select_dtypes(include='int64').columns

# Convertir esas columnas a float64
df_orig[columnas_int64] = df_orig[columnas_int64].astype('float64')

trafico_contamina_interseccion = pd.read_csv('../data/trafico_contamina_intersección.csv')
trafico_contamina_interseccion["Date"] = pd.to_datetime(trafico_contamina_interseccion["Date"])
trafico_contamina_interseccion["Date"] = trafico_contamina_interseccion["Date"].dt.tz_localize("UTC")
trafico_contamina_interseccion.rename(columns={'Date': 'date'}, inplace=True)

columns_interseccion = ["CO", "NO2", "NO", "O3", "SO2", "PM10", "eBC_tot", "eBC_ff", "eBC_bb", "TEMP", "RH", "WS", "WD", "PRES", "eBC"]

trafico_pam = trafico_contamina_interseccion[trafico_contamina_interseccion["truck_pos"] == "PAM_2"]

df_int = trafico_pam[["date"] + columns_interseccion]
columnas_int64 = df_int.select_dtypes(include='int64').columns

# Convertir esas columnas a float64
df_int[columnas_int64] = df_int[columnas_int64].astype('float64')


# Supongamos que el DataFrame inicial es df_int con columnas 'date'

# Definir los dos períodos
start_date_1 = pd.to_datetime("2023-01-17 17:00:00+00:00")
end_date_1 = pd.to_datetime("2023-03-14 11:00:00+00:00")

start_date_2 = pd.to_datetime("2023-06-06 13:00:00+00:00")
end_date_2 = pd.to_datetime("2023-06-27 00:00:00+00:00")

# Crear un rango completo de fechas con frecuencia horaria para ambos períodos combinados
full_date_range = pd.date_range(start=start_date_1, end=end_date_1, freq='h').union(
    pd.date_range(start=start_date_2, end=end_date_2, freq='h')
)

# Asegurar que 'date' sea de tipo datetime
df_int['date'] = pd.to_datetime(df_int['date'])

# Reindexar para completar las horas faltantes
df_int = df_int.set_index('date').reindex(full_date_range).reset_index()
df_int = df_int.rename(columns={'index': 'date'})

df_int["date"] = pd.to_datetime(df_int["date"])
df_int.set_index("date", inplace=True)

# Definir los periodos
periodo_1 = df_int.loc["2023-01-17":"2023-03-14"]
periodo_2 = df_int.loc["2023-06-06":"2023-06-27"]

# initialize the model
saits = SAITS(
    n_steps=len(periodo_1),
    n_features=len(periodo_1.columns)-1,
    n_layers=2,
    d_model=256,
    d_ffn=128,
    n_heads=4,
    d_k=64,
    d_v=64,
    dropout=0.1,
    attn_dropout=0.1,
    ##### Esto diferencia al SAIT del modelo Transformer
    diagonal_attention_mask=True,  # otherwise the original self-attention mechanism will be applied
    ORT_weight=1,  # you can adjust the weight values of arguments ORT_weight
    # and MIT_weight to make the SAITS model focus more on one task. Usually you can just leave them to the default values, i.e. 1.
    MIT_weight=1,
    batch_size=32,
    # here we set epochs=10 for a quick demo, you can set it to 100 or more for better performance
    epochs=10,
    # here we set patience=3 to early stop the training if the evaluting loss doesn't decrease for 3 epoches.
    # You can leave it to defualt as None to disable early stopping.
    patience=3,
    # give the optimizer. Different from torch.optim.Optimizer, you don't have to specify model's parameters when
    # initializing pypots.optim.Optimizer. You can also leave it to default. It will initilize an Adam optimizer with lr=0.001.
    optimizer=Adam(lr=1e-3),
    # this num_workers argument is for torch.utils.data.Dataloader. It's the number of subprocesses to use for data loading.
    # Leaving it to default as 0 means data loading will be in the main process, i.e. there won't be subprocesses.
    # You can increase it to >1 if you think your dataloading is a bottleneck to your model training speed
    num_workers=0,
    # just leave it to default as None, PyPOTS will automatically assign the best device for you.
    # Set it as 'cpu' if you don't have CUDA devices. You can also set it to 'cuda:0' or 'cuda:1' if you have multiple CUDA devices, even parallelly on ['cuda:0', 'cuda:1']
    device=None,  
    # set the path for saving tensorboard and trained model files 
    saving_path="tutorial_results/imputation/saits",
    # only save the best model after training finished.
    # You can also set it as "better" to save models performing better ever during training.
    model_saving_strategy="best",
)


X = periodo_1.drop(columns=["CO"]).values  # Convertir a numpy array
X = X.astype(np.float32)  # 
X = np.expand_dims(X, axis=0)  # Esto agrega la dimensión de la secuencia
y = periodo_1["CO"].values
y = y.astype(np.float32)  # 

# Crear las variables 'X_ori' como copia de 'X'
X_ori = X.copy()

# Crear el diccionario de datos con 'X', 'X_ori' y 'y'
data = {"X": X, "y": y, "X_ori": X_ori}

# Entrenar el modelo
saits.fit(train_set=data, val_set=data)