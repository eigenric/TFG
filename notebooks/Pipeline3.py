#!/usr/bin/env python
# coding: utf-8

# # Región Pampaneira
# 
# 

# 
# Debido al largo contenido del primer Pipeline, que incluye un analisis exploratorio de los
# datos y teniendo en cuenta que hay que restringirse a la región de Pampaneira para realizar el 
# el objetivo principal de generar datos mediante modelos transformer, se ha creado un Notebook
# Pipeline2.
# 
# 

# In[1]:


import pandas as pd

pd.options.mode.chained_assignment = None

pd.reset_option('display.max_rows')  # To show all rows
pd.reset_option('display.max_columns')  # To show all columns


# In[2]:


trafico_feb22_ago23 = pd.read_csv('../data/trafico_feb22_ago23.csv')


# In[3]:


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

df_orig


# In[4]:


trafico_contamina_interseccion = pd.read_csv('../data/trafico_contamina_intersección.csv')
trafico_contamina_interseccion["Date"] = pd.to_datetime(trafico_contamina_interseccion["Date"])
trafico_contamina_interseccion["Date"] = trafico_contamina_interseccion["Date"].dt.tz_localize("UTC")
trafico_contamina_interseccion.rename(columns={'Date': 'date'}, inplace=True)


# In[5]:


columns_interseccion = [
    "CO", 
    "NO2", 
    "NO", 
    "O3", 
    "PM10", 
    "eBC_ff", 
    "eBC_bb", 
    "TEMP", 
    "RH", 
    "WS", 
    "WD", 
    "PRES"
]
# Combinamos ambas listas
columns_interseccion = columns + columns_interseccion

trafico_pam = trafico_contamina_interseccion[trafico_contamina_interseccion["truck_pos"] == "PAM_2"]

df_int = trafico_pam[["date"] + columns_interseccion]
columnas_int64 = df_int.select_dtypes(include='int64').columns

# Convertir esas columnas a float64
df_int[columnas_int64] = df_int[columnas_int64].astype('float64')

df_int


# ## Interpolación para completar los valores nulos.
# 
# 
# 

# ## Rellenar del df_int las horas faltantes con valores NaN
# 
# En los datos del camión, poner un registro por cada hora. 
# Si no se encuentra, añadir todos NaN.
# 
# El método de pandas date_range devuelve el rango de datetimes igualmente espaciados (dada una frecuencia) entre un inicio y un final (por defecto ambos inclusive). El tipo de dato que devuelve es DatetimeIndex.
# Dado que disponemos de dos periodos de fechas con datos disponibles (1er periodo: 17/01/2023-14/03/2023 y 2do periodo 06/06/2023-27/06/2023) realizaremos otro rango para el segundo periodo y lo uniremos mediante el método union perteneciente a la clase Index.
# 
# Explicación paso a paso:
# - `set_index('date')`: 
# Convierte la columna 'date' en el índice del DataFrame df_int.
# - `reindex(full_date_range)`
# Rellena el DataFrame para que tenga todas las fechas en full_date_range, incluso si algunas no estaban en df_int.
# Esto es útil si hay fechas faltantes y quieres asegurar de que el índice sea continuo.
# reset_index()
# 
# Devuelve la columna de fecha como una columna normal en lugar de mantenerla como índice.

# In[6]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

df_int


# ## Periodo_1 

# In[7]:


len(df_int.columns)


# In[8]:


df_int["date"] = pd.to_datetime(df_int["date"])
df_int.set_index("date", inplace=True)


# ## División Por Periodos

# In[9]:


import pandas as pd

# Definir los periodos
periodo_1 = df_int.loc["2023-01-17":"2023-03-14"]
periodo_2 = df_int.loc["2023-06-06":"2023-06-27"]


# ## División en training, validation y test

# Para dividir el periodo del 17 de enero de 2023 al 14 de marzo de 2023 en entrenamiento, validación y prueba siguiendo un esquema análogo al del Beijing Multi-Site Air-Quality Dataset, vamos a asignar los datos de manera similar a cómo se hizo en ese conjunto.
# 
# El esquema original divide los datos en tres partes:
# 
# - **Fase de entrenamiento (80% del dataset):** Primeros 45 días
#    - **Training:** 80% de los 45 días: 36 días
#    - **Validación** 20% restante de los 45 días: 9 días.
# - **Prueba (20% del dataset):** 20% del dataset:  11 días.
# 
# ### Duración total
# El periodo de tiempo es del 17 de enero de 2023 al 14 de marzo de 2023. **hay 57 días.**
# 
# ## Resumen:
# 
# - **Conjunto de entrenamiento: 17 de enero de 2023 - 21 de febrero de 2023 (36 días)**
# - **Conjunto de validación: 22 de febrero de 2023 - 2 de marzo de 2023 (9 días)**.
# - **Conjunto de prueba: 3 de marzo de 2023 - 14 de marzo de 2023) (12 días)**
# 
# 

# In[10]:


periodo_1.loc["2023-03-03":"2023-03-14"]


# In[11]:


# Crear el índice con las fechas y horas de 2023-01-17 00:00:00 hasta 2023-01-17 17:00:00
date_range = pd.date_range(start="2023-01-17 00:00:00", end="2023-01-17 16:00:00", freq="H", tz="UTC")

# Crear el índice con las fechas y horas de 2023-01-17 12:00:00 hasta 2023-01-17 23:00:00
date_range_end = pd.date_range(start="2023-03-14 12:00:00", end="2023-03-14 23:00:00", freq="H", tz="UTC")

# Definir las columnas de calidad del aire
columns = ["vehicles_PAM_1_OUT", "CO", "NO2", "NO", "O3", "PM10", "eBC_ff", "eBC_bb", "TEMP", "RH", "WS", "WD", "PRES"]

# Crear un DataFrame con el índice y columnas, rellenando los valores con NaN
df = pd.DataFrame(np.nan, index=date_range, columns=columns)
df_end = pd.DataFrame(np.nan, index=date_range_end, columns=columns)
periodo_1_df = pd.concat([df, periodo_1, df_end], axis=0)  # axis=0 indica concatenar por filas


# In[12]:


periodo_1_df


# In[13]:


periodo_1_df.loc["2023-03-03":"2023-03-14"]


# In[14]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Dict

def sliding_window(data, n_steps):
    X = []
    for i in range(0, len(data), n_steps):
        # Solo agregamos ventanas completas
        if i + n_steps <= len(data):
            X.append(data[i:i + n_steps])
    
    # Convertimos la lista de ventanas a un array
    return np.array(X)

    
def create_missingness(data: np.ndarray, rate: float, pattern: str, **kwargs) -> np.ndarray:
    """
    Introduce valores faltantes en el conjunto de datos con un patrón específico.
    """
    n_samples, n_features, n_steps = data.shape
    missing_data = data.copy()
    
    if pattern == "point":
        # Introducir valores faltantes de forma aleatoria (punto a punto)
        for i in range(n_samples):
            for j in range(n_features):
                if np.random.rand() < rate:
                    missing_data[i, j, np.random.randint(n_steps)] = np.nan
    elif pattern == "subseq":
        # Introducir valores faltantes en subsecuencias
        for i in range(n_samples):
            for j in range(n_features):
                if np.random.rand() < rate:
                    start_idx = np.random.randint(n_steps)
                    length = np.random.randint(1, n_steps - start_idx)
                    missing_data[i, j, start_idx:start_idx + length] = np.nan
    elif pattern == "block":
        # Introducir valores faltantes en bloques
        for i in range(n_samples):
            for j in range(n_features):
                if np.random.rand() < rate:
                    block_size = np.random.randint(1, n_steps)
                    missing_data[i, j, :block_size] = np.nan
    return missing_data



# In[15]:


# Seleccionar solo las columnas necesarias para las características
features = columns_interseccion


# In[16]:


import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_periodo_1_data(
    periodo_1,
    rate: float,
    n_steps: int,
    pattern: str = "point",
    **kwargs,
) -> dict:
    
    # Asegúrate de que el índice sea de tipo datetime
    if not pd.api.types.is_datetime64_any_dtype(periodo_1.index):
        periodo_1.index = pd.to_datetime(periodo_1.index)

    # Verificar las fechas en el DataFrame
    print("Fechas en periodo_1:", periodo_1.index.min(), "a", periodo_1.index.max())

    # Conjunto de entrenamiento: 17 enero - 21 febrero 2023 (36 días)
    mask_train = (periodo_1.index >= "2023-01-17") & (periodo_1.index < "2023-02-22")
    train_set = periodo_1[mask_train]

    # Verificar si hay datos en train_set
    print("Datos en conjunto de entrenamiento:", train_set.shape)  

    # Conjunto de validación: 22 febrero - 2 marzo 2023 (9 días)
    mask_val = (periodo_1.index >= "2023-02-22") & (periodo_1.index < "2023-03-03")
    val_set = periodo_1[mask_val]

    # Verificar si hay datos en val_set
    print("Datos en conjunto de validación:", val_set.shape)

    # Conjunto de prueba: 3 marzo - 14 marzo 2023 (11 días)
    mask_test = (periodo_1.index >= "2023-03-03") & (periodo_1.index < "2023-03-15")
    test_set = periodo_1[mask_test]

    # Verificar si hay datos en test_set
    print("Datos en conjunto de prueba:", test_set.shape)


    # Verificar si los conjuntos de datos no están vacíos antes de proceder
    if test_set.empty or val_set.empty or train_set.empty:
        raise ValueError("Uno o más conjuntos de datos están vacíos. Revisa las fechas y los filtros.")

    train_set = train_set[features]
    val_set = val_set[features]
    test_set = test_set[features]


    # Normalización de datos
    scaler = StandardScaler()
    train_X = scaler.fit_transform(train_set.loc[:, features])
    val_X = scaler.transform(val_set.loc[:, features])
    test_X = scaler.transform(test_set.loc[:, features])

    # Crear ventanas deslizantes
    train_X = sliding_window(train_X, n_steps)
    val_X = sliding_window(val_X, n_steps)
    test_X = sliding_window(test_X, n_steps)

    # Ensamblar el conjunto de datos final
    processed_dataset = {
        "n_steps": n_steps,
        "n_features": train_X.shape[-1],  # Número de características
        "scaler": scaler,
        "train_X": train_X,
        "val_X": val_X,
        "test_X": test_X,
        "train_X_ori": train_X,  # Versión original de los datos de entrenamiento
        "val_X_ori": val_X,      # Versión original de los datos de validación
        "test_X_ori": test_X,    # Versión original de los datos de prueba
    }

    if rate > 0:
        # hold out ground truth in the original data for evaluation
        train_X_ori = train_X
        val_X_ori = val_X
        test_X_ori = test_X

        # mask values in the train set to keep the same with below validation and test sets
        train_X = create_missingness(train_X, rate, pattern, **kwargs)
        # mask values in the validation set as ground truth
        val_X = create_missingness(val_X, rate, pattern, **kwargs)
        # mask values in the test set as ground truth
        test_X = create_missingness(test_X, rate, pattern, **kwargs)

        # Revertir el escalado (transformación inversa)
        train_X_not_scale = scaler.inverse_transform(train_X.reshape(-1, train_X.shape[-1]))  # Ajuste de forma si es necesario
        val_X_not_scale = scaler.inverse_transform(val_X.reshape(-1, val_X.shape[-1]))  # Ajuste de forma si es necesario
        test_X_not_scale = scaler.inverse_transform(test_X.reshape(-1, test_X.shape[-1]))  # Ajuste de forma si es necesario


        processed_dataset["train_X"] = train_X
        processed_dataset["train_X_not_scale"] = train_X_not_scale
        processed_dataset["train_X_ori"] = train_X_ori


        processed_dataset["val_X"] = val_X
        processed_dataset["val_X_not_scale"] = val_X_not_scale
        processed_dataset["val_X_ori"] = val_X_ori

        processed_dataset["test_X"] = test_X
        processed_dataset["test_X_not_scale"] = test_X_not_scale
        processed_dataset["test_X_ori"] = test_X_ori
    else:
        logger.warning("rate is 0, no missing values are artificially added.")

    return processed_dataset


# In[17]:


periodo_1_df


# In[18]:


print(periodo_1_df.columns)


# In[19]:


# Procesar los datos con una tasa de valores faltantes del 10% y usando el patrón 'point'
periodo_1_dataset = preprocess_periodo_1_data(periodo_1_df, n_steps=24, rate=0.1, pattern='point')

dataset_for_training = {
    "X": periodo_1_dataset['train_X'],
}

dataset_for_validating = {
    "X": periodo_1_dataset['val_X'],
    "X_ori": periodo_1_dataset['val_X_ori'],
}

dataset_for_testing = {
    "X": periodo_1_dataset['test_X'],
    "X_ori": periodo_1_dataset['test_X_ori']
}


# In[20]:


dataset_for_testing['X']


# ## Imputación baseline: mediante la mediana

# In[21]:


import numpy as np

# Copiar el array original
X_imputed_ori = dataset_for_testing['X'].copy()

# Calcular la mediana de cada columna ignorando los NaN
column_medians = np.nanmedian(X_imputed_ori, axis=0)

# Calcular la media de cada fila ignorando los NaN
row_means = np.nanmean(X_imputed_ori, axis=1)

# Reemplazar las columnas que son completamente NaN con la media de las filas
# Usamos la media de las filas para las columnas que tienen NaN en la mediana
column_medians[np.isnan(column_medians)] = np.nanmean(row_means)

# Evitar impresión completa de grandes arrays
np.set_printoptions(threshold=10)

# Reemplazar los NaN en cada columna con su respectiva mediana
X_imputed = np.where(np.isnan(X_imputed_ori), column_medians, X_imputed_ori)

# Cambiar la configuración de numpy para mostrar el array completo
np.set_printoptions(threshold=np.inf)  # Mostrar todos los elementos

# Limitar la impresión a los primeros 100 elementos, por ejemplo
np.set_printoptions(threshold=100)

# Imprimir las medianas calculadas
X_imputed


# ## Imputación por la media

# In[22]:


import numpy as np

# Copiar el array original
X_imputed_ori = dataset_for_testing['X'].copy()

# Calcular la media de cada columna ignorando los NaN
column_means = np.nanmean(X_imputed_ori, axis=0)

# Calcular la media de cada fila ignorando los NaN
row_means = np.nanmean(X_imputed_ori, axis=1)

# Reemplazar las columnas que son completamente NaN con la media de las filas
# Usamos la media de las filas para las columnas que tienen NaN en la media
column_means[np.isnan(column_means)] = np.nanmean(row_means)

# Evitar impresión completa de grandes arrays
np.set_printoptions(threshold=10)

# Reemplazar los NaN en cada columna con su respectiva media
X_imputed_mean = np.where(np.isnan(X_imputed_ori), column_means, X_imputed_ori)

# Cambiar la configuración de numpy para mostrar el array completo
np.set_printoptions(threshold=np.inf)  # Mostrar todos los elementos

# Limitar la impresión a los primeros 100 elementos, por ejemplo
np.set_printoptions(threshold=100)

# Imprimir las medias calculadas
X_imputed_mean


# ## Graficar el primer periodo de las tres variables de contaminación

# In[23]:


import matplotlib.pyplot as plt
import warnings

# Configurar el tamaño global de la figura
plt.rcParams["figure.figsize"] = [12,4]

# Graficar con un color personalizado
periodo_1[["CO"]].plot(color='blue')

warnings.filterwarnings("ignore")

# Mostrar el gráfico
plt.show()


# In[24]:


periodo_1_df[["NO2"]].plot(color='#6c3b2a')


# In[25]:


periodo_1_df[["PM10"]].plot(color='black')


# ## Graficar el segundo periodo de las tres variables de contaminación

# In[26]:


import matplotlib.pyplot as plt

# Configurar el tamaño global de la figura
plt.rcParams["figure.figsize"] = [12,4]

# Graficar con un color personalizado
periodo_2[["CO"]].plot(color='blue')

# Mostrar el gráfico
plt.show()


# In[27]:


periodo_2[["NO2"]].plot(color='#6c3b2a')


# In[28]:


periodo_2[["PM10"]].plot(color='black')


# In[29]:


dataset_for_validating


# ## Imputación usando SAITS

# In[30]:


import torch
from torch.nn import SmoothL1Loss
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pypots.imputation import SAITS

saits = SAITS(
    n_steps=24,
    n_features=periodo_1_dataset['n_features'],
    n_layers=3,              # Aumentamos capacidad sin pasarnos
    d_model=128,
    d_ffn=256,               # Feedforward más potente
    n_heads=4,
    d_k=32,
    d_v=32,
    dropout=0.3,             # Dropout moderado
    attn_dropout=0.2,
    diagonal_attention_mask=True,
    ORT_weight=1,
    MIT_weight=1,

    # Entrenamiento
    batch_size=64,           # Más pequeño, mejora generalización
    epochs=200,              # Más paciencia para aprender
    patience=20,             # Early stopping más paciente
    num_workers=0,
    device=None,

    saving_path="tutorial_results/imputation/saits",
    model_saving_strategy="best",
)

# Optimizer con menor tasa de aprendizaje y más regularización
optimizer = Adam(saits.model.parameters(), lr=5e-5, weight_decay=1e-4)

# Scheduler que reduce el LR si la pérdida no mejora
scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

# Función de pérdida SmoothL1Loss en vez de MAE puro
loss_fn = SmoothL1Loss(beta=0.1)  


# In[31]:


# Entrenar el modelo
saits.fit(train_set=dataset_for_training, val_set=dataset_for_validating)


# In[32]:


saits_results = saits.predict(dataset_for_testing)
saits_imputation = saits_results["imputation"]


# In[33]:


saits_imputation


# In[34]:


# Crear el índice con las fechas y horas de 2023-03-03 00:00:00 hasta 2023-03-14 23:00:00
date_range = pd.date_range(start="2023-03-03 00:00:00", end="2023-03-14 23:00:00", freq="h", tz="UTC")


# In[35]:


import pandas as pd
import numpy as np

# Supongamos que 'saits_imputation' es tu numpy array de shape (12, 24, 114)
# Reestructuramos el array para convertirlo en un formato adecuado para s1cies temporales multivariantes
# Transponemos y aplanamos para que tengamos (12 * 24, 114)


X_reshaped_saits = saits_imputation.reshape(-1, len(features))  # Ahora X_reshaped tiene la forma (12 * 24, 114)


# Crear el índice con las fechas y horas de 2023-03-03 00:00:00 hasta 2023-03-14 23:00:00
date_range = pd.date_range(start="2023-03-03 00:00:00", end="2023-03-14 23:00:00", freq="h", tz="UTC")

# Creamos el DataFrame
periodo_1_imputed = pd.DataFrame(X_reshaped_saits, columns=features, index=date_range)

# Mostrar las primeras filas del DataFrame
periodo_1_imputed


# In[36]:


import pandas as pd
import numpy as np

# Supongamos que 'saits_imputation' es tu numpy array de shape (12, 24, 114)
# Reestructuramos el array para convertirlo en un formato adecuado para series temporales multivariantes
# Transponemos y aplanamos para que tengamos (12 * 24, 114)

X_reshaped = X_imputed.reshape(-1, len(features))  # Ahora X_reshaped tiene la forma (12 * 24, 114)

# Crear el índice con las fechas y horas de 2023-03-03 00:00:00 hasta 2023-03-14 23:00:00
date_range = pd.date_range(start="2023-03-03 00:00:00", end="2023-03-14 23:00:00", freq="h", tz="UTC")

# Creamos el DataFrame
periodo_1_median_imputed = pd.DataFrame(X_reshaped, columns=features, index=date_range)

# Mostrar las primeras filas del DataFrame
periodo_1_median_imputed


# In[37]:


import pandas as pd
import numpy as np

# Supongamos que 'saits_imputation' es tu numpy array de shape (12, 24, 114)
# Reestructuramos el array para convertirlo en un formato adecuado para series temporales multivariantes
# Transponemos y aplanamos para que tengamos (12 * 24, 114)

X_reshaped_mean = X_imputed_mean.reshape(-1, len(features))  # Ahora X_reshaped tiene la forma (12 * 24, 114)

# Crear el índice con las fechas y horas de 2023-03-03 00:00:00 hasta 2023-03-14 23:00:00
date_range = pd.date_range(start="2023-03-03 00:00:00", end="2023-03-14 23:00:00", freq="h", tz="UTC")

# Creamos el DataFrame
periodo_1_mean_imputed = pd.DataFrame(X_reshaped_mean, columns=features, index=date_range)

# Mostrar las primeras filas del DataFrame
periodo_1_mean_imputed


# ## Imputación con información local: Backward y Forward Fill

# <img src="backward_fill.png" width="600" height="500">

# <img src="forward_fill.png" width="600" height="500">

# In[38]:


import pandas as pd
import numpy as np

X_imputed_ori = dataset_for_testing['X'].copy()

X_reshaped_fill = X_imputed_ori.reshape(-1, len(features))  # Ahora X_reshaped tiene la forma (12 * 24, 114)

# Crear el índice con las fechas y horas de 2023-03-03 00:00:00 hasta 2023-03-14 23:00:00
date_range = pd.date_range(start="2023-03-03 00:00:00", end="2023-03-14 23:00:00", freq="h", tz="UTC")

# Creamos el DataFrame
periodo_1_fill = pd.DataFrame(X_reshaped_fill, columns=features, index=date_range)

X_imputed_backward = periodo_1_fill.bfill(axis=0)
X_imputed_backward = X_imputed_backward.ffill(axis=0)

X_imputed_forward = periodo_1_fill.ffill(axis=0)
X_imputed_forward = X_imputed_forward.bfill(axis=0)

# Eliminar las columnas 'WS' y 'WD'
X_imputed_backward_clean = X_imputed_backward.drop(columns=["WS", "WD"])
X_imputed_forward_clean = X_imputed_forward.drop(columns=["WS", "WD"])

pd.reset_option('display.max_columns')
pd.reset_option('display.max_rows')


# ## Interpolación lineal usando PyPots

# In[39]:


import pandas as pd
import numpy as np

X_imputed_ori = dataset_for_testing['X'].copy()

X_reshaped_ori = X_imputed_ori.reshape(-1, len(features))  # Ahora X_reshaped tiene la forma (12 * 24, 114)

# Crear el índice con las fechas y horas de 2023-03-03 00:00:00 hasta 2023-03-14 23:00:00
date_range = pd.date_range(start="2023-03-03 00:00:00", end="2023-03-14 23:00:00", freq="h", tz="UTC")

# Creamos el DataFrame
periodo_1_lerp_ori = pd.DataFrame(X_reshaped_ori, columns=features, index=date_range)


# In[40]:


periodo_1_lerp = periodo_1_lerp_ori.interpolate(method='linear')


# In[41]:


periodo_1_lerp


# ## Comparación de gráficas con los datos imputados

# ## Serie Temporal NO2 con Datos Faltantes
# 

# In[42]:


periodo_1_df.loc["2023-03-03":"2023-03-14"][["NO2"]].plot(color='#6c3b2a')


# In[43]:


columns


# ## Serie Temporal NO2 imputada con la mediana

# In[44]:


periodo_1_median_imputed.loc["2023-03-03":"2023-03-14"][["NO2"]].plot(color='black')


# In[45]:


periodo_1_df.loc["2023-03-03":"2023-03-14"][["NO2"]].plot(color='#6c3b2a')


# ## Serie Temporal NO2 imputada mediante SAITS

# In[46]:


periodo_1_imputed.loc["2023-03-03":"2023-03-14"][["NO2"]].plot(color='black')


# ## Serie PM10 con datos faltantes

# In[47]:


periodo_1_df.loc["2023-03-03":"2023-03-14"][["PM10"]].plot(color='black')


# ## Serie PM10 imputada mediante la mediana

# In[48]:


periodo_1_median_imputed.loc["2023-03-03":"2023-03-14"][["PM10"]].plot(color='black')


# ## Serie PM10 con datos faltantes

# In[49]:


periodo_1_df.loc["2023-03-03":"2023-03-14"][["PM10"]].plot(color='black')


# ## Serie PM10 imputada mediante SAITS

# In[50]:


periodo_1_imputed.loc["2023-03-03":"2023-03-14"][["PM10"]].plot(color='black')


# ## Serie Temporal NO2 imputada mediante SAITS

# In[51]:


periodo_1_imputed.loc["2023-03-03":"2023-03-14"][["PM10"]].plot(color='black')


# ## Serie Temporal CO datos faltantes

# In[52]:


periodo_1_df.loc["2023-03-03":"2023-03-14"][["CO"]].plot(color='blue')


# ## Serie Temporal CO imputada mediante forward fill

# In[53]:


X_imputed_forward[["CO"]].plot(color='blue')


# ## Serie Temporal CO datos faltantes

# In[54]:


periodo_1_df.loc["2023-03-03":"2023-03-14"][["CO"]].plot(color='blue')


# ## Serie Temporal CO imputada mediante backwards fill

# In[55]:


X_imputed_backward[["CO"]].plot(color='blue')


# ## Serie Temporal CO imputada mediante la mediana

# In[56]:


periodo_1_median_imputed.loc["2023-03-03":"2023-03-14"][["CO"]].plot(color='blue')


# ## Serie Temporal CO imputada mediante SAITS

# In[57]:


periodo_1_imputed.loc["2023-03-03":"2023-03-14"][["CO"]].plot(color='blue')


# ## Serie Temporal CO con datos faltantes

# In[58]:


import matplotlib.pyplot as plt

df_temp = periodo_1_df.loc["2023-03-03":"2023-03-14"]

plt.plot(df_temp.index, df_temp["TEMP"], color='blue', linestyle='-', marker='o', markersize=4)  # Línea y puntos
plt.title("Temperatura entre el 3 y el 14 de marzo")
plt.ylabel("TEMP")
plt.xticks(rotation=45)
plt.grid(True)
plt.show()


# ## Serie Temporal CO imputada mediante Interpolación lineal

# In[59]:


periodo_1_lerp.loc["2023-03-03":"2023-03-14"][["TEMP"]].plot(color='blue')


# # Comparación de SAITS con métodos baseline

# In[60]:


def calculate_metrics(y_true, y_pred, mask):
    """Calcula MAE, MSE, RMSE y MRE.

    Args:
        y_true: Valores reales.
        y_pred: Valores predichos.
        mask: Máscara para filtrar valores.

    Returns:
        Una tupla con (rmse, mse, mae, mre).
    """
    mae = calc_mae(y_pred, y_true, mask)
    mse = calc_mse(y_pred, y_true, mask)
    rmse = calc_rmse(y_pred, y_true, mask)
    mre = calc_mre(y_pred, y_true, mask)
    return rmse, mse, mae, mre



# ## Imputación con la mediana

# In[61]:


from pypots.nn.functional import calc_mae, calc_mse, calc_rmse, calc_mre

periodo_1_dataset['X_indicating_mask'] = np.isnan(dataset_for_testing['X']).astype(int)

# Crear una copia de los datos originales
X_ori_fixed = np.copy(periodo_1_dataset['test_X_ori'])

# Reemplazar NaNs por 0 (valor constante)
X_ori_fixed[np.isnan(X_ori_fixed)] = 0

# Verificar si quedan NaNs
num_nans_fixed = np.isnan(X_ori_fixed).sum()
print(f"Número de NaNs en test_X_ori después de imputación (con 0): {num_nans_fixed}")

# Calculate Mean Squared Error (MSE)
mtesting_mae, mtesting_mse, mtesting_rmse, mtesting_mre = calculate_metrics(X_ori_fixed, X_imputed, periodo_1_dataset['X_indicating_mask'])

# Print the errors
print(f"Testing Mean Absolute Error (MAE): {mtesting_mae:.4f}")
print(f"Testing Mean Squared Error (MSE): {mtesting_mse:.4f}")
print(f"Testing Root Mean Squared Error (RMSE): {mtesting_rmse:.4f}")
print(f"Testing Mean Relative Error (MRE): {mtesting_mre:.4f}")


# ## Imputación con la media

# In[62]:


from pypots.nn.functional import calc_mae, calc_mse, calc_rmse, calc_mre

# Aplanar 'X_indicating_mask' a la forma (288, 114)
X_indicating_mask_flat = periodo_1_dataset['X_indicating_mask'].reshape(-1, len(features))

# Asegurarse de que X_ori_fixed y periodo_1_mean_imputed tengan la misma forma
X_ori_fixed_flat = X_ori_fixed.reshape(-1, len(features))  # Aplanar para que tenga la forma (288, 114)

# Convertir periodo_1_mean_imputed a numpy array si es un DataFrame
periodo_1_mean_imputed = periodo_1_mean_imputed.to_numpy() if isinstance(periodo_1_mean_imputed, pd.DataFrame) else periodo_1_mean_imputed

# Verificar que ambas matrices tienen la misma forma
assert periodo_1_mean_imputed.shape == X_ori_fixed_flat.shape == X_indicating_mask_flat.shape, (
    f"Las formas no coinciden: {periodo_1_mean_imputed.shape}, "
    f"{X_ori_fixed_flat.shape}, {X_indicating_mask_flat.shape}"
)


mean_testing_mae, mean_testing_mse, mean_testing_rmse, mean_testing_mre = calculate_metrics(X_ori_fixed_flat, periodo_1_mean_imputed, X_indicating_mask_flat)
# Print the errors
print(f"Testing Mean Absolute Error (MAE): {mean_testing_mae:.4f}")
print(f"Testing Mean Squared Error (MSE): {mean_testing_mse:.4f}")
print(f"Testing Root Mean Squared Error (RMSE): {mean_testing_rmse:.4f}")
print(f"Testing Mean Relative Error (MRE): {mean_testing_mre:.4f}")


# ## Forward Fill y Backwards Fill

# In[63]:


from pypots.nn.functional import calc_mae, calc_mse, calc_rmse, calc_mre
import numpy as np
import pandas as pd

features_fill = [feature for feature in features if feature not in ["WS", "WD"]]
# Crear una copia de los datos originales
X_ori_fixed = np.copy(periodo_1_dataset['test_X_ori'])
X_ori_fixed = np.delete(X_ori_fixed, [111,112],axis=2)
                        
X_indicating_mask_cleaned = np.delete(periodo_1_dataset['X_indicating_mask'], [111, 112], axis=2)

# 1. Aplanar 'X_indicating_mask'
X_indicating_mask_flat = X_indicating_mask_cleaned.reshape(-1, len(features_fill))
# 1. Aplanar 'X_ori_fixed'

X_ori_fixed_flat = X_ori_fixed.reshape(-1, len(features_fill))
# Reemplazar NaNs por 0 (valor constante)
X_ori_fixed_flat[np.isnan(X_ori_fixed_flat)] = 0

# 3. Convertir 'X_imputed_forward_clean' y 'X_imputed_backward_clean' a numpy arrays si es necesario
if isinstance(X_imputed_forward_clean, pd.DataFrame):
    X_imputed_forward_clean = X_imputed_forward_clean.to_numpy()
if isinstance(X_imputed_backward_clean, pd.DataFrame):
    X_imputed_backward_clean = X_imputed_backward_clean.to_numpy()    

#4 Aplanar forward y backward
X_imputed_forward_clean = X_imputed_forward_clean.reshape(-1, len(features_fill))
X_imputed_backward_clean = X_imputed_backward_clean.reshape(-1, len(features_fill))

# Calcular las métricas para cada método y guardarlas en variables
mean_testing_rmse_forward, mean_testing_mse_forward, mean_testing_mae_forward, mean_testing_mre_forward= calculate_metrics(X_ori_fixed_flat, X_imputed_forward_clean, X_indicating_mask_flat)

mean_testing_rmse_backward, mean_testing_mse_backward, mean_testing_mae_backward, mean_testing_mre_backward = calculate_metrics(X_ori_fixed_flat, X_imputed_backward_clean, X_indicating_mask_flat)

# Print the errors
print("Forward Fill:")
print(f"Testing Mean Absolute Error (MAE): {mean_testing_mae_forward:.4f}")
print(f"Testing Mean Squared Error (MSE): {mean_testing_mse_forward:.4f}")
print(f"Testing Root Mean Squared Error (RMSE): {mean_testing_rmse_forward:.4f}")
print(f"Testing Mean Relative Error (MRE): {mean_testing_mre_forward:.4f}")

print("\nBackward Fill:")
print(f"Testing Mean Absolute Error (MAE): {mean_testing_mae_backward:.4f}")
print(f"Testing Mean Squared Error (MSE): {mean_testing_mse_backward:.4f}")
print(f"Testing Root Mean Squared Error (RMSE): {mean_testing_rmse_backward:.4f}")
print(f"Testing Mean Relative Error (MRE): {mean_testing_mre_backward:.4f}")


# ## Imputación con Interpolación lineal

# In[66]:


from pypots.nn.functional import calc_mae, calc_mse, calc_rmse, calc_mre


X_ori_fixed = np.copy(periodo_1_dataset['test_X_ori'])
X_ori_fixed = np.delete(X_ori_fixed, [111,112],axis=2)
X_indicating_mask_cleaned = np.delete(periodo_1_dataset['X_indicating_mask'], [111, 112], axis=2)

X_indicating_mask_flat = X_indicating_mask_cleaned.reshape(-1, len(features_fill))

# Asegurarse de que X_ori_fixed y periodo_1_mean_imputed tengan la misma forma
X_ori_fixed_flat = X_ori_fixed.reshape(-1, len(features_fill))
# Reemplazar NaNs por 0 (valor constante)
X_ori_fixed_flat[np.isnan(X_ori_fixed_flat)] = 0

# Convertir periodo_1_mean_imputed a numpy array si es un DataFrame

periodo_1_lerp_112 = periodo_1_lerp.drop(columns=["WS", "WD"])
periodo_1_lerp_array = np.array(periodo_1_lerp_112)
periodo_1_lerp_flat = periodo_1_lerp_array.reshape(-1, len(features_fill))

linear_testing_mae, linear_testing_mse, linear_testing_rmse, linear_testing_mre = calculate_metrics(X_ori_fixed_flat, periodo_1_lerp_flat, X_indicating_mask_flat)

print(f"Testing Mean Absolute Error (MAE): {linear_testing_mae:.4f}")
print(f"Testing Mean Squared Error (MSE): {linear_testing_mse:.4f}")
print(f"Testing Root Mean Squared Error (RMSE): {linear_testing_rmse:.4f}")
print(f"Testing Mean Relative Error (MRE): {linear_testing_mre:.4f}")


# ## Imputación con SAITS

# In[68]:


from pypots.nn.functional import calc_mae, calc_mse, calc_rmse, calc_mre

periodo_1_dataset['X_indicating_mask'] = np.isnan(periodo_1_dataset['test_X']).astype(int)

# Crear una copia de los datos originales
X_ori_fixed = np.copy(periodo_1_dataset['test_X_ori'])

# Reemplazar NaNs por 0 (valor constante)
X_ori_fixed[np.isnan(X_ori_fixed)] = 0

# Verificar si quedan NaNs
num_nans_fixed = np.isnan(X_ori_fixed).sum()
print(f"Número de NaNs en test_X_ori después de imputación (con 0): {num_nans_fixed}")


testing_mae, testing_mse, testing_rmse, testing_mre =calculate_metrics( saits_imputation, X_ori_fixed,  periodo_1_dataset['X_indicating_mask'],)

# Print the errors
print(f"Testing Mean Absolute Error (MAE): {testing_mae:.4f}")
print(f"Testing Mean Squared Error (MSE): {testing_mse:.4f}")
print(f"Testing Root Mean Squared Error (RMSE): {testing_rmse:.4f}")
print(f"Testing Mean Relative Error (MRE): {testing_mre:.4f}")


# In[ ]:





# In[70]:


import pandas as pd

# Crear una lista de tuplas (nombre del método, rmse, mse, mae, mre)
error_data = [
    ("Median Imputation", mtesting_rmse, mtesting_mse, mtesting_mae, mtesting_mre),
    ("Mean Imputation", mean_testing_rmse, mean_testing_mse, mean_testing_mae, mean_testing_mre),
    ("Forward Fill", mean_testing_rmse_forward, mean_testing_mse_forward, mean_testing_mae_forward, mean_testing_mre_forward),
    ("Backward Fill", mean_testing_rmse_backward, mean_testing_mse_backward, mean_testing_mae_backward, mean_testing_mre_backward),
    ("Linear Imputation", linear_testing_mae, linear_testing_mse, linear_testing_rmse, linear_testing_mre),
    ("SAITS Imputation", testing_rmse, testing_mse, testing_mae, testing_mre),
]


# Crear el DataFrame usando pd.DataFrame.from_records()
error_table = pd.DataFrame.from_records(
    error_data,
    columns=["Method", "RMSE", "MSE", "MAE", "MRE"],
    index="Method"  # Establecer la columna "Method" como índice
).round(4)  #Redondear los valores a 4 decimales



# Mostrar la tabla transpuesta para mejor visualización
error_table.T

