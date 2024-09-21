import torch
import pandas as pd
from datasets import Dataset, load_dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from transformers import TrainingArguments, Trainer
from torch.utils.data import DataLoader

# Paso 1: Cargar el conjunto de datos desde un archivo local
file_path = "BD_POQUEIRA_CLAS_TFG_RICARDO.csv"
df = pd.read_csv(file_path)

# Crear una nueva columna 'text' combinando varias columnas de interés
df['text'] = df[['postcode', 'country', 'town', 'province']].astype(str).agg(' '.join, axis=1)

# Convertir el DataFrame de pandas a un conjunto de datos de Hugging Face
dataset = Dataset.from_pandas(df)

# Dividir el conjunto de datos en entrenamiento y prueba (80% - 20%)
train_test_split = dataset.train_test_split(test_size=0.2)
train_dataset = train_test_split['train']
test_dataset = train_test_split['test']

# Paso 2: Cargar el tokenizador y el modelo
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Paso 3: Tokenizar el conjunto de datos
def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True)

tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)

# Asegúrate de que las etiquetas estén en el formato correcto (convertir a enteros)
def format_labels(examples):
    examples['labels'] = int(examples['label'])  # Cambiar 'label' al nombre real de tu columna de etiquetas
    return examples

tokenized_train_dataset = tokenized_train_dataset.map(format_labels)
tokenized_test_dataset = tokenized_test_dataset.map(format_labels)

# Paso 4: Preparar los datos para entrenamiento
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

train_dataloader = DataLoader(tokenized_train_dataset, shuffle=True, batch_size=8, collate_fn=data_collator)
test_dataloader = DataLoader(tokenized_test_dataset, batch_size=8, collate_fn=data_collator)

# Paso 5: Configurar el entrenamiento
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",  # Cambiado 'evaluation_strategy' a 'eval_strategy'
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_test_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Paso 6: Entrenar el modelo
trainer.train()

# Paso 7: Evaluar el modelo
results = trainer.evaluate()
print(results)
