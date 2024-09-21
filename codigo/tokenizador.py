from transformers import DistilBertTokenizer

# Creamos una instancia del tokenizador DistilBERT
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Oración de ejemplo que queremos tokenizar
sentence = "El gato está durmiendo en la alfombra."

# Tokenizamos la oración usando el tokenizador
tokens = tokenizer.tokenize(sentence)

# Mostramos los tokens generados
print("Tokens:", tokens)

# Convertimos los tokens a IDs numéricos utilizando el vocabulario del tokenizador
input_ids = tokenizer.convert_tokens_to_ids(tokens)

# Mostramos los IDs numéricos correspondientes a los tokens
print("IDs numéricos:", input_ids)

