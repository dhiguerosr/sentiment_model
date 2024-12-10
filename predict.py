import pandas as pd
from joblib import load

def custom_tokenizer(text):
    return text.split(',')

# Cargar el modelo y los objetos necesarios
model = load("sentiment_model.joblib")
vectorizer = load("vectorizer.joblib")
label_encoder = load("label_encoder.joblib")

# Cargar el archivo CSV
input_file = "test.csv"
output_file = "test_with_sentiments.csv"
df = pd.read_csv(input_file, sep="|")

# Verificar que la columna de texto exista
if "Texto" not in df.columns:
    raise ValueError("El archivo CSV debe contener una columna llamada 'text'.")

# Manejo de valores nulos en la columna de texto
df["Texto"] = df["Texto"].fillna("")

# Transformar los datos de texto con el vectorizador
x_data = vectorizer.transform(df["Texto"])

# Transformar los datos de texto con el vectorizador
X = vectorizer.transform(df["Texto"])
# Realizar las predicciones
predicted_sentiments = model.predict(X)

# Convertir las predicciones a etiquetas originales
predicted_labels = label_encoder.inverse_transform(predicted_sentiments)

# Añadir las predicciones al DataFrame original
df["Sentimiento"] = predicted_labels

# Guardar el nuevo DataFrame con la columna 'sentimiento'
df.to_csv(output_file, index=False, sep="|")
print(f"Archivo guardado con predicciones: {output_file}")