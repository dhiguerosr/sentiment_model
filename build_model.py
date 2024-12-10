import pandas as pd 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
from joblib import dump

# Definir una función para el tokenizer
def custom_tokenizer(text):
    return text.split(',')

df = pd.read_csv("train.csv", sep="|")

# Manejo de valores nulos
df["Texto"] = df["Texto"].fillna("")  # Reemplazar valores nulos con cadenas vacías
df["Texto"] = df["Texto"].astype(str)  # Asegurarse de que todos los valores sean cadenas

# Codificación de etiquetas (sentimientos)
label_encoder = LabelEncoder()
df["Sentimiento"] = label_encoder.fit_transform(df["Sentimiento"])  # Convierte a [0, 1, 2]

vectorizer = TfidfVectorizer(tokenizer=custom_tokenizer, lowercase=False, max_features=2000)

x_data = vectorizer.fit_transform(df["Texto"]).toarray()
y_data = df["Sentimiento"]

X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)

# Modelo de clasificación (Multinomial Naive Bayes)
model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Guardar el modelo y el vectorizador
dump(model, "sentiment_model.joblib")
dump(vectorizer, "vectorizer.joblib")
dump(label_encoder, "label_encoder.joblib")

print("Modelo y vectorizador guardados.")