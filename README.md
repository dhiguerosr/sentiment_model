# Sentiment Analysis Project

## Descripción

Este proyecto aplica un modelo de aprendizaje automático para la identificación de sentimientos en publicaciones de texto. Utiliza un modelo previamente entrenado para predecir si el sentimiento de una publicación es **Positive**, **Negative**, o **Neutral**. También incluye scripts para procesar y limpiar datos, realizar predicciones, y generar visualizaciones de los resultados.

---

## Estructura del Proyecto

- **`prepare_data.py`**: Script para procesar y limpiar datos de entrada, eliminando caracteres no deseados, números y stopwords en inglés y español. Genera archivos CSV listos para ser utilizados en el modelo.
- **`build_model.py`**: Script para entrenar el modelo de clasificación de sentimientos y guardar los objetos necesarios (`sentiment_model.joblib`, `vectorizer.joblib`, `label_encoder.joblib`).
- **`predict.py`**: Script para cargar el modelo y predecir los sentimientos de un archivo CSV que contiene publicaciones sin clasificar.
- **`analyze.py`**: Script para analizar los resultados de las predicciones y generar un gráfico del porcentaje de publicaciones por sentimiento para el año 2022, desglosado por mes.
- **`train.csv`**: Archivo CSV de entrenamiento generado a partir de los datos de 2021.
- **`test.csv`**: Archivo CSV de prueba generado a partir de los datos de 2022.
- **`train_with_sentiments.csv`**: Archivo CSV generado que incluye la columna `sentimiento` con las predicciones del modelo.
- **`sentiments_2022.png`**: Gráfico generado que muestra el porcentaje de publicaciones agrupadas por sentimiento en 2022.

---

## Instalación

1. Clonar el repositorio:
```bash
   git clone https://github.com/dhiguerosr/sentimient_model.git
   cd sentiment_model
```   

2. Crear y activa un entorno virtual
```bash
python3 -m venv venv
source venv/bin/activate
```

3. Instalar dependencias
```bash
pip install -r requirements.txt
```

## Uso 

1. Preparar los datos

Ejecuta el script prepare_data.py para procesar los archivos de entrada:

```bash
python3 prepare_data.py
```

Este script generará los archivos `train.csv` y `test.csv` a partir de las carpetas `input/demo_sentiment_2021` y `input/demo_2022`.

2. Entrenar modelo 

```bash
python3 build_model.py
```
Este script generará los archivos `sentiment_model.joblib`, `vectorizer.joblib`  y `label_encoder.joblic` a partir del archivo de entrenamiento `train.csv`.


3. Predecir sentimentos

```bash
python3 predict.py
```
Utiliza el modelo generado en el paso anterior `sentiment_model.joblib`, `vectorizer.joblib`  y `label_encoder.joblic` y los archivos de entrada `test.csv` para generar un nuevo csv con la columna `Sentimiento` con el nombre `test_with_sentiments.csv`.

4. Analizar resultados 

```bash
python3 analyze.py
```
Genera un gráfico del porcentaje de publicaciones agrupadas por sentimiento para 2022, el grafico se guarda como `sentiments_2022.png`

## Archivos de entrada

Se esperan los siguientes archivos:

Carpeta `input/demo_sentiment_2021`:
Debe contener archivos de texto separados por `|` con las siguientes columnas:

* Identificador
* Fecha
* Fuente
* Texto
* Alcance
* Sentimiento

Carpeta `input/demo_2022`:
Debe contener archivos de texto separados por `|` con las siguientes columnas:

* Identificador
* Fecha
* Fuente
* Texto
* Alcance



