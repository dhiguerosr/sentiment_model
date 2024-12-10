import pandas as pd
import matplotlib.pyplot as plt

# Cargar el archivo CSV con las predicciones
input_file = "test_with_sentiments.csv"
df = pd.read_csv(input_file, sep="|")

# Asegurarse de que las columnas necesarias existen
required_columns = ["Texto", "Sentimiento", "Fecha"]
for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"El archivo debe contener una columna llamada '{col}'.")
    
# Convertir la columna 'fecha' a formato datetime
df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce")

# Filtrar datos del año 2022
df_2022 = df[df["Fecha"].dt.year == 2022]

# Crear una columna con el mes de cada publicación
df_2022["Mes"] = df_2022["Fecha"].dt.month

# Agrupar por mes y sentimiento
sentiments_by_month = df_2022.groupby(["Mes", "Sentimiento"]).size().unstack(fill_value=0)

# Calcular el porcentaje de publicaciones por mes
percentages = sentiments_by_month.div(sentiments_by_month.sum(axis=1), axis=0) * 100

# Graficar los resultados
percentages.plot(kind="bar", stacked=True, figsize=(10, 6))

plt.title("Porcentaje de Publicaciones Agrupadas por Sentimiento (2022)")
plt.xlabel("Mes")
plt.ylabel("Porcentaje")
plt.legend(title="Sentimiento")
plt.xticks(ticks=range(0, 12), labels=[
    "Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio",
    "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"
], rotation=45)
plt.tight_layout()

# Guardar el gráfico
output_file = "sentiments_2022.png"
plt.savefig(output_file)
print(f"Gráfico guardado como: {output_file}")

# Mostrar el gráfico
plt.show()

# Respuesta al porcentaje de publicaciones agrupadas por sentimiento
print("\nPorcentaje de publicaciones agrupadas por sentimiento para 2022:")
print(percentages)
