import os
import pandas as pd 
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import re 

# Descargar recursos de NLTK si no están disponibles
nltk.download("stopwords")
nltk.download('punkt_tab')

# Definir stopwords en inglés y español
stop_words = set(stopwords.words("english") + stopwords.words("spanish"))

def prepare_text(text):
  text = text.lower()
  text = re.sub(r"\d+", "", text)
  text = text.translate(str.maketrans("", "", string.punctuation))
  tokens = word_tokenize(text)
  tokens = [word for word in tokens if word not in stop_words]
  return ",".join(tokens)

def read_input_files(folder_path, columns):
  data = []

  for file_name in os.listdir(folder_path):
    file_path = os.path.join(folder_path, file_name)
    with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
      for line in file:
        parts = line.strip().split('|')

        if len(parts) > len(columns):
          text = " ".join(parts[3:len(parts) - len(columns) + 4])
          row = parts[:3] + [prepare_text(text)] + parts[-(len(columns) - 4):]
          data.append(row)
        elif len(parts) == len(columns):
          text = parts[3]
          row = parts[:3] + [prepare_text(text)] + parts[-(len(columns) - 4):]
          data.append(row)

  df = pd.DataFrame(data, columns=columns)
  return df

# Specify the folder paths for 2021 and 2022 data
folder_2021 = "input/demo_sentiment_2021"
folder_2022 = "input/demo_2022"

# Process the 2021 and 2022 data
df_2021 = read_input_files(folder_2021, ["Identificador", "Fecha", "Fuente", "Texto", "Alcance", "Sentimiento"])
df_2022 = read_input_files(folder_2022, ["Identificador", "Fecha", "Fuente", "Texto", "Alcance"])

df_2021.to_csv("train.csv", index=False, encoding="utf-8", sep="|")
df_2022.to_csv("test.csv", index=False, encoding="utf-8", sep="|")
