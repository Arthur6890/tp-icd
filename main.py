import pandas as pd
import os

# Diretório onde os arquivos CSV estão localizados
directory = 'data'

# Listar todos os arquivos CSV na pasta
files = [file for file in os.listdir(directory) if file.endswith('.csv')]

# Lista para armazenar os DataFrames
dataframes = []

# Ler cada arquivo CSV e adicionar à lista
for file in files:
    file_path = os.path.join(directory, file)
    df = pd.read_csv(file_path)
    dataframes.append(df)
    print(f"Arquivo {file} carregado com sucesso.")

# Concatenar todos os DataFrames em um único DataFrame
combined_df = pd.concat(dataframes, ignore_index=True)

# Exibir as primeiras linhas do DataFrame combinado
print(combined_df.head())

# Salvar o DataFrame combinado em um novo arquivo CSV (opcional)
# combined_df.to_csv('data/combined_dataset.csv', index=False)
# print("Dataset combinado salvo em 'data/combined_dataset.csv'")
