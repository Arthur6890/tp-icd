import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import statsmodels.api as sm

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

X = combined_df.drop('target_column', axis=1)  # substitua 'target_column' pelo nome da sua variável dependente
y = combined_df['target_column']

# Dividir os dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinar o modelo de regressão logística
model = LogisticRegression()
model.fit(X_train, y_train)

# Prever no conjunto de teste
y_pred = model.predict(X_test)
score = accuracy_score(y_test, y_pred)
print(f"Acurácia: {score}")

# Para obter os coeficientes e os ICs usando statsmodels
X_train_const = sm.add_constant(X_train)  # adicionar uma constante para o intercepto
logit_model = sm.Logit(y_train, X_train_const)
result = logit_model.fit()

# Obter os intervalos de confiança
conf = result.conf_int()
conf['OR'] = result.params
conf.columns = ['2.5%', '97.5%', 'OR']
print(conf)

# Previsão dos scores (probabilidades) no conjunto de teste
X_test_const = sm.add_constant(X_test)
scores = result.predict(X_test_const)

# Adicionar scores ao DataFrame de teste
X_test['score'] = scores
print(X_test.head())