import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Leer el archivo CSV
data = pd.read_csv('./findex.csv', delimiter=';')

# Mostrar las primeras filas del DataFrame y los nombres de las columnas
print(data.head())
print(data.columns)

# Filtrar los datos donde 'type' sea igual a 'A'
#filtered_data = data[data['type'] == 'A']

# Determinar el año con más registros
most_common_year = data['year'].value_counts().idxmax()

# Filtrar los datos por el año con más registros
filtered_data_by_year = data[data['year'] == most_common_year]

# Filtrar los datos por el año con más registros
filtered_data = filtered_data_by_year[filtered_data_by_year['type'] == 'U']

# Pivotar el DataFrame de formato largo a ancho usando la columna 'name'
pivoted_data = filtered_data.pivot(index=['country', 'year', 'type'], columns='name', values='percentage')

# Resetear el índice para convertir las columnas de índice de vuelta a columnas normales
pivoted_data.reset_index(inplace=True)

# imprimir base de datos pivotada en excel
pivoted_data.to_excel('pivoted_data.xlsx', index=False)

# Mostrar las primeras filas del DataFrame pivotado
print(pivoted_data.head())

# Contar cuantos NaN tiene por columna
print(pivoted_data.isna().sum())

# Seleccionar solo las columnas numéricas para el PCA (después de pivotar)
numeric_features = pivoted_data.columns.difference(['country', 'year', 'type'])
numeric_data = pivoted_data[numeric_features]

# Estandarizar los datos numéricos
scaler = StandardScaler()
numeric_data_scaled = scaler.fit_transform(numeric_data)

# Realizar el PCA, especificando el número de componentes principales
pca = PCA(n_components=2)
principal_components = pca.fit_transform(numeric_data_scaled)

# Crear un DataFrame con los componentes principales
principal_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])

# Añadir información adicional para la interpretación
principal_df = pd.concat([principal_df, pivoted_data[['country', 'year', 'type']].reset_index(drop=True)], axis=1)

# Visualización de los resultados en un gráfico de dispersión
plt.figure(figsize=(10, 8))
sns.scatterplot(data=principal_df, x='PC1', y='PC2', hue='country', palette='viridis')
plt.title(f'PCA de Inclusión Financiera (Type U, Año {most_common_year})')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.legend(title='País')
plt.grid(True)
plt.show()

# Varianza explicada por cada componente principal
explained_variance = pca.explained_variance_ratio_
print(f"Varianza explicada por la PC1: {explained_variance[0]:.2f}")
print(f"Varianza explicada por la PC2: {explained_variance[1]:.2f}")
