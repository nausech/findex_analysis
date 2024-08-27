import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns

# Leer el archivo CSV
data = pd.read_csv('./findex.csv', delimiter=';')

# Mostrar las primeras filas del DataFrame y los nombres de las columnas
print(data.head())
print(data.columns)

# Eliminar las columnas con más del 10% de valores NaN
threshold = 0.1 * len(data)
nan_counts = data.isna().sum()
data = data.loc[:, nan_counts <= threshold]

# Imputar los valores faltantes en las columnas numéricas
numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
imputer = SimpleImputer(strategy='mean')
data[numeric_columns] = imputer.fit_transform(data[numeric_columns])

# Determinar el año con más registros
most_common_year = data['year'].value_counts().idxmax()

# Filtrar los datos por el año con más registros
filtered_data_by_year = data[data['year'] == most_common_year]

# Filtrar los datos por el tipo 'U'
filtered_data = filtered_data_by_year[filtered_data_by_year['type'] == 'U']

# Pivotar el DataFrame de formato largo a ancho usando la columna 'name'
pivoted_data = filtered_data.pivot(index=['country', 'year', 'type'], columns='name', values='percentage')

# Resetear el índice para convertir las columnas de índice de vuelta a columnas normales
pivoted_data.reset_index(inplace=True)

# Mostrar las primeras filas del DataFrame pivotado
print(pivoted_data.head())

# Eliminar las columnas con más del 10% de valores NaN en el DataFrame pivotado
pivoted_data = pivoted_data.loc[:, pivoted_data.isna().mean() <= 0.1]

pivoted_data.to_excel('pivoted_data.xlsx', index=False)

# Imputar valores faltantes en el DataFrame pivotado
numeric_features = pivoted_data.select_dtypes(include=['float64', 'int64']).columns
imputer = SimpleImputer(strategy='mean')
pivoted_data[numeric_features] = imputer.fit_transform(pivoted_data[numeric_features])

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
explained_variance_df = pd.DataFrame({
    'Componente Principal': ['PC1', 'PC2'],
    'Varianza Explicada': explained_variance
})

# Mostrar la varianza explicada por cada componente principal
print(explained_variance_df)
