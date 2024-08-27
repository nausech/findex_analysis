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

# Filtrar los datos donde 'type' sea igual a 'A'
filtered_data = data[data['type'] == 'Q']

# Seleccionar solo las columnas numéricas para el PCA
numeric_features = ['percentage', 'idh']
numeric_data = filtered_data[numeric_features]

# Tratar los valores faltantes imputando la media
imputer = SimpleImputer(strategy='mean')
numeric_data_imputed = imputer.fit_transform(numeric_data)

# Estandarizar los datos numéricos
scaler = StandardScaler()
numeric_data_scaled = scaler.fit_transform(numeric_data_imputed)

# Realizar el PCA, especificando el número de componentes principales
pca = PCA(n_components=2)
principal_components = pca.fit_transform(numeric_data_scaled)

# Crear un DataFrame con los componentes principales
principal_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])

# Añadir información adicional para la interpretación
principal_df = pd.concat([principal_df, filtered_data[['country', 'year', 'type']].reset_index(drop=True)], axis=1)

# Visualización de los resultados en un gráfico de dispersión
plt.figure(figsize=(10, 8))
sns.scatterplot(data=principal_df, x='PC1', y='PC2', hue='year', palette='viridis')
plt.title('PCA de Inclusión Financiera (Type U)')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.legend(title='Año')
plt.grid(True)
plt.show()

# Varianza explicada por cada componente principal
explained_variance = pca.explained_variance_ratio_
print(f"Varianza explicada por la PC1: {explained_variance[0]:.2f}")
print(f"Varianza explicada por la PC2: {explained_variance[1]:.2f}")
