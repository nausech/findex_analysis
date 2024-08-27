import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns


data = pd.read_csv('./findex.csv', delimiter=';')

print(data.head())
print(data.columns)

# Filtrar los datos donde 'type' segun el tipo
filtered_data = data[data['type'] == 'Q']

numeric_features = ['percentage', 'idh']
numeric_data = filtered_data[numeric_features]

imputer = SimpleImputer(strategy='mean')
numeric_data_imputed = imputer.fit_transform(numeric_data)

scaler = StandardScaler()
numeric_data_scaled = scaler.fit_transform(numeric_data_imputed)

pca = PCA(n_components=2)
principal_components = pca.fit_transform(numeric_data_scaled)

principal_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])

principal_df = pd.concat([principal_df, filtered_data[['country', 'year', 'type']].reset_index(drop=True)], axis=1)

plt.figure(figsize=(10, 8))
sns.scatterplot(data=principal_df, x='PC1', y='PC2', hue='year', palette='viridis')
plt.title('PCA de Inclusión Financiera (Type U)')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.legend(title='Año')
plt.grid(True)
plt.show()

explained_variance = pca.explained_variance_ratio_
print(f"Varianza explicada por la PC1: {explained_variance[0]:.2f}")
print(f"Varianza explicada por la PC2: {explained_variance[1]:.2f}")
