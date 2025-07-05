import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import json
from geopandas import gpd
from shapely.wkt import loads
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor, plot_tree, export_text


def cantidad(df, max_categorias=5):
    cat_cols = df.select_dtypes(include=['object', 'bool']).columns

    for col in cat_cols:
        freq = df[col].value_counts(dropna=False).head(max_categorias)
        resumen = pd.DataFrame({
            'Cantidad': freq,
        })
        print(resumen)

def vacios(df):
    cat_cols = df.select_dtypes(include=['object', 'bool']).columns

    for col in cat_cols:
        print(f"\nVariable: '{col}'")
        nulos = df[col].isnull().sum()
        vacios = df[col].astype(str).str.strip().eq('').sum()
        # Solo buscar '[none]' si es texto
        corchete_none = 0
        if df[col].dtype == 'object':
            corchete_none = df[col].astype(str).str.lower().eq('[none]').sum()

        print(f"Null: {nulos}")
        print(f"'[none]': {corchete_none}")
        print(f"Cadena vacía (''): {vacios}")


df = pd.read_csv("data/GBvideos_cc50_202101.csv")

#Category_id
with open("data/GB_category_id.json", "r", encoding="utf-8") as f:
    categories = json.load(f)
cat_map = {int(item['id']): item['snippet']['title'] for item in categories['items']}
# Convertir category_id a int antes de mapear
df['category_id'] = pd.to_numeric(df['category_id'], errors='coerce')
df['category_id'] = df['category_id'].map(lambda x: cat_map.get(x, 'sin categoria'))

print("Columnas:", df.columns)
df.info()

#Convertir variables
df['trending_date'] = pd.to_datetime(df['trending_date'], format='%y.%d.%m')
df['publish_time'] = pd.to_datetime(df['publish_time'])
# Asegurar que trending_date esté en UTC
df['trending_date'] = df['trending_date'].dt.tz_localize('UTC')
# Convertir publish_time a UTC si ya tiene zona horaria, o localizarlo si no la tiene
if df['publish_time'].dt.tz is None:
    df['publish_time'] = df['publish_time'].dt.tz_localize('UTC', ambiguous='NaT')
else:
    df['publish_time'] = df['publish_time'].dt.tz_convert('UTC')
df['geometry'] = df['geometry'].apply(loads)
gdf = gpd.GeoDataFrame(df, geometry='geometry', crs='EPSG:4326')

#Eliminar columnas irrelevantes
df.drop(columns=['thumbnail_link', 'description'], inplace=True, errors='ignore')

# Estadísticas básicas: count, mean, std, min, 25%, 50%, 75%, max
print("\nEstadísticas básicas:")
print("\nNuemericas:")
print(df.select_dtypes(include='number').describe())

print("\nCategorías:")
cantidad(df)

# Verificar valores nulos
print("\nValores nulos:")
print("\nVariables numéricas:")
for col in df.select_dtypes(include='number').columns:
    null_count = df[col].isnull().sum()
    cero_count = (df[col] == 0).sum()
    print(f"{col}: {null_count} nulos, {cero_count} ceros")

print("\nVariables categóricas:")
vacios(df)

#Deteccion de outliers y tratamiento
print("\nDetección de outliers:")
numerics = ['views', 'likes', 'dislikes', 'comment_count']
for col in numerics:
    plt.figure(figsize=(6, 2))
    sns.boxplot(x=np.log10(df[col] + 1))
    plt.title(f'Boxplot {col} (antes)')
    plt.xlabel(col)
    plt.tight_layout()
    plt.show()
    
def reemplazar_outliers_por_mediana(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    mediana = df[col].median()
    outliers_mask = (df[col] < lower) | (df[col] > upper)
    outliers_count = outliers_mask.sum()
    print(f"{col}: {outliers_count} valores atípicos detectados")
    # Contar
    inferiores = (df[col] < lower).sum()
    superiores = (df[col] > upper).sum()
    print(f"{col}: outliers inferiores detectados: {inferiores}")
    print(f"{col}: outliers superiores detectados: {superiores}")
    df[col] = df[col].apply(lambda x: mediana if x < lower or x > upper else x)
    
for col in numerics:
    reemplazar_outliers_por_mediana(df, col)
    plt.figure(figsize=(6, 2))
    sns.boxplot(x=np.log10(df[col] + 1))
    plt.title(f'Boxplot {col} (después)')
    plt.xlabel(col)
    plt.tight_layout()
    plt.show()

#Tratamiento de valores nulos y '[none]'
df['tags'] = df['tags'].replace('[none]', 'sin tags')
df['category_id'] = df['category_id'].fillna('sin categoria')
