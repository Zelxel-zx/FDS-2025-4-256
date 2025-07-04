import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import json
from geopandas import gpd
from shapely.wkt import loads


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
df['category_id'] = df['category_id'].map(cat_map)

print("Columnas:", df.columns)
df.info()

#Convertir variables
df['trending_date'] = pd.to_datetime(df['trending_date'], format='%y.%d.%m', errors='coerce')
df['publish_time'] = pd.to_datetime(df['publish_time'], utc=True, errors='coerce')
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

