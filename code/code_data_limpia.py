import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from geopandas import gpd
from shapely.geometry import Point
import folium
from folium import plugins
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor, plot_tree, export_text

df = pd.read_csv('data/GBvideos_cc50_202101_limpio.csv')

# Convertir fechas a datetime
df['trending_date'] = pd.to_datetime(df['trending_date'])  # formato ISO por defecto
df['publish_time'] = pd.to_datetime(df['publish_time'])

# Asegurar que trending_date esté en UTC
df['trending_date'] = df['trending_date'].dt.tz_localize('UTC')

# Convertir publish_time a UTC si ya tiene zona horaria, o localizarlo si no la tiene
if df['publish_time'].dt.tz is None:
    df['publish_time'] = df['publish_time'].dt.tz_localize('UTC', ambiguous='NaT')
else:
    df['publish_time'] = df['publish_time'].dt.tz_convert('UTC')

# Convertir la columna geometry a objetos Point de Shapely
def str_to_point(point_str):
    try:
        # Extraer las coordenadas del string 'POINT (lon lat)'
        coords = point_str.replace('POINT (', '').replace(')', '').split()
        return Point(float(coords[0]), float(coords[1]))
    except:
        return None

# Convertir la columna geometry a objetos Point
df['geometry'] = df['geometry'].apply(str_to_point)

# Crear el GeoDataFrame
gdf = gpd.GeoDataFrame(df, geometry='geometry', crs='EPSG:4326')

# Análisis por categoría
def analisis_categorias(df):
    # 1. Categorías más tendencia
    cat_tendencia = df['category_id'].value_counts()
    
    # 2. Likes por categoría
    likes_categoria = df.groupby('category_id')['likes'].sum().sort_values(ascending=False)
    dislikes_categoria = df.groupby('category_id')['dislikes'].sum().sort_values(ascending=False)
    
    # 3. Ratio likes/dislikes por categoría
    ratio_likes_dislikes = df.groupby('category_id').agg({
        'likes': 'sum',
        'dislikes': 'sum'
    }).assign(ratio=lambda x: x['likes'] / x['dislikes']).sort_values('ratio', ascending=False)
    
    # 4. Ratio vistas/comentarios por categoría
    ratio_vistas_comentarios = df.groupby('category_id').agg({
        'views': 'sum',
        'comment_count': 'sum'
    }).assign(ratio=lambda x: x['views'] / x['comment_count']).sort_values('ratio', ascending=False)
    
    return cat_tendencia, likes_categoria, dislikes_categoria, ratio_likes_dislikes, ratio_vistas_comentarios

# Análisis temporal
def analisis_temporal(df):
    tendencias_tiempo = df.groupby('trending_date').size()
    return tendencias_tiempo

# Análisis por canal
def analisis_canales(df):
    canales_frecuentes = df['channel_title'].value_counts()
    return canales_frecuentes

# Análisis geográfico
def analisis_geografico(gdf):
    # Crear mapas de calor para diferentes métricas
    def create_heatmap(data, column, filename):
        # Crear mapa centrado en UK
        m = folium.Map(location=[54.5, -3.5], zoom_start=6)
        
        # Preparar datos para el mapa de calor
        heat_data = [[point.y, point.x, value] for point, value 
                    in zip(data.geometry, data[column])]
        
        # Añadir capa de mapa de calor
        heat = plugins.HeatMap(heat_data, radius=15, blur=10)
        heat.add_to(m)
        
        # Crear un grupo de marcadores
        marker_cluster = plugins.MarkerCluster().add_to(m)
        
        # Agrupar datos por ubicación para reducir el número de marcadores
        grouped_data = data.groupby(['geometry']).agg({
            'category_id': lambda x: x.value_counts().index[0],  # categoría más común
            'channel_title': lambda x: x.value_counts().index[0],  # canal más común
            'views': 'sum',
            'likes': 'sum',
            'dislikes': 'sum',
            'comment_count': 'sum',
            'title': 'count'  # contar cuántos videos hay en este punto
        }).reset_index()
        
        # Añadir marcadores con información agregada
        for idx, row in grouped_data.iterrows():
            # Preparar el contenido del popup con información detallada
            popup_content = f"""
            <div style="width: 300px;">
                <h4>Información del Punto</h4>
                <b>Número de videos:</b> {row['title']}<br>
                <b>Categoría más común:</b> {row['category_id']}<br>
                <b>Canal más común:</b> {row['channel_title']}<br>
                <hr>
                <b>Estadísticas totales:</b><br>
                • Vistas: {row['views']:,}<br>
                • Likes: {row['likes']:,}<br>
                • Dislikes: {row['dislikes']:,}<br>
                • Comentarios: {row['comment_count']:,}<br>
                • Tasa de engagement: {(row['likes']/row['views']*100):.2f}%<br>
            </div>
            """
            
            # Crear popup con estilo
            popup = folium.Popup(popup_content, max_width=350)
            
            # Añadir marcador
            folium.Marker(
                location=[row.geometry.y, row.geometry.x],
                popup=popup,
                icon=folium.Icon(color='red', icon='info-sign')
            ).add_to(marker_cluster)
        
        # Añadir control de capas
        folium.LayerControl().add_to(m)
        
        # Guardar mapa
        m.save(f'mapa_calor_{filename}.html')
    
    # Generar mapas de calor para diferentes métricas
    create_heatmap(gdf, 'views', 'views')
    create_heatmap(gdf, 'likes', 'likes')
    create_heatmap(gdf, 'dislikes', 'dislikes')
    
    # Calcular y generar mapa de engagement rate
    gdf['engagement_rate'] = gdf['likes'] / gdf['views']
    create_heatmap(gdf, 'engagement_rate', 'engagement_rate')
    
    # Retornar estadísticas agregadas por geometría
    return gdf.groupby('geometry').agg({
        'views': 'sum',
        'likes': 'sum',
        'dislikes': 'sum',
        'engagement_rate': 'mean'
    })

# Ejecutamos análisis
print("\nAnálisis por categorías:")
cat_tend, likes_cat, dislikes_cat, ratio_ld, ratio_vc = analisis_categorias(df)
print("\nCategorías más tendencia:")
print(cat_tend.head())
print("\nCategorías con más likes:")
print(likes_cat.head())

print("\nAnálisis temporal:")
tend_tiempo = analisis_temporal(df)
plt.figure(figsize=(12, 6))
tend_tiempo.plot()
plt.title('Volumen de videos en tendencia a lo largo del tiempo')
plt.show()

print("\nCanales más frecuentes en tendencias:")
canales = analisis_canales(df)
print(canales.head())

print("\nAnálisis geográfico:")
metricas_geo = analisis_geografico(gdf)
print("\nEstadísticas por región:")
print(metricas_geo)

def preparar_datos_prediccion(df):
    # Características para la predicción
    features = ['category_id', 'comment_count']
    
    # Preparar variables categóricas
    df_model = pd.get_dummies(df[features], columns=['category_id'])
    
    # Convertir fechas a características numéricas
    df_model['dias_desde_publicacion'] = (df['trending_date'].dt.tz_localize(None) - 
                                        df['publish_time'].dt.tz_localize(None)).dt.total_seconds() / (24*60*60)
    
    return df_model

def entrenar_modelo_prediccion(X, y):
    # Dividir datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Escalar características
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Entrenar modelo
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Evaluar modelo
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return model, mse, r2, scaler

def evaluar_factibilidad_predicciones(df):
    """
    Evaluación detallada de la factibilidad de predicción de métricas (Pregunta 9)
    usando árbol de regresión
    """
    # Configurar visualización inicial
    fig_corr = plt.figure(figsize=(15, 20))
    
    # 1. Correlación entre variables
    plt.subplot(4, 1, 1)
    variables = ['views', 'likes', 'dislikes', 'comment_count']
    corr = df[variables].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlación entre Métricas')
    plt.close(fig_corr)
    
    # 2. Distribución de las variables objetivo
    plt.subplot(4, 1, 2)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, var in enumerate(['views', 'likes', 'dislikes']):
        epsilon = 1
        valores = df[var].copy()
        valores = valores.replace(0, epsilon)
        log_valores = np.log10(valores)
        
        # Crear el histograma
        sns.histplot(data=log_valores, ax=axes[i])
        axes[i].set_title(f'Distribución de {var} (log10)')
        
        # Establecer ticks y etiquetas de manera correcta
        min_val = np.floor(log_valores.min())
        max_val = np.ceil(log_valores.max())
        ticks = np.arange(min_val, max_val + 1)
        axes[i].set_xticks(ticks)
        axes[i].set_xticklabels([f'{10**x:,.0f}' for x in ticks], rotation=45)
    plt.tight_layout()
    plt.close(fig)
    
    # 3. Evaluación del árbol de regresión para cada métrica
    metricas = ['views', 'likes', 'dislikes']
    resultados = {}
    
    for metrica in metricas:
        print(f"\nAnálisis para {metrica}:")
        
        # Preparar datos
        X = preparar_datos_prediccion(df)
        y = df[metrica]
        
        # Escalar los datos
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        
        # Entrenar árbol de regresión con parámetros específicos
        modelo = DecisionTreeRegressor(
            random_state=42,
            max_depth=3,  # Limitar profundidad para mejor interpretabilidad
            min_samples_split=50  # Mínimo de muestras para dividir un nodo
        )
        modelo.fit(X_train, y_train)
        
        # Evaluar modelo
        y_pred = modelo.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        resultados[metrica] = {
            'MSE': mse,
            'R2': r2,
            'Feature Importance': dict(zip(X.columns, modelo.feature_importances_))
        }
        
        # Visualizar predicciones vs valores reales
        fig_pred = plt.figure(figsize=(8, 8))
        plt.scatter(y_test, y_pred, alpha=0.5)
        
        # Convertir a array de numpy para evitar problemas de tipo
        y_test_array = np.array(y_test)
        y_pred_array = np.array(y_pred)
        
        min_val = float(min(y_test_array.min(), y_pred_array.min()))
        max_val = float(max(y_test_array.max(), y_pred_array.max()))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        plt.xlabel('Valores reales')
        plt.ylabel('Predicciones')
        plt.title(f'Predicción vs Real para {metrica}')
        plt.grid(True)
        plt.savefig(f'prediccion_vs_real_{metrica}.png')
        plt.close(fig_pred)
        
        # Visualizar árbol
        fig_tree = plt.figure(figsize=(20, 10))
        plot_tree(modelo, feature_names=X.columns, filled=True, rounded=True)
        plt.title(f"Árbol de Regresión para {metrica}")
        plt.savefig(f'arbol_regresion_{metrica}.png')
        plt.close(fig_tree)
        
        # Imprimir reglas del árbol
        print(f"\nReglas del árbol para {metrica}:")
        reglas = export_text(modelo, feature_names=list(X.columns))
        print(reglas)
    
    # Imprimir resultados detallados
    print("\nEVALUACIÓN DE FACTIBILIDAD DE PREDICCIONES")
    print("\nPregunta 9: ¿Es factible predecir el número de 'Vistas', 'Me gusta' o 'No me gusta'?")
    print("\nResultados usando Árbol de Regresión:")
    
    for metrica, res in resultados.items():
        print(f"\n{metrica.upper()}:")
        print(f"- R² Score: {res['R2']:.4f} (Un R² más cercano a 1 indica mejor predicción)")
        print(f"- Error Cuadrático Medio: {res['MSE']:.2e}")
        
        print("\nVariables más importantes para la predicción:")
        sorted_features = sorted(res['Feature Importance'].items(), key=lambda x: x[1], reverse=True)
        for feature, importance in sorted_features[:5]:
            print(f"  * {feature}: {importance:.4f}")
        
        # Evaluación de factibilidad
        if res['R2'] > 0.7:
            print("\nFactibilidad: ALTA")
            print("* El modelo puede hacer predicciones confiables")
        elif res['R2'] > 0.5:
            print("\nFactibilidad: MEDIA")
            print("* Las predicciones tienen precisión moderada")
        else:
            print("\nFactibilidad: BAJA")
            print("* Se necesita más investigación o datos adicionales")

def visualizar_tendencias_categorias(df):
    """Visualización para requerimientos 1, 2 y 3: Análisis por categorías"""
    plt.figure(figsize=(15, 20))
    
    # 1. Categorías más tendencia
    plt.subplot(4, 1, 1)
    cat_counts = df['category_id'].value_counts()
    sns.barplot(x=cat_counts.values, y=cat_counts.index)
    plt.title('Categorías de Videos más Tendencia')
    plt.xlabel('Número de Videos')
    
    # 2. Likes por categoría
    plt.subplot(4, 1, 2)
    likes_by_cat = df.groupby('category_id')['likes'].mean().sort_values(ascending=True)
    sns.barplot(x=likes_by_cat.values, y=likes_by_cat.index)
    plt.title('Promedio de Likes por Categoría')
    plt.xlabel('Promedio de Likes')
    
    # 3. Ratio likes/dislikes
    plt.subplot(4, 1, 3)
    ratio = df.groupby('category_id').agg({
        'likes': 'sum',
        'dislikes': 'sum'
    })
    ratio['ratio'] = ratio['likes'] / ratio['dislikes']
    ratio = ratio['ratio'].sort_values(ascending=True)
    sns.barplot(x=ratio.values, y=ratio.index)
    plt.title('Ratio Likes/Dislikes por Categoría')
    plt.xlabel('Ratio (Likes/Dislikes)')
    
    # 4. Ratio vistas/comentarios
    plt.subplot(4, 1, 4)
    ratio_vc = df.groupby('category_id').agg({
        'views': 'sum',
        'comment_count': 'sum'
    })
    ratio_vc['ratio'] = ratio_vc['views'] / ratio_vc['comment_count']
    ratio_vc = ratio_vc['ratio'].sort_values(ascending=True)
    sns.barplot(x=ratio_vc.values, y=ratio_vc.index)
    plt.title('Ratio Vistas/Comentarios por Categoría')
    plt.xlabel('Ratio (Vistas/Comentarios)')
    
    plt.tight_layout()
    plt.savefig('analisis_categorias.png')
    plt.close()

def visualizar_tendencias_tiempo(df):
    """Visualización para requerimiento 5: Tendencias temporales"""
    plt.figure(figsize=(15, 10))
    
    # Tendencia general
    plt.subplot(2, 1, 1)
    videos_por_dia = df.groupby('trending_date').size()
    videos_por_dia.plot(kind='line')
    plt.title('Volumen de Videos en Tendencia a lo Largo del Tiempo')
    plt.xlabel('Fecha')
    plt.ylabel('Número de Videos')
    
    # Tendencia por categoría
    plt.subplot(2, 1, 2)
    pivot = pd.crosstab(df['trending_date'], df['category_id'])
    pivot.plot(kind='area', stacked=True)
    plt.title('Distribución de Categorías a lo Largo del Tiempo')
    plt.xlabel('Fecha')
    plt.ylabel('Número de Videos')
    plt.legend(title='Categoría', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig('analisis_temporal.png')
    plt.close()

def visualizar_canales(df):
    """Visualización para requerimiento 6: Análisis de canales"""
    plt.figure(figsize=(15, 10))
    
    # Top 20 canales más frecuentes
    plt.subplot(2, 1, 1)
    top_channels = df['channel_title'].value_counts().head(20)
    sns.barplot(x=top_channels.values, y=top_channels.index)
    plt.title('Top 20 Canales más Frecuentes en Tendencias')
    plt.xlabel('Número de Apariciones')
    
    # Bottom 20 canales (excluyendo los que aparecen solo una vez)
    plt.subplot(2, 1, 2)
    bottom_channels = df['channel_title'].value_counts()
    bottom_channels = bottom_channels[bottom_channels > 1].tail(20)
    sns.barplot(x=bottom_channels.values, y=bottom_channels.index)
    plt.title('20 Canales menos Frecuentes en Tendencias\n(excluyendo apariciones únicas)')
    plt.xlabel('Número de Apariciones')
    
    plt.tight_layout()
    plt.savefig('analisis_canales.png')
    plt.close()

def analizar_comentarios(df):
    """Visualización para requerimiento 8: Análisis de comentarios"""
    plt.figure(figsize=(15, 10))
    
    # Relación entre vistas y comentarios
    plt.subplot(2, 1, 1)
    plt.scatter(np.log10(df['views']), np.log10(df['comment_count']), alpha=0.5)
    plt.title('Relación entre Vistas y Comentarios (escala log)')
    plt.xlabel('Log10(Vistas)')
    plt.ylabel('Log10(Comentarios)')
    
    # Distribución de comentarios por categoría
    plt.subplot(2, 1, 2)
    comments_by_cat = df.groupby('category_id')['comment_count'].mean().sort_values(ascending=True)
    sns.barplot(x=comments_by_cat.values, y=comments_by_cat.index)
    plt.title('Promedio de Comentarios por Categoría')
    plt.xlabel('Promedio de Comentarios')
    
    plt.tight_layout()
    plt.savefig('analisis_comentarios.png')
    plt.close()

# Ejecutar todas las visualizaciones
print("\n" + "="*80)
print("ANÁLISIS DE VIDEOS EN TENDENCIA DE YOUTUBE GB")
print("="*80)

print("\n1-4. ANÁLISIS POR CATEGORÍAS")
print("-"*50)
visualizar_tendencias_categorias(df)
print("✓ Generado: analisis_categorias.png")
print("  - Categorías más tendencia")
print("  - Likes/dislikes por categoría")
print("  - Ratios de engagement")

print("\n5. ANÁLISIS TEMPORAL")
print("-"*50)
visualizar_tendencias_tiempo(df)
print("✓ Generado: analisis_temporal.png")
print("  - Volumen de videos en tendencia")
print("  - Distribución por categoría en el tiempo")

print("\n6. ANÁLISIS DE CANALES")
print("-"*50)
visualizar_canales(df)
print("✓ Generado: analisis_canales.png")
print("  - Top 20 canales más frecuentes")
print("  - 20 canales menos frecuentes (excluyendo únicos)")

print("\n7. ANÁLISIS GEOGRÁFICO")
print("-"*50)
metricas_geo = analisis_geografico(gdf)
print("\nEstadísticas por región:")
print(metricas_geo)

print("\n8. ANÁLISIS DE COMENTARIOS")
print("-"*50)
analizar_comentarios(df)
print("✓ Generado: analisis_comentarios.png")
print("\nNota sobre análisis de sentimientos:")
print("* No es posible realizar análisis de sentimientos completo")
print("* El dataset solo contiene conteo de comentarios, no su contenido")
print("* Se recomienda recolectar datos de comentarios para análisis futuro")

print("\n9. EVALUACIÓN DE PREDICCIONES")
print("-"*50)
evaluar_factibilidad_predicciones(df)
print("✓ Generados para cada métrica (views/likes/dislikes):")
print("  - prediccion_vs_real_{metrica}.png")
print("  - arbol_regresion_{metrica}.png")

print("\n" + "="*80)
print("RESUMEN DE ARCHIVOS GENERADOS")
print("="*80)
print("1. analisis_categorias.png - Requerimientos 1, 2, 3 y 4")
print("2. analisis_temporal.png - Requerimiento 5")
print("3. analisis_canales.png - Requerimiento 6")
print("4. mapa_calor_*.html - Requerimiento 7")
print("5. analisis_comentarios.png - Requerimiento 8")
print("6. prediccion_vs_real_*.png y arbol_regresion_*.png - Requerimiento 9")
