"""
📈 ANÁLISIS BÁSICO DEL DATASET LIMPIO
====================================

Este script muestra insights básicos del dataset limpio de YouTube
para demostrar el valor de la limpieza de datos realizada.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Configuración
plt.style.use('default')
sns.set_palette("husl")

print("📊 CARGANDO DATASET LIMPIO...")
df = pd.read_csv('data/youtube_videos_clean.csv')

print(f"✅ Dataset cargado: {len(df):,} videos limpios con {len(df.columns)} variables")

print("\n" + "="*60)
print("🎯 INSIGHTS BÁSICOS DEL DATASET LIMPIO")
print("="*60)

# 1. TOP CATEGORÍAS
print("\n1️⃣ TOP 5 CATEGORÍAS MÁS POPULARES:")
top_categories = df['category_name'].value_counts().head()
for i, (category, count) in enumerate(top_categories.items(), 1):
    percentage = (count / len(df)) * 100
    print(f"   {i}. {category}: {count:,} videos ({percentage:.1f}%)")

# 2. MÉTRICAS DE ENGAGEMENT PROMEDIO
print("\n2️⃣ MÉTRICAS DE ENGAGEMENT PROMEDIO:")
print(f"   • Engagement Rate: {df['engagement_rate'].mean():.4f}")
print(f"   • Like Ratio: {df['like_ratio'].mean():.3f}")
print(f"   • Comment Rate: {df['comment_rate'].mean():.6f}")

# 3. PATRONES TEMPORALES
print("\n3️⃣ MEJORES HORARIOS DE PUBLICACIÓN:")
hourly_engagement = df.groupby('publish_hour')['engagement_rate'].mean().sort_values(ascending=False)
for i, (hour, engagement) in enumerate(hourly_engagement.head(3).items(), 1):
    print(f"   {i}. {hour}:00 hrs - Engagement: {engagement:.4f}")

print("\n4️⃣ DÍAS DE LA SEMANA MÁS EFECTIVOS:")
daily_views = df.groupby('publish_day_of_week')['views'].mean().sort_values(ascending=False)
for i, (day, avg_views) in enumerate(daily_views.head(3).items(), 1):
    print(f"   {i}. {day}: {avg_views:,.0f} views promedio")

# 5. VELOCIDAD DE TRENDING
print("\n5️⃣ DISTRIBUCIÓN DE VELOCIDAD DE TRENDING:")
trending_speed = df['trending_speed'].value_counts()
for speed, count in trending_speed.items():
    percentage = (count / len(df)) * 100
    print(f"   • {speed}: {count:,} videos ({percentage:.1f}%)")

# 6. CARACTERÍSTICAS DE TÍTULOS EXITOSOS
print("\n6️⃣ CARACTERÍSTICAS DE TÍTULOS EXITOSOS:")
high_engagement = df[df['engagement_rate'] > df['engagement_rate'].quantile(0.75)]
print(f"   • Longitud promedio de título: {high_engagement['title_length'].mean():.1f} caracteres")
print(f"   • Palabras promedio: {high_engagement['title_word_count'].mean():.1f}")
print(f"   • Score clickbait promedio: {high_engagement['title_clickbait_score'].mean():.2f}")

print("\n" + "="*60)
print("✨ El dataset limpio permite análisis profundos y confiables!")
print("📚 Cada insight es posible gracias al proceso de limpieza aplicado.")
print("="*60)
