# Análisis de Videos en Tendencia de YouTube UK

## Objetivo

El objetivo del análisis es identificar y describir los factores que contribuyen al éxito de los videos en tendencia en YouTube Gran Bretaña. Se busca responder preguntas como:

- ¿Qué categorías de videos son las de mayor tendencia?
- ¿Qué categorías de videos son los que más gustan? ¿Y las que menos gustan?
- ¿Qué categorías de videos tienen la mejor proporción (ratio) de “Me gusta” / “No me gusta”?
- ¿Qué categorías de videos tienen la mejor proporción (ratio) de “Vistas” / “Comentarios”?
- ¿Qué Canales de YouTube son tendencia más frecuentemente? ¿Y cuáles con menos frecuencia?
- ¿En qué Estados se presenta el mayor número de “Vistas”, “Me gusta” y “No me gusta”? 
- ¿Los videos en tendencia son los que mayor cantidad de comentarios positivos reciben?
- ¿Es factible predecir el número de “Vistas” o “Me gusta” o “No me gusta”?

## Autores

- **Alessandro Bravo Castillo**
- **Nicole Vasquez Tinco** 
- **Sebastian Garcia Melendez**

## Descripción del Dataset

### Dataset Original
- **Archivo**: `data/GBvideos_cc50_202101.csv`
- **Período**: Enero 2021
- **Región**: Gran Bretaña (UK)
- **Registros**: Videos en tendencia de YouTube

### Variables Principales
- `video_id`: Identificador único del video
- `trending_date`: Fecha en que el video estuvo en tendencia
- `title`: Título del video
- `channel_title`: Nombre del canal
- `category_id`: ID de categoría (mapeado a nombres descriptivos)
- `publish_time`: Fecha y hora de publicación
- `tags`: Etiquetas del video
- `views`: Número de visualizaciones
- `likes`: Número de likes
- `dislikes`: Número de dislikes
- `comment_count`: Número de comentarios
- `thumbnail_link`: URL de la miniatura
- `comments_disabled`: Si los comentarios están deshabilitados
- `ratings_disabled`: Si las calificaciones están deshabilitadas
- `video_error_or_removed`: Si el video tiene errores o fue removido
- `description`: Descripción del video

### Mapeo de Categorías
El dataset incluye un archivo `data/GB_category_id.json` que mapea los IDs numéricos a nombres descriptivos de categorías:
- Film & Animation
- Autos & Vehicles
- Music
- Pets & Animals
- Sports
- Travel & Events
- Gaming
- People & Blogs
- Comedy
- Entertainment
- News & Politics
- Howto & Style
- Education
- Science & Technology
- Nonprofits & Activism

## Conclusiones

### Categorías de videos son las de mayor tendencia

### Categorías de videos son los que más gustan y las que menos gustan

### Categorías de videos tienen la mejor proporción (ratio) de “Me gusta” / “No me gusta”

### Categorías de videos tienen la mejor proporción (ratio) de “Vistas” / “Comentarios”

### Canales de YouTube son tendencia más frecuentemente y cuáles con menos frecuencia

### Estados que presentan el mayor número de “Vistas”, “Me gusta” y “No me gusta”

### Videos en tendencia que reciben mayor cantidad de comentarios positivos 

### ¿Sera factible predecir el número de “Vistas” o “Me gusta” o “No me gusta”??


## Licencia

MIT License

Copyright (c) 2025

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
