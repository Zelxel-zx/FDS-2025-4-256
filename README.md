# Análisis de Videos en Tendencia de YouTube UK

## Objetivo

El objetivo del análisis es identificar y describir los factores que contribuyen al éxito de los videos en tendencia en YouTube Gran Bretaña. Se busca responder preguntas como:

- ¿Qué categorías de videos son las de mayor tendencia?
- ¿Qué categorías de videos son los que más gustan? ¿Y las que menos gustan?
- ¿Qué categorías de videos tienen la mejor proporción (ratio) de “Me gusta” / “No me gusta”?
- ¿Qué categorías de videos tienen la mejor proporción (ratio) de “Vistas” / “Comentarios”?
- ¿Cómo ha cambiado el volumen de los videos en tendencia a lo largo del tiempo?
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
La categoría “Music” lidera claramente con 13,754 apariciones, lo que refleja un fuerte interés del público británico por contenidos musicales. Le sigue “Entertainment”, que incluye programas, shows y contenido recreativo. Ambas categorías dominan las tendencias, superando por amplio margen a otras como People & Blogs o Film & Animation. Esta concentración indica que el público tiende a consumir y viralizar mayormente contenidos de ocio y expresión artística.

### Categorías de videos son los que más gustan y las que menos gustan
La categoría Music lidera ampliamente el ranking de likes, acumulando más del doble que la segunda categoría. Esto sugiere una fuerte conexión emocional del público con el contenido musical, además de una alta propensión a reaccionar positivamente. Le siguen categorías asociadas al entretenimiento general y contenido de expresión personal.

Las categorías con menos likes pueden deberse a varios factores: menor presencia en tendencias, menor volumen de videos, o un público menos propenso a interactuar con este tipo de contenido. La categoría “sin categoría” representa videos con datos incompletos, mientras que Shows y Travel reflejan menor engagement.

### Categorías de videos tienen la mejor proporción (ratio) de “Me gusta” / “No me gusta”
Con mayor ratio de aprobación:
La categoría "Shows" destaca por una aprobación extremadamente alta, probablemente debido a la naturaleza de los contenidos presentados (clips de televisión, avances o trailers), que suelen ser bien recibidos.

"Music" y "Pets & Animals" también muestran niveles de aprobación elevados, lo cual es coherente con su carga emocional o entretenida.

Con menor ratio de aprobación:
Categorías como News & Politics tienden a generar más desacuerdos, ya que están vinculadas a temas sensibles o controversiales.

En el caso de Education, el bajo ratio puede reflejar una menor inclinación del público a interactuar emocionalmente con estos contenidos.

### Categorías de videos tienen la mejor proporción (ratio) de “Vistas” / “Comentarios”
Categorías más comentadas proporcionalmente
Un ratio bajo de vistas/comentarios implica mayor participación, ya que los usuarios comentan con mayor frecuencia respecto a las vistas.
Un ratio alto indica baja participación relativa, es decir, el público consume el contenido pero no suele comentar.
Las categorías con mejor ratios (menos comentadas proporcionalmente) son Travel & Events, Science & Technology y Sports.

En estas categorías, el volumen de visitas es alto, pero los comentarios son relativamente bajos. Esto puede deberse a que se trata de contenidos más pasivos o visuales, donde la audiencia tiende a consumir sin necesariamente participar en la conversación

### Cambia el volumen de los videos en tendencia a lo largo del tiempo
La categoría Entertainment mantuvo una presencia dominante a lo largo del tiempo, siendo consistentemente una de las más populares.
Se observa una disminución gradual en categorías como People & Blogs y Film & Animation.
En momentos específicos, algunas categorías como Music o News & Politics aumentan su visibilidad, lo que puede coincidir con eventos o lanzamientos de alto impacto.
La categoría sin categoría, derivada de datos incompletos o no clasificados, se mantiene en niveles bajos, lo que refleja una buena calidad general del dataset

### Canales de YouTube son tendencia más frecuentemente y cuáles con menos frecuencia
Estos canales tienen una presencia más esporádica en la sección de tendencias. Muchos de ellos pueden corresponder a lanzamientos puntuales, artistas emergentes, conferencias académicas o eventos específicos que captaron atención en momentos determinados, pero sin mantener un flujo constante de viralidad.

### Estados que presentan el mayor número de “Vistas”, “Me gusta” y “No me gusta”
Los videos en tendencia se concentran geográficamente en las principales regiones urbanas del Reino Unido, como Londres, Manchester, Birmingham y Edimburgo, las cuales registran los mayores volúmenes de vistas, likes y participación general. A través del análisis espacial se identificó que la categoría más común en estas zonas es “Music”, con canales como The Tonight Show Starring Jimmy Fallon y Jimmy Kimmel Live destacando por su frecuencia. Además, regiones como el norte de Inglaterra y partes de Escocia presentaron tasas de engagement superiores al promedio, superando el 4 % en algunos casos. En conjunto, los resultados confirman que el comportamiento de los usuarios varía según la ubicación, y que las zonas metropolitanas concentran tanto el consumo como la interacción más activa con los videos virales.

### Videos en tendencia que reciben mayor cantidad de comentarios positivos 
Categorías con mayor promedio de comentarios: Music, Film & Animation, Shows, Education.
Estas categorías tienden a generar mayor conversación por parte del público. En el caso de Music, es común que los usuarios comenten sobre letras, artistas o emociones. Shows y Film & Animation suelen fomentar debates sobre tramas o personajes, mientras que Education atrae comentarios de retroalimentación o preguntas.

Categorías con menor promedio de comentarios: Travel & Events, Sports, Howto & Style.
Estas categorías, si bien pueden tener muchas vistas, tienden a generar menor número de comentarios, posiblemente porque el contenido es más visual o práctico, y no promueve tanto la discusión.

Por lo tanto, no se puede determinar si un comentario es positivo o negativo, el análisis muestra que los videos en tendencia suelen recibir más comentarios que el resto, y que ciertas categorías son más comentadas proporcionalmente que otras. Eso permite concluir que la popularidad sí estaría asociada a un mayor nivel de interacción, aunque dicha interacción requiere un análisis de sentimiento texto que es lo que no se presenta en este dataset. 


### ¿Sera factible predecir el número de “Vistas” o “Me gusta” o “No me gusta”??
Las predicciones de vistas presentan alta variabilidad. Aunque el modelo logra captar algunas tendencias generales, la dispersión entre valores reales y predichos es significativa. Esto indica que las vistas dependen de múltiples factores no considerados en el modelo actual, como el momento de publicación, la viralidad externa o el contenido específico del video.
Factibilidad de predicción de vistas: BAJA

El modelo predice con precisión moderada el número de likes, siendo la cantidad de comentarios una variable explicativa importante. También influye si el video es de la categoría Music, que suele concentrar altos niveles de interacción.
Factibilidad de predicción de likes: MEDIA

La predicción de dislikes es la más difícil. Aunque el modelo puede capturar algunos valores generales, la distribución de dislikes es menos estructurada y más dispersa, posiblemente influida por aspectos subjetivos no contenidos en el dataset (controversia, polarización, etc.).
Factibilidad de predicción de dislikes: BAJA



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
