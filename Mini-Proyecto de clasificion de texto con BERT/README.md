# Mini-Proyecto: Clasificación de noticias en español con BERT

Este proyecto clasifica noticias en español según las categorías temáticas incluidas en un corpus público. Sigue la estructura de los ejercicios de clasificación con Transformers y Hugging Face del curso, pero plantea un caso propio: cambia los datos, el dominio y la pregunta de análisis.

El experimento comienza con un baseline clásico de TF-IDF y Regresión Logística. Luego compara BERT (`dccuchile/bert-base-spanish-wwm-cased`) como extractor congelado, una cabeza de clasificación no lineal y el fine-tuning completo del modelo.

## Pregunta de investigación

¿Cómo se compara TF-IDF con Regresión Logística frente a BERT, congelado o ajustado al corpus periodístico, y cuáles categorías se confunden con mayor frecuencia?

La hipótesis es que el fine-tuning podría aumentar el F1 macro al adaptar las representaciones al dominio de noticias, con un costo mayor de cómputo. Los resultados se deben completar después de ejecutar el notebook; este README no presupone cuál enfoque obtiene el mejor desempeño.

## Datos

- Dataset: [`hacktoberfest-corpus-es/colmbian_spanish_news`](https://huggingface.co/datasets/hacktoberfest-corpus-es/colmbian_spanish_news).
- Tamaño publicado: 76.151 registros; contiene las particiones `train`, `valid` y `test`.
- Campos utilizados: `news_title`, `news_text_content` y `category`. El texto para el modelo combina el titular y el cuerpo.
- Acceso: repositorio público; no requiere token de Hugging Face.
- Licencia indicada por el repositorio: CC BY 2.0. Al reutilizar el corpus, conserva la atribución a la fuente y consulta sus condiciones de licencia.

El notebook conserva las particiones oficiales. Para controlar el tiempo de entrenamiento, limita a un máximo de 1.500 artículos por categoría únicamente en `train`; `valid` y `test` no se submuestrean. Por esta razón, el entrenamiento puede tener proporciones distintas a la evaluación.

## Objetivos

- Examinar la distribución de categorías, duplicados y longitud de los artículos.
- Medir truncamiento con el tokenizador de BERT para informar la elección de longitud máxima.
- Establecer una línea base clásica con TF-IDF y Regresión Logística.
- Medir cuánto aporta BERT congelado frente al baseline clásico.
- Compararla con un clasificador no lineal sobre el encoder congelado.
- Evaluar el fine-tuning completo bajo las mismas particiones y métricas.
- Analizar rendimiento por categoría, matriz de confusión y ejemplos erróneos.
- Discutir precisión y costo a partir de resultados observados, no de expectativas.

## Contenido del notebook

El archivo [`1-text-classification-with-hf.ipynb`](1-text-classification-with-hf.ipynb) contiene estas etapas:

1. Instalación de dependencias y detección del entorno.
2. Descarga anónima de archivos Parquet y revisión de las particiones.
3. Limpieza de titulares, textos, etiquetas y duplicados.
4. Exploración de distribución de categorías y longitudes.
5. Submuestreo del split de entrenamiento y mapeo de etiquetas.
6. Tokenización con BERT y estimación de truncamiento.
7. Baseline clásico de TF-IDF y Regresión Logística.
8. Entrenamiento de BERT congelado, clasificador personalizado y fine-tuning completo.
9. Comparación de accuracy y F1 macro en test.
10. Reporte por clase, matriz de confusión y revisión de errores.
11. Discusión de resultados, limitaciones y experimentos futuros.

## Ejecución en Google Colab

Colab es el entorno recomendado. El notebook no necesita secretos ni tokens.

1. Abre `1-text-classification-with-hf.ipynb` en Google Colab.
2. Ejecuta las celdas en orden, empezando por la instalación.
3. Acepta la descarga del corpus público desde Hugging Face. El tamaño total publicado es cercano a 196 MB.
4. Revisa el EDA y el porcentaje de textos que exceden `MAX_LEN` antes de entrenar.
5. Para fine-tuning completo, selecciona GPU en **Entorno de ejecución > Cambiar tipo de entorno de ejecución**. En CPU, el proceso puede ser lento.
6. Espera a que terminen los experimentos antes de ejecutar la comparación, el análisis de errores y las conclusiones.
7. Completa la discusión con las métricas y observaciones que aparezcan en tu ejecución.

Si los recursos de Colab son limitados, reduce `MIN_PER_CLASS`, `MAX_LEN` o el tamaño del batch y documenta cualquier cambio para que la comparación sea interpretable. Si instalas paquetes en un runtime que ya estaba iniciado y surge un error de importación, reinicia el runtime y vuelve a ejecutar desde el principio.

## Tecnologías

- Python
- PyTorch y `torchinfo`
- Hugging Face Transformers y Datasets
- Hugging Face Evaluate
- pandas, NumPy, Matplotlib y requests
- scikit-learn

## Arquitecturas comparadas

- **TF-IDF + Regresión Logística:** referencia clásica basada en unigramas y bigramas.
- **BERT congelado:** el encoder se mantiene fijo y se entrena la cabeza lineal de clasificación.
- **Clasificador personalizado:** encoder congelado y cabeza no lineal de varias capas.
- **Fine-tuning completo:** se actualizan los parámetros del encoder y la cabeza, con una tasa de aprendizaje menor.

Todos los enfoques usan el mismo split de entrenamiento procesado, validación y prueba y las mismas etiquetas. BERT usa su tokenizador correspondiente; el baseline clásico usa TF-IDF.

## Evaluación

- **Accuracy:** fracción total de predicciones correctas.
- **F1 macro:** promedio del F1 por categoría, con el mismo peso para cada clase.
- **Precision, recall y F1 por categoría:** muestran dónde funciona peor el clasificador.
- **Matriz de confusión normalizada:** permite observar las confusiones entre categorías.
- **Análisis de errores:** inspección de ejemplos para formular explicaciones que puedan respaldarse en el texto.

El modelo seleccionado usa F1 macro de validación. El conjunto de prueba se reserva para la evaluación reportada y la comparación final.

## Reproducibilidad

- Semillas de entrenamiento y muestreo: `42`.
- Baseline clásico: TF-IDF con unigramas y bigramas (`min_df=2`, máximo 100.000 características) y Regresión Logística (`max_iter=1000`).
- Máximo de ejemplos de entrenamiento por categoría: `MIN_PER_CLASS = 1500`.
- Longitud máxima inicial: `MAX_LEN = 128`, contrastada con una muestra tokenizada.
- Batch size: 16 en Colab y 8 fuera de Colab.
- BERT: `dccuchile/bert-base-spanish-wwm-cased`.
- Dataset: `hacktoberfest-corpus-es/colmbian_spanish_news`.

Los tiempos y resultados pueden variar según GPU, versiones de las bibliotecas y recursos disponibles. La semilla hace reproducible la configuración del notebook, pero no garantiza igualdad numérica entre distintos dispositivos.

## Resultados y conclusiones

Completa esta sección después de ejecutar el notebook:

- Compara accuracy y F1 macro de TF-IDF, BERT congelado, el clasificador personalizado y el fine-tuning.
- Identifica las categorías con más errores mediante la matriz de confusión y el reporte por clase.
- Indica si los resultados respaldan la hipótesis y relaciona cualquier mejora con su costo de cómputo.
- Registra limitaciones observadas y experimentos futuros.

## Limitaciones

- Las categorías y sus posibles errores provienen de la fuente; no son etiquetas revisadas manualmente en este proyecto.
- Algunas categorías pueden ser semánticamente cercanas y los artículos pueden tratar más de un tema.
- La muestra se limita solo para entrenamiento; las métricas reflejan las particiones de prueba publicadas, no un balance artificial.
- El split publicado no demuestra generalización a otros medios, periodos o países.
- La licencia es la indicada por el repositorio del dataset; verifica sus términos antes de redistribuir los datos.

## Autor

**JHON MONTENEGRO**  
Estudiante MIAA de la Universidad ICESI.
