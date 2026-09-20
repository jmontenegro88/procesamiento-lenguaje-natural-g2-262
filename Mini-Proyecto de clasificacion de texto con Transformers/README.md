# Mini-Proyecto de clacificación de textos con Transformers

Este proyecto implementa un sistema de detección de toxicidad en comentarios escritos en español mediante procesamiento de lenguaje natural y una arquitectura Transformer entrenada desde cero.

El caso de estudio utiliza la configuración española del dataset [Multilingual Toxicity Detection Dataset](https://huggingface.co/datasets/textdetox/multilingual_toxicity_dataset), basado en diferentes corpus de detección de lenguaje tóxico. El notebook compara un baseline clásico de TF-IDF con Regresión Logística frente a dos variantes de un Transformer pequeño implementado manualmente con PyTorch.

## Objetivos

- Explorar las características lingüísticas y estadísticas de los comentarios.
- Analizar el balance de las clases tóxico y no tóxico.
- Establecer un baseline con TF-IDF y Regresión Logística.
- Entrenar un tokenizador Byte-Level BPE sobre los datos de entrenamiento.
- Implementar positional embeddings sinusoidales.
- Construir un bloque Transformer con multi-head attention desde cero.
- Comparar modelos con una y cuatro cabezas de atención.
- Evaluar los modelos mediante accuracy, precision, recall y F1.
- Analizar la matriz de confusión y los errores de clasificación.
- Interpretar los tokens con mayor atención agregada.
- Estudiar el efecto del umbral de decisión sobre precision, recall y F1.
- Documentar una configuración reproducible para repetir el experimento.

## Contenido del notebook

El notebook [`clasificacion-toxicidad-transformer-desde-cero.ipynb`](clasificacion-toxicidad-transformer-desde-cero.ipynb) contiene las siguientes etapas:

1. Preparación del entorno e instalación de dependencias.
2. Carga de la configuración española del dataset desde Hugging Face.
3. Limpieza de textos y normalización de etiquetas.
4. Análisis exploratorio de clases, longitud, mayúsculas, URLs y vocabulario.
5. División estratificada en entrenamiento, validación y prueba.
6. Baseline de TF-IDF con Regresión Logística.
7. Entrenamiento del tokenizador Byte-Level BPE.
8. Creación del dataset y los `DataLoader` de PyTorch.
9. Implementación de positional embeddings sinusoidales.
10. Implementación de multi-head attention y del bloque Transformer.
11. Entrenamiento de modelos con una y cuatro cabezas de atención.
12. Comparación de curvas de pérdida y F1 durante el entrenamiento.
13. Evaluación en el conjunto de prueba y matriz de confusión.
14. Revisión cualitativa de falsos positivos y falsos negativos.
15. Predicción sobre comentarios escritos manualmente.
16. Visualización de atención agregada por token.
17. Análisis de diferentes umbrales de clasificación.
18. Conclusiones y configuración de reproducibilidad.

## Ejecución en Google Colab

El notebook está preparado para ejecutarse en Google Colab, que es el entorno recomendado porque permite instalar las dependencias y utilizar aceleración GPU si está disponible.

1. Abrir [`clasificacion-toxicidad-transformer-desde-cero.ipynb`](clasificacion-toxicidad-transformer-desde-cero.ipynb) en Google Colab.
2. Ejecutar las celdas en orden desde el inicio.
3. Permitir que finalice la instalación de dependencias y la descarga del dataset.
4. Ejecutar el EDA antes de entrenar los modelos.
5. Esperar a que finalicen los experimentos de una y cuatro cabezas de atención.
6. Ejecutar las celdas de evaluación, análisis de errores, atención y umbrales.

La ejecución completa puede tardar dependiendo del tamaño descargado, la disponibilidad de GPU y la versión de las librerías instaladas. El notebook utiliza secuencias de 128 tokens y modelos pequeños para mantener el experimento viable en Colab.

## Tecnologías principales

- Python
- PyTorch
- Hugging Face Datasets
- Hugging Face Tokenizers y Transformers
- scikit-learn
- pandas y NumPy
- Matplotlib y Seaborn

## Arquitectura

El modelo sigue la lógica del notebook de referencia `Sesion2/1-transformers-from-scratch.ipynb`:

```text
Texto
  -> Tokenizador Byte-Level BPE
  -> Embedding de tokens
  -> Positional encoding sinusoidal
  -> Multi-head self-attention
  -> Red feed-forward y normalización
  -> Mean pooling con máscara
  -> Clasificador binario
```

Se comparan dos configuraciones:

- Transformer con una cabeza de atención.
- Transformer con cuatro cabezas de atención.

La clasificación utiliza `mean pooling` sobre los tokens válidos, ignorando el padding mediante `attention_mask`.

## Evaluación

El proyecto reporta varias métricas porque la toxicidad no debe evaluarse únicamente con accuracy:

- **Accuracy:** proporción total de predicciones correctas.
- **Precision:** proporción de comentarios predichos como tóxicos que realmente son tóxicos.
- **Recall:** proporción de comentarios tóxicos que el modelo consigue detectar.
- **F1:** equilibrio entre precision y recall.
- **Matriz de confusión:** distribución de aciertos, falsos positivos y falsos negativos.

También se evalúan umbrales entre `0.10` y `0.90`. Un umbral menor puede aumentar el recall de la clase tóxica, mientras que un umbral mayor puede reducir falsos positivos. La selección debe responder al objetivo del sistema y no únicamente a una métrica aislada.

## Reproducibilidad

La configuración documentada en el notebook utiliza:

- Dataset: `textdetox/multilingual_toxicity_dataset`, configuración `es`.
- Limpieza de textos vacíos, espacios y duplicados.
- División estratificada 70% entrenamiento, 15% validación y 15% prueba.
- `random_state=42` para las divisiones.
- Vocabulario del tokenizador de 12.000 tokens.
- Longitud máxima de 128 tokens.
- Embeddings de dimensión 96.
- Positional encoding sinusoidal.
- AdamW con learning rate `2e-4`.
- Weight decay `1e-4`.
- Cuatro épocas de entrenamiento.
- Clipping de gradiente a `1.0`.
- Semilla de PyTorch `42`.

Las ejecuciones en GPU pueden presentar pequeñas diferencias numéricas. Para comparar experimentos, se recomienda conservar las versiones de las librerías, el hardware utilizado y las métricas con tres decimales.

## Resultados y conclusiones

El análisis permite estudiar que:

- Las características superficiales, como longitud, mayúsculas y URLs, pueden diferir entre clases y deben interpretarse con cuidado.
- TF-IDF constituye un baseline fuerte y proporciona una referencia interpretable para valorar el Transformer.
- Aumentar el número de cabezas de atención no garantiza una mejora cuando el corpus y el modelo son pequeños.
- Los falsos positivos y falsos negativos evidencian la importancia del contexto, la ironía y la intención comunicativa.
- La atención agregada permite explorar qué tokens reciben mayor peso, aunque no debe interpretarse como una explicación causal definitiva.
- Ajustar el umbral permite adaptar el sistema a distintos costos de error.
- Un sistema productivo necesitaría más datos, evaluación de sesgos, calibración, validación humana y comparación con modelos preentrenados en español.

## Limitaciones

- La etiqueta binaria simplifica un fenómeno lingüístico y social complejo.
- Los comentarios pueden contener ironía o contexto insuficiente para una clasificación automática.
- El modelo se entrena desde cero y no representa el estado del arte para español.
- Las visualizaciones de atención son aproximaciones interpretativas, no pruebas de causalidad.
- El desempeño puede variar con nuevas versiones del dataset, las dependencias o el hardware.

## Autor

**JHON MONTENEGRO**  
Estudiante MIAA de la Universidad ICESI.
