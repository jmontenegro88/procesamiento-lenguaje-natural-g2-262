# Mini-proyecto Fundamentos NLP y LSTM

Este proyecto implementa un sistema de detección de spam en mensajes SMS mediante técnicas de procesamiento de lenguaje natural, embeddings y redes neuronales recurrentes LSTM.

El caso de estudio se basa en el dataset público [`ucirvine/sms_spam`](https://huggingface.co/datasets/ucirvine/sms_spam) disponible en Hugging Face el cual tiene 5,574 mensajes SMS en inglés. El notebook compara un modelo clásico de referencia con dos implementaciones de LSTM y analiza sus resultados desde una perspectiva técnica y pedagógica.

## Objetivos

- Explorar las características lingüísticas y estadísticas de los mensajes SMS.
- Establecer un baseline con TF-IDF y Regresión Logística.
- Analizar similitud semántica y embeddings con spaCy.
- Construir una LSTM mínima usando PyTorch.
- Comparar configuraciones avanzadas con PyTorch Lightning.
- Evaluar el efecto del desbalance de clases y del umbral de decisión.

## Contenido del notebook

El notebook [`sms_spam_lstm_miniproyecto.ipynb`](sms_spam_lstm_miniproyecto.ipynb) contiene las siguientes etapas:

1. Preparación del entorno y configuración reproducible.
2. Carga del dataset y análisis exploratorio de los mensajes.
3. Distribución de clases, longitud, vocabulario y patrones heurísticos.
4. Baseline de TF-IDF con Regresión Logística.
5. Exploración semántica con embeddings de spaCy, similitud, aritmética vectorial y análisis OOV.
6. LSTM mínima con `Embedding -> LSTM -> Linear` y entrenamiento manual en PyTorch.
7. LSTM avanzada con PyTorch Lightning, bidireccionalidad, callbacks y checkpoints.
8. Comparación de arquitecturas mediante accuracy, precision, recall, F1 y ROC-AUC.
9. Análisis de errores, matriz de confusión y selección de umbral.
10. Clasificación de mensajes SMS personalizados.

## Ejecución en Google Colab

El notebook está preparado para ejecutarse en Google Colab, que es el entorno recomendado para este proyecto.

1. Abrir [`sms_spam_lstm_miniproyecto.ipynb`](sms_spam_lstm_miniproyecto.ipynb) en Google Colab.
2. Ejecutar las celdas en orden desde el inicio.
3. La primera sección instalar las dependencias necesarias y descarga el modelo `en_core_web_md` de spaCy.
4. Ejecutar la celda de entrenamiento de la LSTM mínima antes de la celda de evaluación en test.
5. Para la sección avanzada, permitir que finalice el barrido de arquitecturas antes de revisar la comparación de modelos.

El checkpoint `best_simple_lstm.pt` se genera automáticamente durante el entrenamiento. Si no existe, la celda de evaluación informa la situación y utiliza los pesos actuales del modelo.

## Tecnologías principales

- Python
- PyTorch
- PyTorch Lightning
- torchmetrics
- Hugging Face Datasets
- spaCy y `en_core_web_md`
- scikit-learn
- pandas, NumPy, Matplotlib y Seaborn

## Resultados y conclusiones

El análisis muestra que:

- El dataset presenta un desbalance importante entre mensajes `ham` y `spam`.
- Las URLs, números largos, símbolos monetarios y ciertos términos transaccionales son señales relevantes.
- TF-IDF con Regresión Logística constituye un baseline fuerte para este problema.
- La LSTM permite modelar el orden de los tokens y comparar una solución manual con un flujo reproducible basado en Lightning.
- Precision, recall, F1 y ROC-AUC son más informativas que accuracy por sí sola.
- El umbral de clasificación puede ajustarse según se quiera reducir falsos positivos o aumentar la detección de spam.

## Autor

**JHON MONTENEGRO**  
Estudiante MIAA de la Universidad ICESI.
