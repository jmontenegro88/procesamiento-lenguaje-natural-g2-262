# Mini-Proyecto: Generación de reseñas de productos con GPT-2

Este proyecto adapta `mrm8488/spanish-gpt2` para generar **reseñas sintéticas en español condicionadas por una valoración de 1 a 5 estrellas**.

## Pregunta de investigación

¿El fine-tuning mejora la perplexity y la correspondencia entre estrellas solicitadas y tono del texto respecto al modelo base, sin aumentar excesivamente la repetición?

La hipótesis se contrasta por componentes. Se espera mejor adaptación probabilística a las reseñas y se estudia si esa mejora se traduce en tono controlado y generaciones menos repetitivas.

## Objetivos

- Explorar el balance de valoraciones, las longitudes y el vocabulario del corpus.
- Limpiar los textos y evitar coincidencias normalizadas entre particiones.
- Explicar la atención causal, los logits y la predicción del siguiente token.
- Comparar GPT-2 antes y después del ajuste con contextos y semillas comunes.
- Contrastar greedy, top-k y top-p a diferentes temperaturas.
- Evaluar perplexity, diversidad, repetición, tono y similitud con entrenamiento.
- Documentar hallazgos, errores y limitaciones con evidencia de la ejecución.

## Datos

- Fuente: [SetFit/amazon_reviews_multi_es](https://huggingface.co/datasets/SetFit/amazon_reviews_multi_es), copia en español de Amazon Reviews Multi.
- Revisión fijada: `16015418b488c9186fce74b058877ea939ca934d`.
- Campos: `id`, `text`, `label`, `label_text`. Las etiquetas 0–4 se convierten en 1–5 estrellas.
- Particiones originales: 200.000 reseñas de entrenamiento, 5.000 de validación y 5.000 de prueba.
- Registros después de limpiar: **197.651 / 4.945 / 4.954**, respectivamente.
- Muestra utilizada: **3.000 / 500 / 500**, con 600 ejemplos por estrella en entrenamiento y 100 por estrella en cada evaluación.

No se encontraron textos nulos ni vacíos. Se excluyeron duplicados, textos con etiquetas contradictorias y coincidencias entre particiones antes del submuestreo. Las estadísticas de esas etapas no son aditivas: algunos casos pueden solaparse.

Las reseñas de dos estrellas fueron más largas en promedio que las de cinco: 31,4 frente a 24,9 palabras. En la muestra del EDA, el **2,5 %** excedió los 128 tokens, incluyendo contexto y EOS. La ventana elegida cubre la mayoría de los textos, pero puede truncar información de la cola de longitudes.

La copia no contiene atributos ni identificadores de producto o comprador para evaluar generalización por esas entidades. El experimento genera continuaciones de reseñas, no opiniones verificadas sobre un catálogo.

**Condiciones de uso:** la ficha del espejo no declara una licencia explícita. Deben revisarse las condiciones originales y del espejo antes de reutilizar o redistribuir datos o checkpoints. El acceso público no implica permiso irrestricto; el corpus no se incorpora al repositorio.

## Contenido del notebook

1. Problema, hipótesis, objetivos y relación con el material de clase.
2. Configuración del entorno y registro de versiones y revisiones.
3. Descarga, limpieza y deduplicación entre particiones oficiales.
4. EDA de valoraciones, palabras, caracteres, vocabulario y tokens.
5. Preparación del contexto por estrellas, etiquetas y padding.
6. Inspección de logits y generación incremental.
7. Generación del modelo base con cuatro configuraciones de decodificación.
8. Fine-tuning completo, curvas y selección del checkpoint en validación.
9. Comparación de perplexity y control con estrellas cambiadas.
10. Métricas de generación y sonda de tono TF-IDF.
11. Auditoría de similitud, revisión de errores y plantilla de evaluación humana.
12. Demo opcional, exportación, discusión y conclusiones de la ejecución.

## Arquitectura y entrenamiento

```text
Valoración solicitada + inicio de reseña
-> Tokenizador de GPT-2 en español
-> Embeddings de tokens y posiciones
-> Bloques Transformer decoder con atención causal
-> Proyección al vocabulario
-> Selección del siguiente token
-> Continuación autorregresiva de la reseña
```

Se ajusta un modelo preentrenado, no una arquitectura desde cero. La pérdida utiliza los tokens de la reseña y EOS; contexto y padding se excluyen con `-100`. El modelo desplaza internamente las etiquetas. No se enmascaran EOS reales por compartir identificador con padding.

El checkpoint declara 50.257 embeddings y el tokenizador asigna EOS al ID 50.265. Por ello se amplían los embeddings a **50.266**, se sincronizan tokens especiales y se validan los datos en CPU antes de usar CUDA. El vocabulario ampliado se mantiene igual en las mediciones antes y después del ajuste.

La ejecución utilizó **124.446.720 parámetros**, tres épocas, learning rate `2e-5`, weight decay `0.01`, batch por dispositivo de 4 y acumulación de 4: batch efectivo de 16 en un dispositivo. Se emplearon FP16 y gradient checkpointing. El mejor checkpoint se eligió solo por pérdida de validación.

## Resultados

| Indicador | GPT-2 base | GPT-2 ajustado |
| --- | --- | --- |
| NLL por token en prueba | 5,234146 | 4,211462 |
| Perplexity en prueba | 187,568824 | 67,455062 |
| Tokens evaluados | 17.199 | 17.199 |

La perplexity se redujo aproximadamente **64,0 %**. El entrenamiento tardó **4,4 minutos** y se seleccionó `checkpoint-564`, de la tercera época. Las pérdidas de validación fueron 4,370884, 4,305634 y 4,287746. La validación mejoró en las tres evaluaciones; no se observó un deterioro que permita afirmar sobreajuste dentro de ese horizonte.

El control con estrellas cambiadas elevó la NLL de **4,2115 a 4,2248**: diferencia de **0,0133 nats por token**. Sugiere sensibilidad limitada al contexto, no una correspondencia fuerte o garantizada entre estrellas y tono.

La sonda de tono obtuvo **accuracy de 0,696 y F1 macro de 0,653** en reseñas reales de prueba. Su F1 para intermedias fue 0,427, por debajo de negativas (0,753) y positivas (0,779); esa debilidad limita su interpretación en textos generados.

### Decodificación

El protocolo combina cinco valoraciones, tres inicios, dos semillas y cuatro métodos, con 50 tokens nuevos y penalización de repetición de 1,1. Hay **120 salidas originales por modelo**; al contar greedy una sola vez por contexto quedan 105 observaciones por modelo.

| Modelo | Método | Distinct-2 | Repetición de trigramas | Coincidencia de tono |
| --- | --- | --- | --- | --- |
| Base | Greedy | 0,165 | 0,448 | 0,333 |
| Ajustado | Greedy | 0,096 | 0,650 | 0,533 |
| Base | Top-k 40, temperatura 0,7 | 0,302 | 0,002 | 0,333 |
| Ajustado | Top-k 40, temperatura 0,7 | 0,403 | 0,002 | 0,333 |
| Base | Top-p 0,9, temperatura 0,7 | 0,318 | 0,050 | 0,433 |
| Ajustado | Top-p 0,9, temperatura 0,7 | 0,462 | 0,016 | 0,333 |
| Base | Top-p 0,9, temperatura 1,1 | 0,334 | 0,000 | 0,300 |
| Ajustado | Top-p 0,9, temperatura 1,1 | 0,424 | 0,000 | 0,467 |

El ajuste aumenta la repetición en greedy. El muestreo ofrece un compromiso más favorable, aunque el control de tono no mejora en todas las configuraciones. Top-p a temperatura 1,1 es una opción exploratoria por su combinación de diversidad, baja repetición y coincidencia de tono; no se declara ganador en coherencia humana.

**Ninguna generación terminó por EOS**; todas se detuvieron por el presupuesto de tokens. No hubo continuaciones vacías. Esto impide afirmar que el modelo haya aprendido a finalizar reseñas completas de manera natural.

### Similitud y evaluación humana

No hubo coincidencias normalizadas exactas con la muestra de entrenamiento ni similitudes TF-IDF de 0,8 o superiores. El mayor coseno mostrado fue **0,373569**. No hay indicios fuertes de copia literal dentro de esta auditoría, pero no se descartan paráfrasis ni memorización de datos de preentrenamiento desconocidos.

Se preparó una plantilla ciega de **24 reseñas**, pero **la ejecución guardada no contiene puntuaciones humanas completadas**. Se documenta esa limitación sin inventar calificaciones ni conclusiones sobre coherencia humana. La demo quedó mostrada como widget; sus interacciones no forman parte de las métricas experimentales.

## Conclusiones

El fine-tuning adapta el registro de GPT-2 al dominio y reduce notablemente la perplexity bajo el mismo protocolo. Sin embargo, **mejor adaptación probabilística no equivale a mejor control de estrellas, menor repetición en todos los decodificadores ni generación de reseñas completas**.

La hipótesis de menor perplexity queda respaldada en esta muestra. La de mejor control de tono tiene evidencia irregular, y la expectativa de no aumentar repetición falla con greedy. La ausencia de EOS y la evaluación humana pendiente delimitan lo que se puede concluir sobre utilidad y calidad.

El caso amplía el ejemplo del curso mediante datos propios del dominio, EDA, condicionamiento, cuatro decodificadores, evaluación antes y después, una sonda de tono y auditoría de similitud. Los errores y los resultados desfavorables se conservan como parte del análisis.

## Reproducibilidad y ejecución

La semilla inicial es 42; se utilizan 42 y 43 para generación. `QUICK_RUN=False`, `MAX_LEN=128`, `MAX_NEW_TOKENS=50` y `EPOCHS=3` corresponden a los resultados presentados. Una semilla no garantiza identidad numérica entre dispositivos.

La revisión del modelo registrada fue `fc396ce08c65ea334dcf541de7dc05cac89a29b2`. El código resuelve la revisión actual al ejecutar y la exporta; para reproducir exactamente este checkpoint debe fijarse esa revisión, en lugar de asumir que `main` permanece invariable.

| Componente | Versión registrada |
| --- | --- |
| Python | 3.13.15 |
| PyTorch | 2.11.0+cu130 |
| Transformers | 5.17.0 |
| Datasets | 4.8.5 |
| Accelerate | 1.15.0 |
| pandas | 2.2.3 |
| NumPy | 2.1.3 |
| Matplotlib | 3.10.0 |
| scikit-learn | 1.6.1 |
| huggingface_hub | 1.31.0 |

Se utilizó CUDA, pero no quedó registrado el modelo específico de GPU. IPython permite visualizar tablas y `ipywidgets` es opcional para la demo. Las versiones anteriores son las observadas en la ejecución, no un requisito instalado automáticamente ni una certificación de compatibilidad de cualquier otro entorno.

Para repetir el experimento, abre el notebook en Colab o Jupyter con un entorno preparado manualmente y ejecuta las celdas en orden desde el modelo base. `QUICK_RUN=True` sirve para una prueba reducida y no reproduce las cifras del informe. Si cambia la configuración o la muestra de evaluación humana, utiliza otra carpeta de resultados para no mezclar experimentos. No se incluyen instalaciones ni se necesitan API keys.

CPU/MPS están contemplados, pero pueden ser lentos. Si aparece un `device-side assert` de CUDA, debe reiniciarse el runtime o kernel antes de ejecutar nuevamente el código corregido; reejecutar solo la celda no restaura el contexto de GPU.

Los literales de prompts, nombres de columnas y salidas se mantienen tal como se ejecutaron, incluidas sus grafías originales. Cambiar retrospectivamente esos textos alteraría el protocolo o la evidencia. La narrativa y este README utilizan ortografía española.

El código exporta figuras, métricas, configuración, generaciones, plantilla y clave de evaluación humana, checkpoints y modelo ajustado a `resultados_resenas_gpt2/`. Esa carpeta se excluye de Git; las tablas y figuras principales permanecen en las salidas del notebook ejecutado. No deben publicarse automáticamente corpus, checkpoints o referencias textuales del entrenamiento sin revisar sus condiciones.

## Referencias

- Radford et al. (2018). [Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf).
- Keung et al. (2020). [The Multilingual Amazon Reviews Corpus](https://aclanthology.org/2020.emnlp-main.369/).
- [Documentación de generación de Transformers](https://huggingface.co/docs/transformers/en/main_classes/text_generation).

## Autor

**JHON MONTENEGRO**  
Estudiante MIAA de la Universidad ICESI. Grupo Echo.
