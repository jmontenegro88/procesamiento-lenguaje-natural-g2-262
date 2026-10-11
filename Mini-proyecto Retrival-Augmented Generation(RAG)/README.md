# Asistente RAG para servicios de posgrado en Icesi

Este mini-proyecto implementa un asistente conversacional de Retrieval-Augmented Generation (RAG) para consultar información pública de servicios y procesos relacionados con posgrados de la Universidad Icesi. Sigue el flujo presentado en los notebooks de `Sesion5`, pero utiliza páginas institucionales distintas a Wikihow e incorpora comparación cuantitativa de recuperación, análisis exploratorio y evaluación cualitativa.

## Problema y alcance

Quienes estudian o consideran estudiar un posgrado pueden necesitar consultar páginas diferentes para conocer servicios, admisiones, asesoría, pagos, acompañamiento y biblioteca. El notebook recupera fragmentos de páginas oficiales y condiciona la respuesta de un modelo generativo a esos fragmentos, mostrando también los enlaces consultados.

**Pregunta de investigación:** ¿cómo cambia la recuperación de información cuando se compara BM25 con embeddings multilingües y cómo influye el tamaño de fragmento?

El corpus incluye la [Resolución 1691 de 2026, que modifica el reglamento de posgrado](https://www.icesi.edu.co/wp-content/uploads/2026/04/Reso-1691-Modifica-reglamento-de-posgrado.pdf), además de páginas institucionales de servicios. No contiene el reglamento completo ni el calendario académico. El asistente puede responder sobre la resolución solo en la medida en que el fragmento recuperado lo sustente; no debe generalizarla a todo el reglamento ni responder sobre fechas académicas sin una fuente verificable. El prototipo no sustituye a los canales oficiales de Icesi.

## Fuentes

El notebook descarga páginas directamente desde el dominio oficial `icesi.edu.co` al ejecutarse. La cobertura puede cambiar y algunos sitios pueden limitar la extracción automatizada; el estado de cada URL se muestra en una tabla y se reporta la fecha de consulta.

- [Portal de estudiantes](https://www.icesi.edu.co/estudiantes/)
- [Servicios para estudiantes de posgrado](https://www.icesi.edu.co/estudiantes-posgrado-centro-de-talento-icesi/)
- [Admisiones de posgrado](https://www.icesi.edu.co/admisiones-de-posgrado/)
- [Inscripción y forma de pago de posgrados](https://www.icesi.edu.co/inscripcion-y-forma-de-pago-posgrados/)
- [Asesores de posgrados](https://www.icesi.edu.co/asesores-posgrados/)
- [Ecosistema de acompañamiento a estudiantes](https://www.icesi.edu.co/decanatura-de-innovacion-educativa-y-fortalecimiento-del-pei/ecosistema-de-acompanamiento-a-estudiantes/)
- [Biblioteca Icesi](https://www.icesi.edu.co/biblioteca/)
- [Programa Estas en Casa](https://www.icesi.edu.co/bienestar-universitario/programa-estas-en-casa/)
- [Resolución 1691 de 2026: modifica el reglamento de posgrado](https://www.icesi.edu.co/wp-content/uploads/2026/04/Reso-1691-Modifica-reglamento-de-posgrado.pdf)

## Contenido y experimentos

El notebook [`rag-asistente-posgrados-icesi.ipynb`](./rag-asistente-posgrados-icesi.ipynb) incluye:

1. Definición del problema, alcance, hipótesis y relación con el material del curso.
2. Descarga con reporte explícito de las páginas públicas oficiales.
3. EDA de la cobertura y longitud de las páginas extraídas.
4. Fragmentación con dos tamaños y solapamientos configurables.
5. Comparación de BM25 (baseline léxico) y embeddings multilingües.
6. Evaluación de recuperación con Recall@3 y MRR sobre preguntas anotadas manualmente, separando selección y prueba.
7. Análisis de las fuentes recuperadas en consultas reservadas.
8. Recuperación híbrida BM25 + embeddings, combinada mediante Reciprocal Rank Fusion (RRF), para el chatbot.
9. Selección de citas textuales con Qwen2.5-1.5B-Instruct; cada cita se valida contra los fragmentos recuperados antes de mostrarla con un enlace clicable.
10. Demostración conversacional mediante Gradio y una respuesta explícita cuando no hay citas verificables.
11. Guía de conclusiones basada en métricas observadas y discusión de limitaciones.

## Ejecución en Google Colab

1. Carga `rag-asistente-posgrados-icesi.ipynb` en Google Colab.
2. Selecciona GPU desde **Entorno de ejecución → Cambiar tipo de entorno de ejecución**.
3. Ejecuta las celdas en orden. La instalación añade solo las dependencias extra del notebook y deja que Colab conserve sus versiones preinstaladas de NumPy, pandas, requests, Matplotlib, Seaborn, scikit-learn y PyTorch.
4. Revisa la tabla de estado de fuentes. Si Icesi devuelve `403 Forbidden`, guarda cada página web fallida como HTML (`.html`) y descarga el PDF de la resolución desde el navegador; súbelos cuando el notebook lo solicite usando los identificadores indicados en pantalla (por ejemplo, `biblioteca.html` y `reglamento_posgrado_res_1691.pdf`). El texto se extrae localmente y se conserva el enlace oficial como referencia.
5. Compara resultados en las consultas de selección y revisa después la evaluación reservada.
6. Prueba preguntas respondibles, compuestas y fuera del corpus. El chatbot presenta pasajes textuales verificados en lugar de completar respuestas con inferencias no respaldadas; revisa los enlaces y confirma que los pasajes resuelvan cada parte de la pregunta.
7. Ejecuta Gradio para grabar la demostración de la interfaz y las citas, tal como pide la consigna.

El modelo generativo descarga varios GB; GPU es recomendable y CPU puede ser lenta. El notebook fue preparado, pero no ejecutado en esta entrega. La conexión, el acceso de las páginas y la disponibilidad de GPU deben validarse en tu sesión de Colab.

Si el servidor de Icesi bloquea las solicitudes del runtime de Colab (`403 Forbidden`), el notebook ofrece subir los HTML y PDF descargados manualmente desde el navegador. La resolución se lee con `pypdf`; conserva la página oficial como procedencia y revisa la extracción antes de confiar en sus respuestas.

Si una instalación anterior quedó ejecutándose, interrúmpela con el botón de detener de Colab y restablece el entorno desde **Entorno de ejecución → Desconectar y eliminar entorno de ejecución**. Luego vuelve a abrir/subir la versión actualizada del notebook y ejecuta las celdas desde el inicio. Una instalación nueva puede tardar unos minutos, pero no debería quedarse indefinidamente en `Installing build dependencies`; si lo hace, revisa que estés usando la versión actualizada del notebook y que Colab tenga conexión a internet.

## Relación con los mini-proyectos previos

El proyecto aprovecha la estructura de entregas previas: caso y pregunta explícitos, hipótesis comprobable, EDA, baseline, comparación bajo las mismas consultas, métricas, revisión de errores, reproducibilidad y conclusiones prudentes. Del flujo de `Sesion5` toma la segmentación de documentos, recuperación, generación con contexto y chatbot; cambia por completo el corpus de Wikihow y agrega evaluación sistemática de métodos y tamaños de fragmento.

## Tecnologías

- Python, pandas, NumPy, Matplotlib y Seaborn
- Requests, BeautifulSoup y pypdf para extraer páginas y el documento PDF
- BM25 (`rank-bm25`) y Sentence Transformers para recuperación
- Transformers y PyTorch para generación
- Gradio para el chatbot

## Resultados y conclusiones

Completa esta sección después de ejecutar el notebook. Reporta Recall@3 y MRR en selección y prueba, indica las fuentes que no se descargaron, analiza ejemplos de recuperación incorrecta y contrasta la respuesta generada con sus páginas citadas. No atribuyas mejoras al modelo sin evidencia observable.

## Limitaciones

- La cobertura depende de páginas públicas y de su extracción actual; contenido dinámico puede no estar disponible.
- Las preguntas y etiquetas de relevancia son pocas y anotadas manualmente, por lo que las métricas son exploratorias.
- El corpus contiene una resolución modificatoria, no el reglamento completo ni el calendario académico; no debe presentarse como cobertura normativa exhaustiva.
- La recuperación híbrida mejora la combinación de señales léxicas y semánticas, pero no garantiza que los fragmentos recuperados respondan por completo. Las citas verificadas prueban que el texto aparece en el corpus, no que la respuesta cubra todos los matices; toda respuesta requiere verificación humana.
- El código requiere acceso a internet y la descarga de modelos; las versiones y resultados pueden variar.

## Autor

**JHON MONTENEGRO**  
Estudiante MIAA de la Universidad ICESI
