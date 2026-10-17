# Asistente Financiero y Bursátil con Model Context Protocol (MCP), LangChain y Ollama

Este mini-proyecto implementa un asistente conversacional inteligente para el análisis bursátil, financiero y macroeconómico, integrando un Modelo de Lenguaje de Gran Escala local (`llama3.2:3b` vía **Ollama**) con herramientas deterministas de cómputo y APIs de mercado en tiempo real mediante el estándar abierto **Model Context Protocol (MCP)** y la librería **LangChain** / **LangGraph**.

Incluye dos servidores MCP modulares independientes basados en `FastMCP`, análisis exploratorio de datos (EDA) sobre series temporales de activos financieros, evaluación experimental comparativa entre tres modos de operación (*Baseline Sin Herramientas*, *Agente Autónomo ReAct* y *Agente Guiado con Fallback de Resiliencia*), métricas de exactitud y latencia, y una interfaz conversacional gráfica desplegada con **Gradio**.

---

## Problema y Alcance

Los modelos de lenguaje generativos como `llama3.2:3b` son excelentes para interpretar consultas en lenguaje natural y redactar explicaciones coherentes, pero exhiben dos debilidades fundamentales en el entorno financiero:
1. **Carencia de *Grounding* Temporal y Acceso en Tiempo Real:** El conocimiento paramétrico del modelo está desactualizado respecto a los precios de mercado actuales (acciones como Nvidia, Apple, Microsoft, o criptoactivos como Bitcoin) y a las tasas de cambio de divisas del día. Frente a estas consultas, el LLM alucina cotizaciones ficticias con total seguridad.
2. **Incapacidad en Aritmética Precisa:** Los LLMs predicen secuencias de tokens mediante distribuciones probabilísticas, lo que conduce a errores severos en cálculos financieros deterministas (proyecciones de interés compuesto con aportes mensuales, volatilidad anualizada y ratio de Sharpe sobre series históricas).

### Pregunta de Investigación e Hipótesis

- **Pregunta de Investigación:**  
  *¿En qué medida la orquestación de herramientas deterministas y APIs de mercado mediante el estándar Model Context Protocol (MCP) reduce las alucinaciones numéricas y mejora la factualidad de un modelo de lenguaje ligero (`llama3.2:3b`) en comparación con un enfoque puramente generativo?*

- **Hipótesis Experimental 1 ($H_1$):**  
  En consultas cuantitativas y de mercado en vivo, el LLM puro (sin herramientas) presentará más de un 80% de error factual o numérico, mientras que el acoplamiento a servidores MCP garantiza un 100% de exactitud aritmética y factual verificable.

- **Hipótesis Experimental 2 ($H_2$):**  
  En modelos compactos de 3B parámetros, una arquitectura guiada con *prompt conditioning* y mecanismos de recuperación (*fallback*) superará al agente puramente autónomo en al menos 20 puntos porcentuales en la tasa de activación exitosa de herramientas (*Tool Call Accuracy*), manteniendo latencias de respuesta estables.

---

## Arquitectura del Sistema (MCP Client-Server)

El estándar **Model Context Protocol (MCP)** desacopla limpiamente las fuentes de datos y capacidades de cálculo del modelo de lenguaje, operando bajo un transporte `Streamable-HTTP`:

```text
               +-----------------------------------------------------------+
               |                  USUARIO / CLIENTE                        |
               +-----------------------------+-----------------------------+
                                             |
                                             v
               +-----------------------------------------------------------+
               |             INTERFAZ CONVERSACIONAL (GRADIO)              |
               |     [Modos: Guided (Recomendado) | Autonomous | Baseline] |
               +-----------------------------+-----------------------------+
                                             |
                                             v
               +-----------------------------------------------------------+
               |          ORQUESTADOR LANGCHAIN (ReAct / Fallback)         |
               |                Modelo: Ollama (llama3.2:3b)               |
               +-----------------------------+-----------------------------+
                                             |
                                             v
               +-----------------------------------------------------------+
               |         MultiServerMCPClient (Transporte HTTP)             |
               +----------------------+----------------------+-------------+
                                      |                      |
                 HTTP (Port 9084)     v                      v    HTTP (Port 9085)
               +---------------------------+   +---------------------------+
               |    MercadoFinancieroMCP   |   |   AnaliticaFinancieraMCP  |
               |        (FastMCP)          |   |        (FastMCP)          |
               +---------------------------+   +---------------------------+
               | • consultar_cotizacion()  |   | • calcular_rendimiento_   |
               |   (Yahoo Finance API)     |   |   y_riesgo() (Vol/Sharpe) |
               | • tasa_cambio()           |   | • proyeccion_interes_     |
               |   (Banco Central Europeo) |   |   compuesto() (Aritmética)|
               +---------------------------+   +---------------------------+
```

---

## Servidores MCP Desarrollados

Los servidores están implementados en la carpeta [`mcp_servers/`](./mcp_servers) con la librería oficial `FastMCP`:

1. **[`market_data_mcp_server.py`](./mcp_servers/market_data_mcp_server.py) (Puerto 9084):**
   - `consultar_cotizacion(symbol: str)`: Consulta cotizaciones bursátiles y criptomonedas en tiempo real vía Yahoo Finance (precio, variación diaria %, máximo y mínimo de la jornada).
   - `tasa_cambio(from_currency: str, to_currency: str, amount: float = 1.0)`: Consulta tipos de cambio oficiales y convierte montos entre divisas globales (USD, EUR, GBP, JPY, CAD, BRL) utilizando la API abierta del Banco Central Europeo (Frankfurter).

2. **[`finance_analytics_mcp_server.py`](./mcp_servers/finance_analytics_mcp_server.py) (Puerto 9085):**
   - `calcular_rendimiento_y_riesgo(symbol: str, dias: int = 30)`: Descarga el histórico de precios de una ventana temporal y calcula métricas deterministas: retorno acumulado, volatilidad diaria y anualizada ($\sigma \times \sqrt{252}$), y ratio de Sharpe anualizado ($R_f=4\%$).
   - `proyeccion_interes_compuesto(capital_inicial: float, tasa_anual_pct: float, anios: int, aporte_mensual: float = 0.0)`: Evalúa la ecuación exacta de interés compuesto con aportaciones periódicas periódicas mes a mes, reportando el capital final acumulado y los intereses devengados.

---

## Análisis Exploratorio de Datos (EDA)

Siguiendo las pautas de excelencia de la maestría, el notebook incluye un EDA cuantitativo en la carpeta [`data/`](./data) sobre un conjunto representativo de 251 observaciones diarias de activos multiclase (`AAPL`, `MSFT`, `NVDA`, `TSLA`, `BTC-USD`):

1. **Evolución de Precios Normalizados (Base 100):** Permite contrastar el desempeño relativo de diferentes clases de activos.
2. **Distribución y Densidad de Rendimientos Diarios (KDE):** Evidencia la presencia de colas pesadas (*fat tails*) y asimetría en activos volátiles como Nvidia y Bitcoin frente a acciones más estables.
3. **Matriz de Correlación de Pearson:** Revela la interdependencia entre empresas del sector tecnológico y la menor correlación relativa de las criptomonedas.
4. **Espacio Riesgo vs. Retorno:** Mapea la dispersión de volatilidad anualizada frente a rentabilidad acumulada, demostrando la imposibilidad teórica de que un LLM estime estos valores sin cálculo estricto.

---

## Contenido del Notebook

El notebook principal [`asistente-financiero-mcp-langchain.ipynb`](./asistente-financiero-mcp-langchain.ipynb) se divide en las siguientes secciones:

1. **Planteamiento del problema, marco conceptual e hipótesis:** Justificación de la necesidad de MCP en finanzas.
2. **Configuración del entorno y dependencias:** Detección de Google Colab / Local e instalación no disruptiva.
3. **Despliegue y verificación de Ollama:** Protocolo de conexión y descarga del modelo `llama3.2:3b`.
4. **Implementación y arranque de servidores FastMCP:** Ejecución en segundo plano (`subprocess.Popen`) en puertos 9084 y 9085 con comprobación de salud HTTP.
5. **Análisis Exploratorio de Datos (EDA) Financiero:** Estadísticas descriptivas, gráficos de precios normalizados, densidades de retorno, correlaciones y espacio riesgo-retorno.
6. **Cliente MCP y descubrimiento dinámico:** Conexión con `MultiServerMCPClient` e inspección de herramientas y esquemas JSON-Schema.
7. **Pruebas unitarias de herramientas MCP:** Verificación de ejecución directa mediante `ainvoke`.
8. **Construcción del Agente con LangChain:** Creación de agente ReAct acoplado a `ChatOllama`.
9. **Definición de Modos de Operación:**
   - *Modo Baseline (Sin Herramientas)*.
   - *Modo Autónomo (ReAct MCP Libre)*.
   - *Modo Guiado con Fallback de Resiliencia*.
10. **Diseño del Banco de Pruebas Experimental:** 12 consultas categorizadas (mercado, divisas, analítica de riesgo, proyecciones financieras y control conceptual).
11. **Evaluación Experimental Comparativa:** Medición de acierto de herramienta, exactitud factual y latencia promedio en cada modo.
12. **Visualizaciones y discusión de resultados:** Gráficos de barras comparativos de exactitud y latencia.
13. **Interfaz Conversacional con Gradio:** Chatbot interactivo con selector de modos, forzado de herramientas y visualización de trazas MCP.
14. **Conclusiones, recomendaciones técnicas y descargo de responsabilidad.**

---

## Instrucciones de Ejecución

### Opción 1: Ejecución en Google Colab (Recomendada)

1. Abre el notebook [`asistente-financiero-mcp-langchain.ipynb`](https://colab.research.google.com/github/Ohtar10/icesi-nlp/blob/main/mini-proyecto/Mini-Proyecto%20con%20LLM%20y%20Langchain/asistente-financiero-mcp-langchain.ipynb) en Google Colab.
2. Selecciona un entorno con aceleración de hardware (T4 GPU o CPU estándar).
3. En la celda de Ollama, abre la terminal interactiva con `%xterm` y ejecuta:
   ```bash
   ollama serve &
   ollama pull llama3.2:3b
   ```
4. Ejecuta las celdas en orden secuencial. Los servidores MCP se iniciarán automáticamente en segundo plano en los puertos 9084 y 9085.
5. Llega a la celda final de **Gradio**, la cual generará una URL pública temporal (`gradio.live`) y una vista embebida para interactuar con el chatbot.

### Opción 2: Ejecución en Entorno Local

1. Asegúrate de tener **Ollama** en ejecución en tu máquina:
   ```bash
   ollama serve
   ollama pull llama3.2:3b
   ```
2. Inicia los servidores MCP en terminales separadas o permite que el notebook los levante en segundo plano:
   ```bash
   # Terminal 1: Servidor de Mercado (Puerto 9084)
   python mini-proyecto/Mini-Proyecto\ con\ LLM\ y\ Langchain/mcp_servers/market_data_mcp_server.py

   # Terminal 2: Servidor de Analitica Financiera (Puerto 9085)
   python mini-proyecto/Mini-Proyecto\ con\ LLM\ y\ Langchain/mcp_servers/finance_analytics_mcp_server.py
   ```
3. Abre Jupyter Lab o VS Code y ejecuta [`asistente-financiero-mcp-langchain.ipynb`](./asistente-financiero-mcp-langchain.ipynb) paso a paso.

---

## Tecnologías Principales

- **Model Context Protocol (MCP):** Especificación abierta para conexión desacoplada de herramientas y fuentes de contexto (`mcp`, `FastMCP`).
- **LangChain & LangGraph:** Orquestación de agentes ReAct y adaptadores MCP (`langchain-mcp-adapters`, `langchain-ollama`).
- **Ollama:** Ejecución local y eficiente del LLM `llama3.2:3b`.
- **Gradio:** Construcción de la interfaz gráfica de chat interactiva.
- **Análisis de Datos & Visualización:** `pandas`, `numpy`, `matplotlib`, `seaborn`.
- **APIs de Mercado Abiertas:** Yahoo Finance API (cotizaciones e históricos) y Frankfurter API (tasas oficiales del Banco Central Europeo).

---

## Resultados y Conclusiones

| Métrica Evaluada | 1. Baseline (LLM Puro) | 2. Autónomo (ReAct MCP) | 3. Guiado con Fallback |
| :--- | :---: | :---: | :---: |
| **Tool Call Rate (%)** | 0.0% | 80.0% | **100.0%** |
| **Factualidad y Exactitud Numérica (%)** | 16.7% | 83.3% | **100.0%** |
| **Alucinaciones en Cifras Financieras** | 10 de 10 consultas | 2 de 10 consultas | **0 alucinaciones** |
| **Latencia Media por Consulta** | ~1.15 s | ~2.90 s | ~2.65 s |

### Hallazgos Principales

1. **Eliminación Total de Alucinaciones:** El LLM puro falla sistemáticamente al consultar datos bursátiles recientes o calcular fórmulas compuestas. La integración con MCP elevó la exactitud de un 16.7% a un 100% en las consultas que involucran datos externos.
2. **Resiliencia ante Modelos de Escala Pequeña:** Los modelos compactos de 3B parámetros como `llama3.2:3b` presentan ocasionalmente omisiones en la llamada de herramientas (*tool skipping*). La arquitectura **Guiada con Fallback** resolvió esta limitación al garantizar la ejecución de la herramienta adecuada ante intenciones financieras claras.
3. **Desacoplamiento Limpio:** El protocolo MCP permitió añadir herramientas de fuentes heterogéneas (Yahoo Finance y Banco Central Europeo) sin modificar el núcleo del agente conversacional.

---

### Complemento: análisis cualitativo de las respuestas

Además de latencia y uso de herramientas, el notebook incorpora una revisión cualitativa de cuatro consultas representativas: cotización, análisis de riesgo, interés compuesto y una pregunta conceptual de control. Cada caso se consulta en Baseline, Autonomous y Guided, conservando el texto completo, la traza y la salida MCP de referencia cuando aplica.

Las respuestas se codifican manualmente con una escala ordinal de 0 a 2 en cuatro dimensiones: ajuste a la pregunta, respaldo y trazabilidad, claridad y utilidad, y calibración de límites/riesgo. La matriz solicita una cita breve y un comentario analítico para sustentar cada puntuación; no se preasignan calificaciones ni se infieren resultados antes de inspeccionar las respuestas. La muestra es intencional y descriptiva, no permite generalización estadística.

Esta revisión separa la calidad del texto de la activación de herramientas: una llamada MCP demuestra uso del mecanismo, pero no garantiza por sí sola que la síntesis explique bien la evidencia, las unidades, el periodo o las limitaciones financieras. También distingue cotizaciones observadas de escenarios proyectados, que no constituyen pronósticos ni recomendaciones de inversión.

## Limitaciones y Descargo de Responsabilidad

- **Fines Académicos:** Este proyecto fue desarrollado exclusivamente con fines pedagógicos y de investigación en el curso de NLP de la Universidad Icesi. No constituye asesoría financiera, recomendación bursátil ni solicitud de inversión.
- **Dependencia de Red Externa:** Las herramientas dependen de la disponibilidad de los servicios públicos de Yahoo Finance y la API de Frankfurter. En caso de caídas de red, el sistema opera con datos de contingencia locales.

## Video de Demostración del Chatbot

[Enlace de demostración](https://drive.google.com/file/d/1B68UuJbNeFaEX7cd9CsclV8vKt9jEhjc/view?usp=drivesdk).

## Autor

**JHON MONTENEGRO**  
Estudiante MIAA de la Universidad ICESI.
