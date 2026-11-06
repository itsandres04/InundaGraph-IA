# InundaGraph IA · Prototipo de vulnerabilidad vial

> Proyecto del curso de Matemáticas Discretas (Universidad del Sinú, Seccional Cartagena).
> Representa zonas piloto de Cartagena como grafos y analiza escenarios de
> inundación simulados: rutas mínimas, pérdida de conectividad y clasificación
> de criticidad de intersecciones.
>
> **Atribución de datos:** Datos © Colaboradores de OpenStreetMap (ODbL).
> Los resultados son simulaciones académicas y no sustituyen estudios
> hidrológicos oficiales ni sistemas de alerta.

## Etapa actual (2 del ABP)

- Red vial **real** descargada de OpenStreetMap con OSMnx (parser interno de
  OSMnx + descarga propia vía Overpass con reintentos) para **4 zonas piloto**:
  Centro Histórico/Getsemaní, Manga, Bocagrande y Pie de la Popa.
- Motor de rutas con **Dijkstra** y la fórmula del ABP: `we = te·(1 + β·re)`,
  con afectación `re ∈ [0, 1]`.
- Escenarios de inundación simulada: núcleo de **cierre total** dentro de un
  radio y **afectación parcial** hasta el radio completo.
- Mapa interactivo sobre geometrías viales de OpenStreetMap proyectadas a
  coordenadas métricas; al pasar el cursor sobre un centro se resaltan las
  vías cerradas y afectadas para ese escenario. Al pasar sobre una vía se
  muestra su estado (cerrada, afectada o sin afectación) y su nombre cuando
  está disponible en OpenStreetMap.
- Indicadores de conectividad: grado, intermediación aproximada, proporción
  de intersecciones con acceso a los puntos de referencia configurados y nodos
  aislados. Estos puntos no están validados como refugios.
- **Árbol de decisión** (scikit-learn) experimental con etiquetas derivadas de
  reglas explícitas del equipo. En la interfaz, el expander de intersecciones
  críticas explica esas reglas; no presenta sus resultados como predicciones
  del árbol ni como evaluación de riesgo real. Las intersecciones se
  identifican por las calles conectadas, con una etiqueta descriptiva cuando
  OpenStreetMap no proporciona nombres de vías.
- Caché en disco (`data/`) en pickle para uso instantáneo y GeoJSON
  interoperable para documentar los datos.

## Requisitos

- Python 3.12+ (probado con 3.14).
- Conexión a internet solo para la primera descarga de cada zona.

## Instalación en Windows

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Si PowerShell bloquea la activación:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
```

## Publicar en GitHub y Streamlit Community Cloud

GitHub aloja el código; Streamlit Community Cloud ejecuta la aplicación.
GitHub Pages no puede ejecutar esta aplicación de Streamlit.

1. Crea un repositorio público en GitHub.
2. Sube el código del proyecto y la carpeta `data/`, que contiene las redes
   viales cacheadas para evitar descargarlas al iniciar. No subas `.venv/`,
   `cache/` ni archivos `*.log`.
3. Entra a [Streamlit Community Cloud](https://share.streamlit.io/) y conecta
   tu cuenta de GitHub.
4. Crea una aplicación seleccionando el repositorio, la rama `main` y
   `app.py` como archivo principal.
5. En la configuración avanzada, selecciona Python 3.12 y despliega.

Las redes se obtienen de OpenStreetMap; conserva la atribución indicada en esta
documentación. La aplicación puede reconstruir el grafo desde los GeoJSON de
`data/` si el archivo pickle no está disponible.

## Ejecutar la aplicación

```powershell
streamlit run app.py
```

La primera vez, cada zona se descarga al cache local; después la carga es
instantánea. Si un servidor de Overpass está saturado, el sistema prueba
varios endpoints con esperas automáticas.
Si el pickle de una zona falta pero su GeoJSON está disponible, la red se
reconstruye desde ese archivo sin conexión y se vuelve a guardar en caché.
En la aplicación, el botón «?» abre una guía breve sobre los controles,
incluido un glosario que explica cómo el centro, los radios, el nivel de
afectación y la penalización modifican el escenario y el costo de ruta; también
describe los colores del mapa y el alcance académico de las simulaciones. Las
instrucciones detalladas de la ruta están disponibles bajo demanda debajo de
los indicadores.

## Pruebas

```powershell
python -m unittest discover -s tests -v
```

Las pruebas del motor usan la red sintética de seis nodos; las pruebas de red
real se omiten automáticamente si la caché no existe aún.

## Estructura

```text
Prototipo_Red_Vial/
├── app.py                    # Interfaz Streamlit (etapa 2)
├── requirements.txt
├── README.md
├── core/
│   ├── __init__.py
│   ├── network.py            # Motor de rutas y fórmula de costo (ABP §9.3)
│   ├── geodata.py            # Zonas piloto, descarga OSM y caché (ABP §9.7)
│   ├── analysis.py           # Grado, intermediación, aislamiento (ABP §9.5)
│   ├── criticality.py        # Árbol de decisión explicable (ABP §9.6)
│   └── visualization.py      # Mapa interactivo con geometrías de OpenStreetMap
├── data/                     # Caché (pickle, GeoJSON, crudo Overpass)
└── tests/
    └── test_network.py
```

## Lógica de costo (ABP §9.3)

`costo = tiempo_base * (1 + beta * re)`

- `re ∈ [0, 1]` es la afectación simulada del segmento.
- `beta ≥ 0` controla cuánto penaliza el escenario.
- `costo ≥ 0` siempre, condición necesaria para Dijkstra.
- Los cierres se marcan retirando la vía del grafo.
- Los regímenes y niveles de simulación son variables experimentales
  del equipo, no mediciones hidrológicas oficiales.

## Criterios de etiquetado del árbol (ABP §9.6)

- **ALTA:** el nodo queda aislado (sin acceso a ningún punto seguro) o
  pierde ≥ 2 aristas a la vez que conserva grado ≤ 2.
- **MEDIA:** pierde ≥ 1 arista, hereda afectación `re`, o su intermediación
  cae por debajo de la mitad de la base.
- **BAJA:** sin cierres ni señales de pérdida de conectividad.

El árbol reproduce estas reglas: su exactitud no demuestra capacidad de
predecir inundaciones reales.

La evaluación reserva escenarios completos mediante `leave-one-scenario-out`:
cada pliegue entrena con varios escenarios y prueba sobre uno no visto. Se
reportan F1 macro y exactitud balanceada como promedio y desviación entre
escenarios. Los escenarios provienen de variaciones reproducibles de centros
y radios sobre la misma red; por tanto, la evaluación mide transferencia entre
escenarios de esa zona, no entre zonas ni frente a inundaciones observadas.
Como las etiquetas son generadas por reglas que también usan las características,
las métricas miden qué tan bien el árbol reproduce esas reglas y no constituyen
validación predictiva independiente.

## Próxima etapa

Incorporar más puntos seguros por zona, evaluar la transferencia entre zonas y
contrastar las etiquetas con datos observados o criterios externos antes de
interpretar el árbol como predictor (ABP §7 y §10.5).
