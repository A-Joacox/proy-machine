# Predicción de la duración de viajes en taxi - NYC Yellow Taxi

Proyecto final del curso de Machine Learning. Predicción de la duración de un viaje en taxi amarillo de NYC usando únicamente información disponible al momento del recojo.

Ver [`proposal.md`](proposal.md) para la formulación completa del problema.

**Grupo**: 5

**Integrantes**

- Joaquin Mercado
- Joaquin Justo
- Diego Alarcon
- Randy Rojas

## Reproducir la exploración inicial

Requiere Python 3.10+.

```bash
# 1. Clonar el repositorio y entrar
git clone <URL_DEL_REPO>
cd proy-machine

# 2. Crear entorno e instalar dependencias
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. Ejecutar el notebook de exploración
jupyter notebook notebooks/01_exploracion_inicial.ipynb
```

El notebook descarga automáticamente los datos de enero 2025 (~60 MB en Parquet) desde la TLC a `data/` la primera vez que se ejecuta. No se versionan los datos en el repositorio (ver `data/README.md`).

Todas las figuras se guardan en `reports/figures/` y las métricas del baseline en `outputs/metrics_baseline.json`. La semilla aleatoria está fijada (`SEED = 42`).

## Estructura del repositorio

```
proy-machine/
├── README.md
├── proposal.md
├── requirements.txt
├── data/                  # parquet descargado (no versionado) + Taxi_Zone_Lookup.csv
├── notebooks/
│   ├── 01_exploracion_inicial.ipynb
│   ├── 02_baselines_6meses.ipynb   # baselines + Ridge/Lasso -> outputs/metrics_6meses.json
│   ├── 03_modelos_arboles.ipynb    # Random Forest, HistGB, LightGBM -> outputs/metrics_arboles.json
│   ├── 04_busqueda_hiperparametros.ipynb  # búsqueda con TimeSeriesSplit -> outputs/metrics_tuning.json,
│   │                                      #   outputs/mejores_hiperparametros.json (tarda ~50 min)
│   └── 04b_tamano_muestra.ipynb    # efecto de usar 10/20/50 % de cada mes, chequeo de HistGB y
│                                   #   árboles con 50 % -> outputs/config_modelo_final.json
│                                   #   (tarda ~40 min, ~8 GB de RAM)
├── src/
│   ├── data.py            # descarga, limpieza y split temporal
│   ├── features.py        # features y preprocesador sin leakage
│   └── tuning.py          # validación cruzada temporal y búsqueda aleatoria para LightGBM
├── reports/
│   └── figures/
└── outputs/
```

## Fuente de datos

NYC Taxi & Limousine Commission — Trip Record Data
https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page

Datos públicos, formato Parquet, periodo usado: enero-junio 2025.


## Semana 13: interpretabilidad y limitaciones

La ejecución de la semana 13 está completada. El modelo reconstruido
reproduce las métricas de la semana 12: MAE = 3.693972 min y
RMSLE = 0.306646.

El análisis incluye importancia de variables, SHAP sobre 2000 viajes
de junio y 41 variables transformadas, explicaciones individuales
y discusión de sesgos y limitaciones.

### Archivos y resultados

- [Notebook ejecutado](notebooks/06_interpretabilidad.ipynb)
- [Código del análisis](src/interpretabilidad.py)
- [Dependencias](requirements.txt)
- [Informe de resultados](reports/semana13/informe_semana13.md)
- [Tablas y metadatos](outputs/semana13/)
- [Figuras](reports/semana13/figures/)

### Ejecución en Google Colab

1. Abrir el notebook mediante el enlace de abajo.
2. Guardar una copia en Google Drive y seleccionar un entorno CPU.
3. Ejecutar las secciones en orden. La sección 2 instala las
   dependencias de requirements.txt.
4. Si Colab solicita reiniciar la sesión tras la instalación,
   reiniciarla y volver a ejecutar desde la sección 1.
5. Autorizar el acceso a Drive en la sección 4.
6. Ejecutar la sección 5 para cargar el modelo guardado o reconstruirlo
   si todavía no existe.
7. Ejecutar la sección 6 para verificar las métricas de junio.
   Si falla la comprobación, revisar la causa antes de continuar.
8. Ejecutar las secciones 7–10 para generar y descargar los análisis,
   las figuras y el informe.

[Abrir en Google Colab](https://colab.research.google.com/github/A-Joacox/proy-machine/blob/main/notebooks/06_interpretabilidad.ipynb)

El notebook descarga el código de la rama main.

### Modelo guardado y recuperación

El modelo y el preprocesador se guardan en el Drive de quien ejecuta
el notebook, dentro de:

`MyDrive/ML_NYC_semana13_desde_semana12/modelo/`

La sección 5 reutiliza el checkpoint cuando coincide con el código
y las versiones registradas. Si no existe, realiza el entrenamiento.
El guardado ocurre al terminar: una interrupción anterior obliga
a repetirlo.

Los resultados se respaldan en:

`MyDrive/ML_NYC_semana13_desde_semana12/resultados/`

Guardar el notebook en Drive y guardar el modelo son operaciones
distintas. El notebook contiene el código y sus salidas; el checkpoint
contiene el modelo y el preprocesador entrenados.

### Metodología

- Entrenamiento: 9,319,633 viajes de enero–mayo de 2025.
- Evaluación: 2,008,054 viajes de junio de 2025.
- Muestreo: 50 % por mes, con semilla 42.
- Modelo: LightGBM con la configuración final y 1710 árboles.
- Preprocesamiento por bloques en float64, conservando el
  cross-fitting de TargetEncoder.
- SHAP: muestra aleatoria de 2000 viajes, semilla 42 y método
  tree_path_dependent.
- Las contribuciones SHAP explican log(1 + duración), no minutos
  directamente, y no demuestran causalidad.
- Las métricas por segmento incluyen tamaño, MAE, error firmado
  y proporción de subestimaciones.

La configuración, las versiones y los hashes se registran en
`outputs/semana13/`. Los datos originales se descargan desde
la fuente oficial mediante el código del proyecto.

Los resultados generados en Colab deben subirse a GitHub por separado.
No se deben incluir los Parquet crudos ni el checkpoint del modelo.
