# Predicción de la duración de viajes en taxi — NYC Yellow Taxi

Proyecto final del curso de Machine Learning. Predicción de la duración de un viaje en taxi amarillo de NYC usando únicamente información disponible al momento del recojo.

Ver [`proposal.md`](proposal.md) para la formulación completa del problema.

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
│   └── 01_exploracion_inicial.ipynb
├── src/                   # funciones reutilizables (se poblará en semanas 8+)
├── reports/
│   └── figures/
└── outputs/
```

## Fuente de datos

NYC Taxi & Limousine Commission — Trip Record Data
https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page

Datos públicos, formato Parquet, periodo usado: enero–junio 2025.
