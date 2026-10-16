"""Descarga, carga y limpieza de los datos de NYC Yellow Taxi (ver proposal.md).

Las reglas de limpieza (REGLAS) se aplican en secuencia y quedan registradas en
reports/limpieza_reglas.csv, con el número de filas y el % del total original
que elimina cada una.
"""

from __future__ import annotations

import ssl
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"

URL_TEMPLATE = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{y}-{m:02d}.parquet"

# Meses disponibles para el proyecto: enero a junio de 2025 (ver proposal.md sección 3)
MESES = [(2025, m) for m in range(1, 7)]

# Columnas disponibles al momento del recojo (lista blanca, ver proposal.md sección 7/8).
# trip_distance se incluye solo para la limpieza (velocidad implícita) y para construir
# features de distancia estimada en features.py; nunca se usa como feature directa porque
# es leakage (se mide al terminar el viaje).
COLUMNAS = [
    "VendorID",
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
    "passenger_count",
    "PULocationID",
    "DOLocationID",
    "trip_distance",
]


def _ruta_local(year: int, month: int) -> Path:
    """Ruta local donde se guarda/busca el parquet de un mes."""
    return DATA_DIR / f"yellow_tripdata_{year}-{month:02d}.parquet"


def descargar_mes(year: int, month: int) -> Path:
    """Descarga el parquet de un mes desde la TLC si no existe localmente."""
    ruta = _ruta_local(year, month)
    if ruta.exists():
        return ruta

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    url = URL_TEMPLATE.format(y=year, m=month)
    print(f"Descargando {url}...")
    try:
        urllib.request.urlretrieve(url, ruta)
    except urllib.error.URLError:
        # Algunas instalaciones de Python en macOS no traen certificados raíz; usamos los de certifi
        import certifi

        ctx = ssl.create_default_context(cafile=certifi.where())
        with urllib.request.urlopen(url, context=ctx) as r, open(ruta, "wb") as f:
            f.write(r.read())
    return ruta


def cargar_mes(year: int, month: int) -> pd.DataFrame:
    """Carga un mes ya descargado, calcula trip_duration_min y reduce los tipos de dato."""
    ruta = _ruta_local(year, month)
    df = pd.read_parquet(ruta, columns=COLUMNAS)

    df["trip_duration_min"] = (
        df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]
    ).dt.total_seconds() / 60

    df["VendorID"] = df["VendorID"].astype("int8")
    df["PULocationID"] = df["PULocationID"].astype("int16")
    df["DOLocationID"] = df["DOLocationID"].astype("int16")
    df["passenger_count"] = df["passenger_count"].astype("float32")
    df["trip_distance"] = df["trip_distance"].astype("float32")
    df["trip_duration_min"] = df["trip_duration_min"].astype("float32")

    return df


def _fuera_de_mes(df: pd.DataFrame, year: int, month: int) -> pd.Series:
    """Recojos fuera del mes que dice el nombre del archivo (registros rezagados)."""
    inicio = pd.Timestamp(year=year, month=month, day=1)
    fin = inicio + pd.DateOffset(months=1)
    return (df["tpep_pickup_datetime"] < inicio) | (df["tpep_pickup_datetime"] >= fin)


def _duracion_invalida(df: pd.DataFrame, year: int, month: int) -> pd.Series:
    """Duración fuera de [1, 180] minutos (taxímetros mal cerrados, ver notebook 01)."""
    return (df["trip_duration_min"] < 1) | (df["trip_duration_min"] > 180)


def _zona_desconocida(df: pd.DataFrame, year: int, month: int) -> pd.Series:
    """Recojo o destino en una zona desconocida (264/265)."""
    return df["PULocationID"].isin([264, 265]) | df["DOLocationID"].isin([264, 265])


def _distancia_cero(df: pd.DataFrame, year: int, month: int) -> pd.Series:
    """Viajes sin distancia recorrida (cancelaciones o error de registro)."""
    return df["trip_distance"] <= 0


def _velocidad_imposible(df: pd.DataFrame, year: int, month: int) -> pd.Series:
    """Velocidad implícita (distancia/tiempo) mayor a 80 mph.

    Se aplica después de duracion_invalida y distancia_cero, por lo que para estas
    filas la duración ya es >= 1 min y la distancia > 0 (sin división por cero).
    """
    horas = df["trip_duration_min"] / 60
    velocidad_mph = df["trip_distance"] / horas
    return velocidad_mph > 80


def _reversion_pago(df: pd.DataFrame, year: int, month: int) -> pd.Series:
    """Filas exactamente duplicadas en COLUMNAS.

    Verificado en enero 2025: ~98.7% de estos pares no son errores de carga sino
    pares cargo/anulación de la TLC (misma fila de viaje repetida con total_amount
    de signo opuesto, columna que no está en COLUMNAS). Ambas filas describen el
    mismo viaje, así que se conserva solo una copia para no contarlo dos veces
    en el entrenamiento.
    """
    return df.duplicated(keep="first")


# Reglas de limpieza, aplicadas en secuencia. No se filtra passenger_count: se
# tratará como categoría (con valor "desconocido" para NaN) en features.py.
REGLAS = [
    ("fuera_de_mes", _fuera_de_mes),
    ("duracion_invalida", _duracion_invalida),
    ("zona_desconocida", _zona_desconocida),
    ("distancia_cero", _distancia_cero),
    ("velocidad_imposible", _velocidad_imposible),
    ("reversion_pago", _reversion_pago),
]


def limpiar_mes(df: pd.DataFrame, year: int, month: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aplica REGLAS en secuencia sobre un mes.

    Devuelve (df_limpio, log), donde log tiene una fila por regla con el mes,
    las filas eliminadas y el % que representan sobre el total original del mes.
    """
    n_original = len(df)
    actual = df
    registros = []

    for nombre, regla in REGLAS:
        mask = regla(actual, year, month)
        n_eliminadas = int(mask.sum())
        registros.append(
            {
                "mes": f"{year}-{month:02d}",
                "regla": nombre,
                "filas_eliminadas": n_eliminadas,
                "pct_del_total_original": round(n_eliminadas / n_original * 100, 4) if n_original else 0.0,
            }
        )
        actual = actual[~mask]

    log = pd.DataFrame(registros)
    return actual, log


def split_temporal(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split temporal (ver proposal.md sección 10): train ene-abr, val may, test jun."""
    mes_num = df["tpep_pickup_datetime"].dt.month
    train = df[mes_num.between(1, 4)]
    val = df[mes_num == 5]
    test = df[mes_num == 6]
    return train, val, test


def cargar_todo(meses=MESES, sample_frac: float | None = None, seed: int = 42) -> pd.DataFrame:
    """Descarga, carga y limpia cada mes; concatena el resultado y guarda el log de limpieza.

    Si sample_frac no es None, se muestrea esa fracción de cada mes por separado
    (con random_state=seed) antes de concatenar.
    """
    dfs = []
    logs = []

    for year, month in meses:
        descargar_mes(year, month)
        df_mes = cargar_mes(year, month)
        df_limpio, log = limpiar_mes(df_mes, year, month)

        if sample_frac is not None:
            df_limpio = df_limpio.sample(frac=sample_frac, random_state=seed)

        df_limpio = df_limpio.copy()
        df_limpio["mes"] = f"{year}-{month:02d}"

        dfs.append(df_limpio)
        logs.append(log)

    df_total = pd.concat(dfs, ignore_index=True)
    log_total = pd.concat(logs, ignore_index=True)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    log_total.to_csv(REPORTS_DIR / "limpieza_reglas.csv", index=False)

    return df_total


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Descarga, limpia y guarda los datos de NYC Yellow Taxi.")
    parser.add_argument(
        "--meses",
        type=int,
        nargs="+",
        default=[m for _, m in MESES],
        help="Meses de 2025 a procesar, p. ej. --meses 1 2 3 (default: 1 a 6)",
    )
    parser.add_argument("--sample-frac", type=float, default=None, help="Fracción de muestreo por mes")
    parser.add_argument("--seed", type=int, default=42, help="Semilla para el muestreo")
    args = parser.parse_args()

    meses_elegidos = [(2025, m) for m in args.meses]
    df = cargar_todo(meses=meses_elegidos, sample_frac=args.sample_frac, seed=args.seed)

    salida = ROOT / "data" / "processed" / "trips_clean.parquet"
    salida.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(salida, index=False)
    print(f"Guardado {len(df):,} filas en {salida}")
