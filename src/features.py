"""
Ingenieria de features y pipeline de preprocesamiento sin data leakage.

Implementa lo acordado en proposal.md (secciones 7, 8 y 12) y da continuidad
a src/data.py:
  1. Estimacion de distancia historica entre pares (PULocationID, DOLocationID)
     ajustada UNICAMENTE sobre el conjunto de entrenamiento (Train) antes de
     descartar la columna `trip_distance` (evita data leakage).
  2. Enriquecimiento geografico con `data/Taxi_Zone_Lookup.csv` (Borough,
     service_zone, viajes de aeropuerto e intra-borough).
  3. Extraccion de variables temporales (hora, dia de semana, fin de semana,
     hora punta, codificacion ciclica sen/cos y feriados de NYC 2025).
  4. Tratamiento explicito de `passenger_count` con categoria "desconocido"
     para valores faltantes (NaN).
  5. Target Encoding de zonas de alta cardinalidad ajustado solo en Train
     mediante pipelines de scikit-learn.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder

RUTA_ZONAS = Path("data/Taxi_Zone_Lookup.csv")
SEMILLA = 42

# Feriados federales en NYC durante el primer semestre de 2025 (ene - jun)
FERIADOS_2025 = {
    pd.Timestamp("2025-01-01").date(),  # New Year's Day
    pd.Timestamp("2025-01-20").date(),  # Martin Luther King Jr. Day
    pd.Timestamp("2025-02-17").date(),  # Presidents' Day
    pd.Timestamp("2025-05-26").date(),  # Memorial Day
    pd.Timestamp("2025-06-19").date(),  # Juneteenth
}

# IDs oficiales de aeropuertos en Taxi_Zone_Lookup.csv
ZONAS_AEROPUERTO = {1, 132, 138}  # 1: Newark, 132: JFK, 138: LaGuardia

# Lista blanca de columnas crudas de entrada permitidas antes de transformar
# Nota: trip_distance entra al fit() de Train solo para calcular medianas por
# ruta y se elimina inmediatamente en transform() para garantizar cero leakage.
FEATURES_ENTRADA = [
    "tpep_pickup_datetime",
    "PULocationID",
    "DOLocationID",
    "passenger_count",
    "VendorID",
]


def categorizar_passenger_count(serie: pd.Series) -> pd.Series:
    """
    Convierte passenger_count en variable categorica tratando los NaN
    explicitamente como 'desconocido' (segun lo documentado en src/data.py).
    Agrupa conteos >= 5 en '5+' para evitar categorias raras.
    """
    s = serie.copy()
    resultado = pd.Series("desconocido", index=s.index, dtype="object")
    mask_valido = s.notna()
    valores = s[mask_valido].astype(int)

    resultado.loc[mask_valido & (valores <= 0)] = "0"
    resultado.loc[mask_valido & (valores == 1)] = "1"
    resultado.loc[mask_valido & (valores == 2)] = "2"
    resultado.loc[mask_valido & (valores == 3)] = "3"
    resultado.loc[mask_valido & (valores == 4)] = "4"
    resultado.loc[mask_valido & (valores >= 5)] = "5+"
    return resultado


class ExtractorFeaturesNYC(BaseEstimator, TransformerMixin):
    """
    Transformador compatible con scikit-learn que:
      - En `.fit(X, y)`: aprende la mediana historica de `trip_distance` por
        par (PULocationID, DOLocationID) y por PULocationID usando SOLO Train.
      - En `.transform(X)`: construye features temporales, geograficas y de
        distancia estimada, descartando `trip_distance` y cualquier variable
        post-viaje.
    """

    def __init__(self, ruta_zonas: Path = RUTA_ZONAS):
        self.ruta_zonas = ruta_zonas

    def fit(self, X: pd.DataFrame, y=None):
        # 1. Cargar catalogo de zonas TLC
        zonas = pd.read_csv(self.ruta_zonas)
        self.mapa_borough_ = zonas.set_index("LocationID")["Borough"].to_dict()
        self.mapa_service_ = zonas.set_index("LocationID")["service_zone"].to_dict()

        # 2. Aprender distancia estimada sin leakage (solo con datos de Train)
        if "trip_distance" in X.columns:
             pares = (
                X.groupby(["PULocationID", "DOLocationID"])["trip_distance"]
                .median()
                .to_dict()
            )
            por_origen = (
                X.groupby("PULocationID")["trip_distance"].median().to_dict()
            )
            mediana_global = float(X["trip_distance"].median())
        else:
            pares = {}
            por_origen = {}
            mediana_global = 1.6  # valor de respaldo razonable en millas

        self.dist_pares_ = pares
        self.dist_origen_ = por_origen
        self.dist_global_ = mediana_global
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = pd.DataFrame(index=X.index)

        # --- 1. Variables Temporales ---
        pickup = pd.to_datetime(X["tpep_pickup_datetime"])
        hora = pickup.dt.hour
        dia_semana = pickup.dt.dayofweek

        df["pickup_hour"] = hora
        df["pickup_dayofweek"] = dia_semana
        df["pickup_month"] = pickup.dt.month
        df["es_fin_de_semana"] = (dia_semana >= 5).astype(int)
        df["es_feriado"] = pickup.dt.date.isin(FERIADOS_2025).astype(int)

        # Hora punta en dia laboral (7-10 am o 4-8 pm de lunes a viernes)
        es_laboral = (dia_semana < 5) & (df["es_feriado"] == 0)
        df["es_hora_punta"] = (
            es_laboral & ((hora.between(7, 9)) | (hora.between(16, 19)))
        ).astype(int)

        # Codificacion ciclica para capturar continuidad (ej. 23h cerca de 0h)
        df["hora_sin"] = np.sin(2 * np.pi * hora / 24.0)
        df["hora_cos"] = np.cos(2 * np.pi * hora / 24.0)
        df["dia_sin"] = np.sin(2 * np.pi * dia_semana / 7.0)
        df["dia_cos"] = np.cos(2 * np.pi * dia_semana / 7.0)

        # --- 2. Variables Geograficas y Zonas TLC ---
        pu = X["PULocationID"].astype(int)
        do = X["DOLocationID"].astype(int)

        df["PULocationID"] = pu
        df["DOLocationID"] = do
        df["misma_zona"] = (pu == do).astype(int)

        pu_borough = pu.map(self.mapa_borough_).fillna("Desconocido")
        do_borough = do.map(self.mapa_borough_).fillna("Desconocido")
        df["pu_borough"] = pu_borough
        df["do_borough"] = do_borough
        df["mismo_borough"] = (pu_borough == do_borough).astype(int)

        # Indicadores de aeropuerto (JFK=132, LaGuardia=138, Newark=1)
        df["es_aeropuerto_origen"] = pu.isin(ZONAS_AEROPUERTO).astype(int)
        df["es_aeropuerto_destino"] = do.isin(ZONAS_AEROPUERTO).astype(int)
        df["viaje_aeropuerto"] = (
            (df["es_aeropuerto_origen"] == 1) | (df["es_aeropuerto_destino"] == 1)
        ).astype(int)

        # --- 3. Distancia Estimada Historica (Cero Leakage) ---
        # Busca mediana del par (PU, DO) aprendida en Train; si el par es nuevo,
        # usa la mediana del origen PU; si no existe, usa la mediana global.
        indice_pares = pd.MultiIndex.from_arrays([pu, do])
        serie_pares = pd.Series(indice_pares.map(self.dist_pares_), index=X.index)
        serie_origen = pu.map(self.dist_origen_)
        df["distancia_estimada_mi"] = (
            serie_pares.fillna(serie_origen).fillna(self.dist_global_).astype(float)
        )
        df["log_distancia_estimada"] = np.log1p(
            np.clip(df["distancia_estimada_mi"], a_min=0.0, a_max=None)
        )

        # --- 4. Pasajeros y Proveedor ---
        df["passenger_cat"] = categorizar_passenger_count(X["passenger_count"])
        df["VendorID"] = X["VendorID"].astype(str)

        return df


# Listas de columnas resultantes de ExtractorFeaturesNYC por tipo de tratamiento
COLS_NUMERICAS = [
    "pickup_hour",
    "pickup_dayofweek",
    "pickup_month",
    "es_fin_de_semana",
    "es_feriado",
    "es_hora_punta",
    "hora_sin",
    "hora_cos",
    "dia_sin",
    "dia_cos",
    "misma_zona",
    "mismo_borough",
    "es_aeropuerto_origen",
    "es_aeropuerto_destino",
    "viaje_aeropuerto",
    "distancia_estimada_mi",
    "log_distancia_estimada",
]

COLS_ALTA_CARDINALIDAD = ["PULocationID", "DOLocationID"]

COLS_CATEGORICAS_BAJA = [
    "pu_borough",
    "do_borough",
    "passenger_cat",
    "VendorID",
]


def construir_preprocesador(escalar_numericas: bool = True) -> Pipeline:
    """
    Construye el Pipeline completo de ingenieria de features y codificacion
    listo para conectarse con cualquier estimador de scikit-learn.

    - Aplica `TargetEncoder` con validacion cruzada interna sobre PULocationID
      y DOLocationID (evitando sobreajuste y leakage).
    - Aplica `OneHotEncoder` a variables categoricas de baja cardinalidad.
    - Escala las variables numericas si `escalar_numericas=True` (util para
      modelos lineales como Ridge/Lasso; opcional para arboles).
    """
    transformador_num = StandardScaler() if escalar_numericas else "passthrough"

    codificador_columnas = ColumnTransformer(
        transformers=[
            ("num", transformador_num, COLS_NUMERICAS),
            (
                "target_enc",
                TargetEncoder(
                    target_type="continuous",
                    smooth="auto",
                    cv=5,
                    random_state=SEMILLA,
                ),
                COLS_ALTA_CARDINALIDAD,
            ),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                COLS_CATEGORICAS_BAJA,
            ),
        ],
        remainder="drop",
    )

    pipeline_features = Pipeline(
        steps=[
            ("extractor", ExtractorFeaturesNYC(ruta_zonas=RUTA_ZONAS)),
            ("codificador", codificador_columnas),
        ]
    )
    return pipeline_features


def preparar_xy(df: pd.DataFrame, target_col: str = "duration_min", usar_log_target: bool = False):
    """
    Separa el DataFrame limpio en matriz de entrada X y vector objetivo y.
    Conserva `trip_distance` dentro de X unicamente para que `ExtractorFeaturesNYC.fit()`
    calcule las distancias historicas en Train; `transform()` nunca expone `trip_distance`.
    """
    cols_disponibles = [c for c in FEATURES_ENTRADA + ["trip_distance"] if c in df.columns]
    X = df[cols_disponibles].copy()
    y = df[target_col].to_numpy(dtype=float)
    if usar_log_target:
        y = np.log1p(y)
    return X, y
