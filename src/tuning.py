"""Búsqueda de hiperparámetros de LightGBM con validación cruzada temporal.

Semana 11 del plan de trabajo (proposal.md, secciones 10 y 13). Todo ocurre
dentro de entrenamiento (enero-abril): mayo solo compara configuraciones
finales y junio no se toca.

Control de leakage: el preprocesador de src/features.py (distancia histórica
por par de zonas y TargetEncoder) se reajusta en cada fold usando solo la parte
de entrenamiento de ese fold. Se ajusta una vez por fold y se reutiliza para
todas las configuraciones, porque no depende de los hiperparámetros del modelo.
"""

from __future__ import annotations

import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_log_error
from sklearn.model_selection import TimeSeriesSplit

from src.features import construir_preprocesador, preparar_xy

SEMILLA = 42

# Parámetros que no se buscan. El número de árboles lo decide el early stopping.
PARAMS_FIJOS = {
    "subsample_freq": 1,
    "n_jobs": -1,
    "random_state": SEMILLA,
    "verbose": -1,
}

# Configuración del notebook 03 (modelo H), usada como referencia en la búsqueda.
PARAMS_SEMANA_10 = {
    "objective": "regression",
    "learning_rate": 0.1,
    "num_leaves": 127,
    "min_child_samples": 100,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.0,
    "reg_lambda": 0.0,
}


def mae_minutos(y_log_real, y_log_pred):
    """Métrica de early stopping: MAE en minutos (el modelo entrena sobre log1p)."""
    mae = np.mean(np.abs(np.expm1(y_log_pred) - np.expm1(y_log_real)))
    return "mae_min", mae, False


def metricas(y_real, y_pred) -> dict:
    """MAE (principal) y RMSLE (secundaria), igual que en los notebooks 02 y 03."""
    y_pred = np.clip(y_pred, 0, None)
    return {
        "MAE": float(mean_absolute_error(y_real, y_pred)),
        "RMSLE": float(np.sqrt(mean_squared_log_error(y_real, y_pred))),
    }


def preparar_folds(train: pd.DataFrame, n_splits: int = 3) -> list[dict]:
    """Construye los folds de TimeSeriesSplit con las features ya transformadas.

    Ordena train por fecha de recojo, de modo que cada fold entrena con el pasado
    y valida con el bloque de viajes inmediatamente posterior.
    """
    train = train.sort_values("tpep_pickup_datetime").reset_index(drop=True)
    X, y_log = preparar_xy(train, usar_log_target=True)
    y_min = train["trip_duration_min"].to_numpy(dtype=float)
    fechas = train["tpep_pickup_datetime"]

    folds = []
    for i, (idx_tr, idx_va) in enumerate(TimeSeriesSplit(n_splits=n_splits).split(X), start=1):
        prep = construir_preprocesador(escalar_numericas=False)
        Xt_tr = prep.fit_transform(X.iloc[idx_tr], y_log[idx_tr]).astype(np.float32)
        Xt_va = prep.transform(X.iloc[idx_va]).astype(np.float32)
        folds.append(
            {
                "fold": i,
                "Xt_tr": Xt_tr,
                "y_tr_log": y_log[idx_tr],
                "Xt_va": Xt_va,
                "y_va_log": y_log[idx_va],
                "y_va": y_min[idx_va],
                "nombres": prep.named_steps["codificador"].get_feature_names_out(),
                "train_desde": fechas.iloc[idx_tr[0]],
                "train_hasta": fechas.iloc[idx_tr[-1]],
                "val_desde": fechas.iloc[idx_va[0]],
                "val_hasta": fechas.iloc[idx_va[-1]],
            }
        )
    return folds


def resumen_folds(folds: list[dict]) -> pd.DataFrame:
    """Tabla con el tamaño y el rango de fechas de cada fold."""
    return pd.DataFrame(
        [
            {
                "fold": f["fold"],
                "filas_train": len(f["y_tr_log"]),
                "train": f"{f['train_desde']:%d-%b} a {f['train_hasta']:%d-%b}",
                "filas_val": len(f["y_va"]),
                "val": f"{f['val_desde']:%d-%b} a {f['val_hasta']:%d-%b}",
            }
            for f in folds
        ]
    )


def columnas_a_usar(nombres, excluir=()) -> np.ndarray:
    """Índices de las features que se conservan al excluir algunas por nombre."""
    faltan = set(excluir) - set(nombres)
    if faltan:
        raise ValueError(f"Features inexistentes: {sorted(faltan)}")
    return np.array([i for i, n in enumerate(nombres) if n not in set(excluir)])


def muestrear_configuraciones(n: int, seed: int = SEMILLA, learning_rate: float = 0.1) -> list[dict]:
    """Muestrea n configuraciones al azar del espacio de búsqueda.

    El learning rate queda fijo durante la búsqueda: con early stopping solo
    intercambia velocidad por número de árboles, y se baja al final.
    """
    rng = np.random.default_rng(seed)

    def log_uniforme(bajo, alto):
        return float(np.exp(rng.uniform(np.log(bajo), np.log(alto))))

    configs = []
    for _ in range(n):
        configs.append(
            {
                # L2 sobre log1p apunta a la media en log (RMSLE); L1 a la mediana (MAE)
                "objective": str(rng.choice(["regression", "regression_l1"])),
                "learning_rate": learning_rate,
                "num_leaves": int(round(log_uniforme(31, 511))),
                "min_child_samples": int(round(log_uniforme(20, 500))),
                "subsample": round(float(rng.uniform(0.6, 1.0)), 3),
                "colsample_bytree": round(float(rng.uniform(0.5, 1.0)), 3),
                "reg_alpha": round(log_uniforme(1e-3, 10), 4),
                "reg_lambda": round(log_uniforme(1e-3, 10), 4),
            }
        )
    return configs


def evaluar_config(
    params: dict,
    folds: list[dict],
    excluir=(),
    max_arboles: int = 2000,
    paciencia: int = 50,
) -> dict:
    """Evalúa una configuración en todos los folds, con early stopping por MAE en minutos.

    Devuelve el promedio de MAE y RMSLE entre folds, el detalle por fold y el
    número de árboles en el que se detuvo cada uno.
    """
    t0 = time.time()
    maes, rmsles, arboles = [], [], []
    for f in folds:
        cols = columnas_a_usar(f["nombres"], excluir)
        modelo = lgb.LGBMRegressor(n_estimators=max_arboles, metric="None", **PARAMS_FIJOS, **params)
        modelo.fit(
            f["Xt_tr"][:, cols],
            f["y_tr_log"],
            eval_set=[(f["Xt_va"][:, cols], f["y_va_log"])],
            eval_metric=mae_minutos,
            callbacks=[lgb.early_stopping(paciencia, verbose=False)],
        )
        pred = np.expm1(modelo.predict(f["Xt_va"][:, cols]))
        m = metricas(f["y_va"], pred)
        maes.append(m["MAE"])
        rmsles.append(m["RMSLE"])
        arboles.append(int(modelo.best_iteration_))

    return {
        **params,
        "cv_MAE": float(np.mean(maes)),
        "cv_MAE_std": float(np.std(maes)),
        "cv_RMSLE": float(np.mean(rmsles)),
        "MAE_por_fold": [round(v, 4) for v in maes],
        "arboles_por_fold": arboles,
        "segundos": round(time.time() - t0, 1),
    }


def busqueda_aleatoria(configs: list[dict], folds: list[dict], excluir=(), **kwargs) -> pd.DataFrame:
    """Evalúa cada configuración con evaluar_config y devuelve la tabla ordenada por cv_MAE."""
    filas = []
    for i, params in enumerate(configs):
        r = evaluar_config(params, folds, excluir=excluir, **kwargs)
        r["config"] = i
        filas.append(r)
        print(
            f"[{i + 1:>2}/{len(configs)}] cv_MAE = {r['cv_MAE']:.4f} | cv_RMSLE = {r['cv_RMSLE']:.4f} | "
            f"{r['objective']:<14} hojas={r['num_leaves']:<4} árboles={r['arboles_por_fold']} | {r['segundos']:.0f} s"
        )
    return pd.DataFrame(filas).sort_values("cv_MAE").reset_index(drop=True)
