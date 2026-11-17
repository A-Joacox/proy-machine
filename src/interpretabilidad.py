"""Semana 13: reconstrucción e interpretación desde la configuración de semana 12.

Solo utiliza datos TLC y transformadores existentes del proyecto. No carga resultados
del notebook 06 eliminado. Las matrices de entrenamiento se escriben por bloques a
disco en float64; TargetEncoder mantiene su fit_transform global con cross-fitting.
"""
from pathlib import Path
import gc
import hashlib
import importlib.metadata
import json
import platform
import time

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_log_error

from src import features, data


def guardar_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def versiones():
    names = ["numpy", "pandas", "scikit-learn", "lightgbm", "pyarrow", "shap", "joblib"]
    return {"python": platform.python_version(), **{n: importlib.metadata.version(n) for n in names}}


class PreprocesadorBloques:
    """Mismas features, codificaciones y orden que construir_preprocesador(False).

    OneHotEncoder aprende de combinaciones categóricas reales distintas, ya que no
    utiliza frecuencias. TargetEncoder sí usa TODAS las filas y el target original.
    No se ajusta un codificador independiente en cada bloque.
    """

    def __init__(self, ruta_zonas):
        self.ruta_zonas = str(ruta_zonas)

    def fit_transform_disk(self, X, y, path, block_size=100_000):
        original = features.construir_preprocesador(escalar_numericas=False)
        self.extractor = clone(original.named_steps["extractor"])
        self.extractor.set_params(ruta_zonas=Path(self.ruta_zonas))
        self.extractor.fit(X, y)
        transformers = original.named_steps["codificador"].transformers
        assert [t[0] for t in transformers] == ["num", "target_enc", "onehot"]
        assert transformers[0][1] == "passthrough"
        self.num_cols = list(transformers[0][2])
        self.te_cols = list(transformers[1][2])
        self.cat_cols = list(transformers[2][2])
        self.target_encoder = clone(transformers[1][1])
        self.onehot = clone(transformers[2][1])

        # fit_transform conserva el cross-fitting del pipeline original.
        encoded_oof = self.target_encoder.fit_transform(X[self.te_cols].astype(int), y)
        observed_categories = []
        for start in range(0, len(X), block_size):
            engineered = self.extractor.transform(X.iloc[start:start + block_size])
            observed_categories.append(engineered[self.cat_cols].drop_duplicates())
        del engineered
        actual_combinations = pd.concat(observed_categories, ignore_index=True).drop_duplicates()
        self.onehot.fit(actual_combinations)
        del observed_categories, actual_combinations
        self.feature_names = (
            ["num__" + c for c in self.num_cols]
            + ["target_enc__" + c for c in self.target_encoder.get_feature_names_out()]
            + ["onehot__" + c for c in self.onehot.get_feature_names_out()]
        )
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        matrix = np.lib.format.open_memmap(
            path, mode="w+", dtype=np.float64, shape=(len(X), len(self.feature_names))
        )
        for start in range(0, len(X), block_size):
            end = min(start + block_size, len(X))
            engineered = self.extractor.transform(X.iloc[start:end])
            matrix[start:end] = np.column_stack([
                engineered[self.num_cols].to_numpy(dtype=np.float64),
                encoded_oof[start:end],
                self.onehot.transform(engineered[self.cat_cols]),
            ])
            if start == 0 or end == len(X) or end % 1_000_000 == 0:
                print(f"Preprocesamiento: {end:,}/{len(X):,}", flush=True)
        matrix.flush()
        del encoded_oof
        gc.collect()
        return matrix

    def transform(self, X):
        engineered = self.extractor.transform(X)
        return np.column_stack([
            engineered[self.num_cols].to_numpy(dtype=np.float64),
            self.target_encoder.transform(engineered[self.te_cols]),
            self.onehot.transform(engineered[self.cat_cols]),
        ])


def verificar_equivalencia(X, y, ruta_zonas, work_dir):
    """Compara ambas implementaciones sobre las mismas 3,000 filas REALES de train."""
    positions = np.random.default_rng(42).choice(len(X), min(3000, len(X)), replace=False)
    xs, ys = X.iloc[positions].reset_index(drop=True), y[positions]
    original = features.construir_preprocesador(escalar_numericas=False)
    original.set_params(extractor__ruta_zonas=Path(ruta_zonas))
    expected = original.fit_transform(xs, ys)
    disk_path = Path(work_dir) / "equivalencia.npy"
    optimized = PreprocesadorBloques(ruta_zonas)
    actual = optimized.fit_transform_disk(xs, ys, disk_path, block_size=731)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    assert optimized.feature_names == list(original.named_steps["codificador"].get_feature_names_out())
    # Compara también la transformación de inferencia, distinta al cross-fitting.
    np.testing.assert_allclose(optimized.transform(xs), original.transform(xs), rtol=1e-12, atol=1e-12)
    del actual
    disk_path.unlink()
    print("Equivalencia verificada contra el pipeline original en datos reales.")


def cargar_muestra_mes(month, config):
    path = data.descargar_mes(2025, month)
    raw = data.cargar_mes(2025, month)
    cleaned, log = data.limpiar_mes(raw, 2025, month)
    del raw
    sample = cleaned.sample(frac=config["sample_frac"], random_state=config["seed"]).copy()
    sample["fila_archivo"] = sample.index
    del cleaned
    return sample.reset_index(drop=True), log, {"archivo": path.name, "sha256": sha256(path)}


def reconstruir_modelo(root, checkpoint_dir, work_dir, config, expected_n_train):
    root, checkpoint_dir, work_dir = map(Path, [root, checkpoint_dir, work_dir])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    code_paths = ["src/data.py", "src/features.py", "src/interpretabilidad.py", "outputs/config_modelo_final.json"]
    provenance = {"config": config, "versiones": versiones(), "codigo_sha256": {p: sha256(root / p) for p in code_paths}}
    meta_path = checkpoint_dir / "modelo_metadata.json"
    model_path = checkpoint_dir / "modelo_preprocesador.joblib"
    if model_path.exists() or meta_path.exists():
        if not (model_path.exists() and meta_path.exists()):
            raise RuntimeError("Checkpoint incompleto. Conserva esta carpeta y usa una nueva para reconstruir.")
        saved = json.loads(meta_path.read_text())
        for key, value in provenance.items():
            if saved[key] != value:
                raise RuntimeError(f"El checkpoint no coincide en {key}. No se sobrescribió: revisa las versiones/código.")
        if sha256(model_path) != saved["modelo_sha256"]:
            raise RuntimeError("El archivo del modelo no coincide con su hash guardado.")
        payload = joblib.load(model_path)
        print("Modelo y preprocesador recuperados de Drive; no se reentrena.")
        return payload["modelo"], payload["preprocesador"], saved

    if config["features_excluidas"] or config["target"] != "log1p(trip_duration_min)":
        raise ValueError("La configuración cambió respecto a la semana 12 revisada.")
    pieces, targets, logs, files = [], [], [], []
    counts = {}
    for month in range(1, 6):
        sample, log, file_info = cargar_muestra_mes(month, config)
        X, y = features.preparar_xy(sample, usar_log_target=True)
        pieces.append(X)
        targets.append(y)
        counts[str(month)] = len(sample)
        logs.append(log)
        files.append(file_info)
        print(f"2025-{month:02d}: {len(sample):,} viajes de entrenamiento.", flush=True)
        del sample, X, y
        gc.collect()
    X = pd.concat(pieces, ignore_index=True)
    y = np.concatenate(targets)
    del pieces, targets
    gc.collect()
    if len(X) != expected_n_train:
        raise RuntimeError(f"Filas train {len(X):,}; semana 12 {expected_n_train:,}. Revisar datos antes de entrenar.")
    verificar_equivalencia(X, y, root / "data/Taxi_Zone_Lookup.csv", work_dir)
    prep = PreprocesadorBloques(root / "data/Taxi_Zone_Lookup.csv")
    matrix_path = work_dir / "X_train_float64.npy"
    matrix = prep.fit_transform_disk(X, y, matrix_path)
    del X
    gc.collect()
    model = lgb.LGBMRegressor(**config["params"], **config["params_fijos"], n_estimators=config["n_estimators"])
    start = time.monotonic()

    def progress(env):
        if (env.iteration + 1) % 100 == 0:
            print(f"Árbol {env.iteration + 1}/{config['n_estimators']} | {(time.monotonic()-start)/60:.1f} min", flush=True)

    print("Entrenamiento con enero-mayo; junio no participa.", flush=True)
    model.fit(matrix, y, callbacks=[progress])
    del matrix, y
    gc.collect()
    matrix_path.unlink()
    tmp = checkpoint_dir / "modelo_preprocesador.joblib.tmp"
    joblib.dump({"modelo": model, "preprocesador": prep}, tmp, compress=3)
    tmp.replace(model_path)
    metadata = {
        **provenance, "n_train": sum(counts.values()), "filas_por_mes": counts,
        "archivos_train": files, "dtype_entrenamiento": "float64", "feature_names": prep.feature_names,
        "modelo_sha256": sha256(model_path), "segundos_entrenamiento": time.monotonic() - start,
        "nota": "Reconstrucción; versiones y hashes de la ejecución original de semana 12 no estaban archivados.",
    }
    guardar_json(meta_path, metadata)
    pd.concat(logs, ignore_index=True).to_csv(checkpoint_dir / "limpieza_train.csv", index=False)
    print("Modelo y preprocesador guardados. Ya se puede retomar sin entrenar de nuevo.")
    return model, prep, metadata


def evaluar_junio(model, prep, config, original_metrics, output_dir, block_size=50_000):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    test, cleaning, file_info = cargar_muestra_mes(6, config)
    if len(test) != original_metrics["n_test"]:
        raise RuntimeError("El número de viajes de junio difiere de semana 12. Revisar antes de continuar.")
    predictions = np.empty(len(test), dtype=np.float64)
    for start in range(0, len(test), block_size):
        end = min(start + block_size, len(test))
        predictions[start:end] = np.clip(np.expm1(model.predict(prep.transform(test.iloc[start:end]))), 0, None)
    test["pred_min"] = predictions
    test["error_firmado"] = test["pred_min"] - test["trip_duration_min"]
    test["error_abs"] = test["error_firmado"].abs()
    current = {
        "MAE": float(mean_absolute_error(test["trip_duration_min"], predictions)),
        "RMSLE": float(np.sqrt(mean_squared_log_error(test["trip_duration_min"], predictions))),
    }
    # Umbrales de control de reproducción, fijados antes de evaluar; no son IC.
    tolerances = {"MAE": 0.02, "RMSLE": 0.002}
    diffs = {m: current[m] - original_metrics[m] for m in current}
    check = {"reconstruccion": current, "semana12": original_metrics, "diferencias": diffs,
             "tolerancias_reproduccion": tolerances, "archivo_test": file_info,
             "compatible": all(abs(diffs[m]) <= tolerances[m] for m in current)}
    guardar_json(output_dir / "verificacion_modelo.json", check)
    cleaning.to_csv(output_dir / "limpieza_junio.csv", index=False)
    print(json.dumps(check, indent=2, ensure_ascii=False))
    if not check["compatible"]:
        raise RuntimeError("Discrepancia de reproducción: detener SHAP y revisar; no ajustar hiperparámetros con junio.")
    return test, check


def analisis_segmentos(test, prep, output_dir):
    frame = pd.DataFrame({
        "hora": test["tpep_pickup_datetime"].dt.hour,
        "borough_origen": test["PULocationID"].map(prep.extractor.mapa_borough_).fillna("Desconocido"),
        "aeropuerto": (test["PULocationID"].isin([1,132,138]) | test["DOLocationID"].isin([1,132,138])),
        "pasajeros_faltantes": test["passenger_count"].isna(),
        "distancia_real_tramo": pd.cut(test["trip_distance"], [0,1,3,6,10,20,np.inf],
                                      labels=["0-1", "1-3", "3-6", "6-10", "10-20", "20+"], include_lowest=True),
        "error_abs": test["error_abs"], "error_firmado": test["error_firmado"],
        "subestimado": test["error_firmado"] < 0,
    })
    tables = []
    for segment in ["hora", "borough_origen", "aeropuerto", "pasajeros_faltantes", "distancia_real_tramo"]:
        t = frame.groupby(segment, observed=True, dropna=False).agg(
            n=("error_abs", "size"), MAE=("error_abs", "mean"),
            error_medio_firmado=("error_firmado", "mean"), fraccion_subestimada=("subestimado", "mean")
        ).reset_index().rename(columns={segment: "grupo"})
        t["segmento"] = segment
        t["grupo"] = t["grupo"].astype(str)
        t["porcentaje_test"] = 100 * t["n"] / len(test)
        tables.append(t)
    result = pd.concat(tables, ignore_index=True)
    result.to_csv(Path(output_dir) / "errores_segmentos.csv", index=False)
    return result


def interpretar(model, prep, test, output_dir, figures_dir, n_shap=2000, seed=42):
    import matplotlib.pyplot as plt
    import shap
    output_dir, figures_dir = Path(output_dir), Path(figures_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    if len(test) < n_shap:
        raise ValueError("Junio no contiene el tamaño de muestra SHAP predefinido.")
    positions = np.sort(np.random.default_rng(seed).choice(len(test), n_shap, replace=False))
    sample = test.iloc[positions].copy().reset_index(drop=True)
    matrix = prep.transform(sample)
    names = prep.feature_names
    explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent", model_output="raw")
    blocks = []
    for start in range(0, n_shap, 100):
        blocks.append(explainer.shap_values(matrix[start:start+100], check_additivity=True))
        print(f"SHAP: {min(start+100, n_shap)}/{n_shap}", flush=True)
    values = np.concatenate(blocks)
    base = float(np.asarray(explainer.expected_value).reshape(-1)[0])
    raw_pred = model.predict(matrix)
    reconstruction = base + values.sum(axis=1)
    np.testing.assert_allclose(reconstruction, raw_pred, rtol=1e-5, atol=1e-6)
    np.savez_compressed(output_dir / "shap_valores.npz", values=values, base=base, X=matrix,
                        posiciones_test=positions, feature_names=np.asarray(names))
    sample["posicion_test"] = positions
    sample.to_csv(output_dir / "muestra_shap.csv", index=False)
    gain = model.booster_.feature_importance("gain")
    splits = model.booster_.feature_importance("split")
    table = pd.DataFrame({"feature": names, "gain": gain, "split": splits,
                          "mean_abs_shap_log1p": np.abs(values).mean(axis=0)})
    table["gain_pct"] = 100 * gain / gain.sum() if gain.sum() else 0
    table.to_csv(output_dir / "importancias.csv", index=False)
    for metric, filename, xlabel in [
        ("gain_pct", "01_importancia_gain.png", "Porcentaje de ganancia de entrenamiento"),
        ("mean_abs_shap_log1p", "02_importancia_shap.png", "Media de |SHAP| en escala log1p(minutos)"),
    ]:
        top = table.nlargest(15, metric).sort_values(metric)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(top.feature, top[metric], color="#276b85")
        ax.set_xlabel(xlabel)
        fig.tight_layout()
        fig.savefig(figures_dir / filename, dpi=150, bbox_inches="tight")
        plt.close(fig)
    shap.summary_plot(values, matrix, feature_names=names, max_display=15, show=False)
    plt.title("SHAP: muestra aleatoria de junio (escala log1p)")
    plt.savefig(figures_dir / "03_shap_distribucion.png", dpi=150, bbox_inches="tight")
    plt.close("all")
    for feature, color_feature, filename in [
        ("num__distancia_estimada_mi", "num__pickup_hour", "04_dependencia_distancia.png"),
        ("num__pickup_hour", "num__es_fin_de_semana", "05_dependencia_hora.png"),
    ]:
        shap.dependence_plot(feature, values, matrix, feature_names=names, interaction_index=color_feature, show=False)
        plt.savefig(figures_dir / filename, dpi=150, bbox_inches="tight")
        plt.close("all")
    # Casos elegidos por error, no se presentan como una muestra representativa.
    order = np.argsort(sample["error_abs"].to_numpy(), kind="stable")
    cases = [("menor_error", int(order[0])), ("error_mediano", int(order[len(order)//2])), ("mayor_error", int(order[-1]))]
    local_rows = []
    for label, pos in cases:
        explanation = shap.Explanation(values=values[pos], base_values=base, data=matrix[pos], feature_names=names)
        shap.plots.waterfall(explanation, max_display=12, show=False)
        row = sample.iloc[pos]
        plt.title(f"{label}: real {row.trip_duration_min:.2f} min; predicción {row.pred_min:.2f} min\nContribuciones en log1p, no en minutos")
        plt.savefig(figures_dir / f"06_caso_{label}.png", dpi=150, bbox_inches="tight")
        plt.close("all")
        local_rows.append({"criterio": label, "posicion_test": int(positions[pos]), "fila_archivo": int(row.fila_archivo),
                           "real_min": float(row.trip_duration_min), "pred_min": float(row.pred_min),
                           "error_firmado": float(row.error_firmado)})
    pd.DataFrame(local_rows).to_csv(output_dir / "casos_locales.csv", index=False)
    metadata = {"n_shap": n_shap, "seed": seed, "seleccion": "Aleatoria simple sin reemplazo sobre el test junio al 50%",
                "escala": "log1p(trip_duration_min)", "feature_perturbation": "tree_path_dependent",
                "base_log1p": base, "base_transformada_min": float(np.expm1(base)),
                "nota_base": "La base transformada no es la duración media en minutos.",
                "error_max_aditividad": float(np.max(np.abs(reconstruction-raw_pred))),
                "casos": local_rows, "versiones": versiones()}
    guardar_json(output_dir / "interpretabilidad_metadata.json", metadata)
    return table, metadata


def escribir_informe(path, check, table, shap_meta, segments, model_meta):
    """Informe descriptivo calculado con esta ejecución, sin cifras heredadas."""
    def markdown(frame):
        header = "| " + " | ".join(map(str, frame.columns)) + " |"
        rule = "| " + " | ".join(["---"] * len(frame.columns)) + " |"
        rows = ["| " + " | ".join(str(v).replace("|", "/") for v in row) + " |"
                for row in frame.itertuples(index=False, name=None)]
        return "\n".join([header, rule, *rows])
    ranking = table.nlargest(15, "mean_abs_shap_log1p")[["feature", "mean_abs_shap_log1p", "gain_pct"]].round(6)
    geography = segments[segments.segmento == "borough_origen"].drop(columns="segmento").round(4)
    airports = segments[segments.segmento == "aeropuerto"].drop(columns="segmento").round(4)
    missing = segments[segments.segmento == "pasajeros_faltantes"].drop(columns="segmento").round(4)
    cases = pd.DataFrame(shap_meta["casos"]).round(4)
    text = f"""# Semana 13: interpretabilidad, sesgos y limitaciones

## Alcance y procedencia

Análisis nuevo desde la configuración de semana 12. No se leen ni reutilizan las
figuras o estadísticas del notebook 06 eliminado. Se usan registros reales de NYC
Yellow Taxi (enero-junio de 2025), la limpieza de `src/data.py` y las variables de
`src/features.py`. Entrenamiento: {model_meta['n_train']:,} viajes de enero-mayo;
test: {check['semana12']['n_test']:,} de junio. Ambos conservan el muestreo mensual
del 50 % y semilla 42. La configuración de LightGBM conserva sus 1710 árboles.

El preprocesamiento por bloques conserva float64 y el cross-fitting de
TargetEncoder. Se compara con el pipeline original sobre 3000 filas reales antes
del entrenamiento completo. Esta comprobación técnica no es una evaluación predictiva.

## Control de reproducción

- MAE reconstruido: {check['reconstruccion']['MAE']:.6f} minutos.
- RMSLE reconstruido: {check['reconstruccion']['RMSLE']:.6f}.
- Diferencia MAE frente a semana 12: {check['diferencias']['MAE']:+.6f} minutos.
- Diferencia RMSLE: {check['diferencias']['RMSLE']:+.6f}.

Los límites de control se fijaron antes de evaluar: 0.02 minutos de MAE y 0.002
de RMSLE. No son intervalos de confianza ni prueban identidad entre los modelos.
No se archivaron todas las versiones ni hashes de los datos de la ejecución de
semana 12; esta ejecución sí los registra. Junio se usa para verificar y describir
el modelo congelado, nunca para seleccionar hiperparámetros o variables nuevas.

## Importancia global

SHAP se calcula en {shap_meta['n_shap']:,} viajes seleccionados al azar, sin reemplazo,
del test de junio, con semilla {shap_meta['seed']}. El entrenamiento usa todas las
filas indicadas arriba. La muestra SHAP solo reduce el costo de explicar predicciones;
no garantiza cobertura suficiente de rutas raras.

{markdown(ranking)}

Gain es la importancia acumulada en los splits del entrenamiento; no representa un
porcentaje de mejora del MAE de test. `mean_abs_shap_log1p` mide magnitud de contribución
al modelo en escala log1p. Ninguna de las dos medidas demuestra causalidad.

El método es TreeSHAP `tree_path_dependent`, con salida `raw`. Las contribuciones
explican log(1 + duración), no minutos. Se verifica base + suma(SHAP) = salida del
modelo; error máximo observado: {shap_meta['error_max_aditividad']:.3g}.
La transformación inversa se aplica a la suma completa. No se aplica `expm1` a cada
contribución por separado para sumarla en minutos. La base transformada tampoco
equivale a la duración media de los viajes.

Las variables de distancia y su logaritmo son redundantes, y existen varias
codificaciones de hora y zona. Esa dependencia puede repartir la importancia entre
columnas; una posición en el ranking no equivale a una contribución causal aislada.
Los gráficos de dependencia coloreados sugieren patrones del modelo, pero no prueban
una interacción causal ni sustituyen el cálculo de valores SHAP de interacción.

## Ejemplos individuales

Se seleccionan el menor error, el error mediano y el mayor error absoluto dentro
de la muestra SHAP. Son casos ilustrativos elegidos por su error, no una muestra
representativa adicional ni evidencia de un sesgo sistemático.

{markdown(cases)}

Las figuras waterfall muestran contribuciones en log1p. Los errores residuales
no permiten atribuir el resultado a congestión, clima o incidentes no observados.

## Cobertura geográfica y errores

Estas tablas se calculan sobre todo el test de junio, no solo la muestra SHAP.
`error_medio_firmado` = predicción menos duración real: negativo indica subestimación
media en el grupo observado, positivo sobreestimación media. `fraccion_subestimada`
es la proporción de viajes con predicción inferior a la duración real. Un MAE alto
por sí solo no identifica la dirección del error. Se informa n y cobertura para
no interpretar grupos pequeños como evidencia concluyente.

### Borough de origen

{markdown(geography)}

### Viajes con origen o destino en un aeropuerto

{markdown(airports)}

### Disponibilidad del número de pasajeros

{markdown(missing)}

Las diferencias entre estos grupos son descriptivas y pueden reflejar diferencias
en rutas, horarios o duración. No prueban discriminación causal ni la causa de los
datos faltantes. No se afirma una relación con `payment_type`, que no se analiza aquí.

## Limitaciones y riesgos de uso

- El alcance es taxi amarillo y el periodo observado; no está demostrada la
  generalización a otras flotas, ciudades o estaciones del año.
- La limpieza elimina duraciones fuera de 1–180 minutos, zonas 264/265, distancias
  no positivas y velocidades mayores a 80 mph, entre otras reglas. Las métricas
  describen el conjunto retenido, no todos los viajes crudos o futuros.
- La distancia usada por el modelo es una mediana histórica por ruta, aprendida
  solo en entrenamiento, con respaldo por origen y mediana global. No es distancia
  entre centroides ni la distancia real del viaje de test. Los tramos de distancia
  real del CSV de errores son análisis posterior, no variables predictoras.
- La inferencia requiere conocer el destino declarado al recojo. Las zonas no
  identifican direcciones exactas ni condiciones de tráfico en vivo.
- Los seis meses no permiten caracterizar un ciclo anual completo. `pickup_month`
  no aprende una tendencia extrapolable por tener un valor SHAP distinto de cero;
  junio no estuvo en entrenamiento.
- El TargetEncoder conserva el cross-fitting interno de cinco folds del proyecto.
  Esa partición interna no es temporal estricta, aunque el test junio permanece
  separado del ajuste. Debe distinguirse de la validación temporal externa.
- Las predicciones son estimaciones puntuales. No se han calibrado intervalos ni
  validado decisiones de asignación de flota o garantías de llegada. Su uso como
  garantía puede perjudicar a usuarios o conductores ante errores grandes.
- Futuro trabajo: validación en meses posteriores, auditoría de rutas poco
  frecuentes, evaluación de incertidumbre y de datos de tráfico disponibles al
  recojo. Estas propuestas no se presentan como mejoras ya verificadas.

## Evidencia y reproducción

Ejecutar `notebooks/06_interpretabilidad.ipynb` con `requirements_semana13.txt`.
Los resultados de esta ejecución están en `outputs/semana13/` y las figuras en
`reports/semana13/figures/`. El modelo y el preprocesador se guardan en el directorio
de Drive indicado por el notebook. El archivo `modelo_metadata.json` registra
configuración, versiones y hashes. Las tablas completas están en
`importancias.csv`, `errores_segmentos.csv` y `casos_locales.csv`.

La redacción interpretativa adicional deberá contrastarse con las figuras nuevas;
no se trasladan conclusiones numéricas de la ejecución descartada.
"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path
