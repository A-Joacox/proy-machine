# Predicción de la duración de viajes en taxi en NYC

## 1. Título del proyecto
Predicción de la duración de viajes de taxis amarillos en Nueva York usando información disponible al momento del recojo.

## 2. Integrantes
- Diego Alarcon
- Joaquin Mercado
- Joaquin Justo
- Randy Rojas

## 3. Dataset elegido
**NYC Yellow Taxi Trip Records** — NYC Taxi & Limousine Commission (TLC).
URL: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page

Usamos los registros de taxis amarillos de **enero a junio de 2025** (6 meses), descargados directamente de la TLC en formato Parquet. Cada mes contiene aproximadamente 3 millones de viajes, por lo que el volumen total supera los 18 millones de filas. **Licencia y acceso:** datos abiertos publicados por la TLC bajo los términos de NYC Open Data.

El dataset es de fuente oficial verificable y presenta problemas reales: outliers extremos, valores faltantes, registros corruptos (duraciones negativas, distancias de 0), deriva temporal y riesgo de leakage.

Para la exploración inicial trabajamos con un mes completo, este siendo enero del 2025; para el modelado final usaremos los 6 meses con muestreo estratificado si el volumen excede la capacidad de cómputo disponible.

## 4. Pregunta predictiva
¿Cuántos minutos durará un viaje en taxi, dado lo que se sabe **en el momento en que el pasajero sube al vehículo**?

Esta pregunta es accionable: una predicción de duración alimenta estimaciones de llegada (ETA), asignación de flota y detección de viajes anómalos.

## 5. Variable objetivo
`trip_duration_min` = (`tpep_dropoff_datetime` − `tpep_pickup_datetime`) en minutos.

Es una variable derivada, continua y positiva. El problema es de **regresión**.

## 6. Unidad de predicción
Un viaje individual (una fila del dataset = un viaje con recojo y destino).

## 7. Variables disponibles antes de la predicción
Al momento del recojo se conoce:

- `tpep_pickup_datetime` → hora del día, día de la semana, mes, feriado
- `PULocationID` (zona de recojo) y `DOLocationID` (zona de destino, declarada por el pasajero)
- `passenger_count`
- Distancia **estimada** entre centroides de las zonas de recojo y destino (feature construida por nosotros; ver punto 8)
- `VendorID`

## 8. Riesgos de leakage
Este dataset tiene leakage evidente si se usa sin criterio. Las siguientes columnas **solo existen al finalizar el viaje** y quedan excluidas del modelo:

| Variable | Por qué es leakage |
|---|---|
| `trip_distance` | Es la distancia recorrida medida por el taxímetro, conocida solo al terminar |
| `fare_amount`, `total_amount`, `extra`, `mta_tax` | La tarifa depende del tiempo y distancia finales |
| `tip_amount`, `tolls_amount` | Se registran al pagar |
| `payment_type` | Se conoce al final del viaje |
| `tpep_dropoff_datetime` | Es directamente la respuesta |
| `RatecodeID` | Puede modificarse durante el viaje (p. ej. tarifa negociada); solo el código de aeropuerto sería conocido al recojo |
| `congestion_surcharge`, `cbd_congestion_fee`, `improvement_surcharge`, `Airport_fee` | Recargos calculados al cerrar el viaje según el recorrido efectivo |
| `store_and_fwd_flag` | Indica si el registro se guardó offline; se fija al terminar el viaje |

En lugar de `trip_distance`, construimos una distancia estimada (haversine entre centroides de las 265 zonas TLC), que sí estaría disponible al momento del recojo.

**Medidas de control:**
- Lista blanca explícita de features (sección 7); toda columna fuera de esa lista se descarta en `src/features.py`, no se excluye "a mano" en cada notebook.
- Split temporal (sección 10): evita filtrar patrones del futuro al pasado, un leakage sutil que el split aleatorio no detecta.
- Codificaciones que dependen del target (target encoding de zonas) se ajustan solo con datos de entrenamiento, dentro del pipeline de scikit-learn.
- Chequeo de sanidad: si algún modelo obtiene un MAE muy por debajo de lo razonable (p. ej. < 2 min), se audita la lista de features antes de reportarlo.

## 9. Métrica principal y métrica secundaria
- **Principal: MAE (Mean Absolute Error) en minutos.** Interpretable directamente ("nos equivocamos en promedio por X minutos") y robusta frente a los outliers extremos que este dataset contiene.
- **Secundaria: RMSLE (Root Mean Squared Log Error).** Penaliza errores relativos, apropiada porque la duración tiene distribución con cola larga y un error de 5 min pesa distinto en un viaje de 8 min que en uno de 60.

No usamos accuracy (no aplica a regresión) y evitamos RMSE puro como métrica principal porque los outliers la dominarían.

## 10. Plan de validación
**Split temporal**, no aleatorio: los viajes tienen deriva estacional y de demanda, y un split aleatorio filtraría información del futuro al pasado.

- Entrenamiento: enero–abril 2025
- Validación (decisiones de modelado e hiperparámetros): mayo 2025
- Test final (se toca una sola vez, al final): junio 2025

Dentro del entrenamiento usaremos validación cruzada temporal (`TimeSeriesSplit`) para la búsqueda de hiperparámetros. El conjunto de test no se usa para ninguna decisión de modelado.

## 11. Modelo baseline
Tres niveles de baseline, del más simple al más informativo:

1. **Mediana global** de la duración (predictor constante).
2. **Mediana por grupo**: mediana condicionada a (zona de recojo, hora del día).
3. **Regresión lineal** con features temporales y de zona.

Todo modelo posterior (árboles, gradient boosting) debe justificar su complejidad superando estos baselines.

## 12. Riesgos técnicos
- **Volumen**: 18M+ filas exceden la RAM de una laptop; mitigación: lectura por columnas con PyArrow, tipos optimizados, muestreo estratificado documentado.
- **Calidad de datos**: duraciones negativas o de días enteros, `passenger_count` = 0, zonas desconocidas (264/265); requieren reglas de limpieza explícitas y justificadas.
- **Deriva temporal**: patrones de tráfico cambian entre meses; el split temporal lo expone honestamente.
- **Sesgos**: el dataset solo cubre taxis amarillos (predominantes en Manhattan y aeropuertos); las conclusiones no generalizan a toda la movilidad de NYC ni a zonas periféricas, donde el servicio y los datos son más escasos.
- **Alta cardinalidad**: 265 zonas TLC; codificarlas mal (one-hot completo) explota la dimensionalidad; evaluaremos target encoding con cuidado de hacerlo dentro del pipeline para no filtrar información.
- **Random Forest sin límite de profundidad**: con `max_depth=None` y muchos árboles sobre 18M+ filas, cada árbol completo se guarda en memoria y puede superar la RAM disponible; mitigación: acotar `max_depth`/`max_samples`, o usar `HistGradientBoostingRegressor`.

## 13. Plan de trabajo — semanas restantes
| Semana | Actividad |
|---|---|
| 8 | Limpieza definitiva y pipeline de features reproducible (`src/data.py`, `src/features.py`) |
| 9 | Baselines formales + regresión lineal regularizada sobre los 6 meses |
| 10 | Modelos de árboles y gradient boosting (Random Forest, LightGBM/XGBoost) |
| 11 | Búsqueda de hiperparámetros con validación temporal |
| 12 | Evaluación final sobre junio, análisis de errores por segmento (zona, hora, distancia) |
| 13 | Interpretabilidad (importancia de features, SHAP), discusión de sesgos y limitaciones |
| 14–15 | Informe final, figuras, presentación |
| 16 | Entrega final |
