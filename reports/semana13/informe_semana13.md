# Semana 13: interpretabilidad, sesgos y limitaciones

## Alcance y procedencia

Análisis nuevo desde la configuración de semana 12. Se usan registros reales de NYC
Yellow Taxi (enero-junio de 2025), la limpieza de `src/data.py` y las variables de
`src/features.py`. Entrenamiento: 9,319,633 viajes de enero-mayo;
test: 2,008,054 de junio. Ambos conservan el muestreo mensual
del 50 % y semilla 42. La configuración de LightGBM conserva sus 1710 árboles.

El preprocesamiento por bloques conserva float64 y el cross-fitting de
TargetEncoder. Se compara con el pipeline original sobre 3000 filas reales antes
del entrenamiento completo. Esta comprobación técnica no es una evaluación predictiva.

## Control de reproducción

- MAE reconstruido: 3.693972 minutos.
- RMSLE reconstruido: 0.306646.
- Diferencia MAE frente a semana 12: +0.000000 minutos.
- Diferencia RMSLE: +0.000000.

Los límites de control se fijaron antes de evaluar: 0.02 minutos de MAE y 0.002
de RMSLE. No son intervalos de confianza ni prueban identidad entre los modelos.
No se archivaron todas las versiones ni hashes de los datos de la ejecución de
semana 12; esta ejecución sí los registra. Junio se usa para verificar y describir
el modelo congelado, nunca para seleccionar hiperparámetros o variables nuevas.

## Importancia global

SHAP se calcula en 2,000 viajes seleccionados al azar, sin reemplazo,
del test de junio, con semilla 42. El entrenamiento usa todas las
filas indicadas arriba. La muestra SHAP solo reduce el costo de explicar predicciones;
no garantiza cobertura suficiente de rutas raras.

| feature | mean_abs_shap_log1p | gain_pct |
| --- | --- | --- |
| num__distancia_estimada_mi | 0.312823 | 52.516179 |
| num__log_distancia_estimada | 0.128189 | 22.292732 |
| num__hora_cos | 0.061433 | 2.032984 |
| num__pickup_hour | 0.054683 | 3.880828 |
| num__pickup_month | 0.04882 | 0.940246 |
| target_enc__PULocationID | 0.047862 | 5.074294 |
| target_enc__DOLocationID | 0.047614 | 5.419307 |
| num__hora_sin | 0.021116 | 1.227957 |
| num__dia_sin | 0.020943 | 0.855743 |
| num__pickup_dayofweek | 0.016375 | 1.142038 |
| num__dia_cos | 0.014982 | 0.558468 |
| num__es_fin_de_semana | 0.014961 | 0.518058 |
| onehot__passenger_cat_desconocido | 0.014423 | 0.32545 |
| num__mismo_borough | 0.012645 | 0.451648 |
| onehot__passenger_cat_1 | 0.010924 | 0.216529 |

Gain es la importancia acumulada en los splits del entrenamiento; no representa un
porcentaje de mejora del MAE de test. `mean_abs_shap_log1p` mide magnitud de contribución
al modelo en escala log1p. Ninguna de las dos medidas demuestra causalidad.

El método es TreeSHAP `tree_path_dependent`, con salida `raw`. Las contribuciones
explican log(1 + duración), no minutos. Se verifica base + suma(SHAP) = salida del
modelo; error máximo observado: 7.55e-14.
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

| criterio | posicion_test | fila_archivo | real_min | pred_min | error_firmado |
| --- | --- | --- | --- | --- | --- |
| menor_error | 1342885 | 1741025 | 22.8667 | 22.8672 | 0.0005 |
| error_mediano | 1417339 | 3724593 | 15.3167 | 12.758 | -2.5586 |
| mayor_error | 1310628 | 3439373 | 55.5333 | 13.7584 | -41.775 |

Las figuras waterfall muestran contribuciones en log1p. Los errores residuales
no permiten atribuir el resultado a congestión, clima o incidentes no observados.

## Interpretación de las figuras

Sección redactada a partir de las figuras de `reports/semana13/figures/` y de las
tablas de `outputs/semana13/`. No la genera el notebook: si se regenera el informe,
hay que conservarla. Todas las contribuciones están en escala log1p; como referencia,
+0.10 equivale a multiplicar la duración predicha por ≈1.11 y −0.20, por ≈0.82.

### Distribución de los valores SHAP (`03_shap_distribucion.png`)

- **La distancia estimada define el orden de magnitud del viaje.** Sus
  contribuciones van de −0.7 a +1.1 (y de −0.3 a +0.5 en su versión logarítmica):
  valores bajos (azul) restan y valores altos (rojo) suman. Las demás variables se
  mueven aproximadamente entre −0.25 y +0.3.
- **Hora del día.** `hora_cos` alta (horas cercanas a la medianoche) resta y baja
  (horas cercanas al mediodía) suma; `pickup_hour` y `hora_sin` reparten el resto
  del mismo efecto. Las tres codifican la misma información y deben leerse juntas.
- **Zonas de origen y destino.** Las codificaciones de zona tienen contribuciones de
  ambos signos sin un patrón de color claro: ajustan cada viaje según la zona,
  después de considerar la distancia.
- **Fin de semana.** `es_fin_de_semana = 1` resta (hasta ≈−0.12): con la misma
  ruta y hora, el modelo predice viajes algo más cortos en sábado y domingo.
- **`pickup_month`.** En la muestra de junio casi todas las contribuciones son
  positivas (hasta ≈+0.13). Como junio no estuvo en entrenamiento, el modelo le
  aplica el efecto aprendido para los meses más recientes; no es una tendencia
  estimada para junio.
- **`passenger_cat_desconocido = 1` suma** (hasta ≈+0.12). No mide el número de
  pasajeros: en el archivo crudo de junio, los registros sin `passenger_count`
  (28 %) coinciden exactamente con los de `payment_type = 0`, un bloque de registros
  reportados sin esos metadatos. La variable funciona como indicador de ese tipo de
  registro.
- **`mismo_borough = 0` suma**: los viajes entre boroughs distintos se predicen más
  largos que los viajes de igual distancia estimada dentro de un mismo borough.

### Dependencia de la distancia (`04_dependencia_distancia.png`)

La relación es creciente y cóncava. Entre ≈0.5 y ≈3 millas la contribución sube de
−0.7 a +0.3; desde ahí crece más despacio (≈+0.5 a 7 millas y ≈+1.0 entre 18 y 26
millas). Cada milla adicional agrega menos duración en viajes largos, lo que es
compatible con mayor velocidad media en trayectos largos (autopistas, puentes). Esta
figura muestra solo `distancia_estimada_mi`; `log_distancia_estimada` agrega su
propia contribución en la misma dirección.

El color (hora de recojo) aparece mezclado a lo largo de la curva: para una misma
distancia, la contribución no cambia de forma visible según la hora. El modelo trata
distancia y hora de forma aproximadamente aditiva; la hora actúa sobre todo a través
de sus propias variables. Esto no descarta interacciones pequeñas, que requerirían
valores SHAP de interacción.

### Dependencia de la hora (`05_dependencia_hora.png`)

- La contribución de `pickup_hour` es negativa de madrugada (mínima entre las 2 y
  las 6 h, entre −0.10 y −0.25) y positiva entre las 10 y las 17 h (máxima a las
  15-16 h, ≈+0.05 a +0.12). Desde las 18 h vuelve a bajar.
- **Fin de semana (rojo) frente a días hábiles (azul):** entre las 0 y las 2 h la
  contribución es menos negativa en fin de semana; a las 8-9 h es más baja, porque
  no hay hora punta matutina; y entre las 18 y las 23 h queda levemente por encima.
  Es compatible con más actividad nocturna y menos tráfico de mañana los fines de
  semana.
- Es solo la parte del efecto asignada a `pickup_hour`; el efecto completo de la
  hora suma también `hora_sin`, `hora_cos` y `es_hora_punta`.

### Casos individuales (`06_caso_*.png`)

Rutas y distancias reales obtenidas de las filas indicadas en `casos_locales.csv`
(`fila_archivo`) dentro de `yellow_tripdata_2025-06.parquet`. La distancia real
(`trip_distance`) se usa solo para describir el caso; no es una variable del modelo.

| caso | ruta | recojo | distancia estimada | distancia real | real | predicción |
| --- | --- | --- | --- | --- | --- | --- |
| menor error | East Chelsea → Upper West Side North | mar 17-jun, 17:54 | 3.66 mi | 4.30 mi | 22.9 min | 22.9 min |
| error mediano | Financial District North → UN/Turtle Bay South | dom 15-jun, 22:36 | 4.75 mi | 5.33 mi | 15.3 min | 12.8 min |
| mayor error | Sutton Place/Turtle Bay North → Central Park | dom 8-jun, 15:30 | 1.77 mi | 3.32 mi | 55.5 min | 13.8 min |

- **Menor error.** La distancia suma +0.43 (entre sus dos versiones) y la hora punta
  de la tarde +0.12 (`pickup_hour` y `hora_cos`). El destino resta −0.09. Es un caso
  típico: ruta frecuente de Manhattan en día hábil, donde la mediana histórica
  describe bien el viaje.
- **Error mediano.** La distancia suma +0.37, pero las zonas de origen y destino
  restan −0.23, la hora nocturna −0.10 (`hora_cos` y `pickup_hour`) y el domingo
  −0.04. El modelo predice 12.8 min y el
  viaje duró 15.3: la distancia real superó a la estimada en 0.6 millas.
- **Mayor error.** La distancia estimada es corta (1.77 mi) y casi no mueve la
  predicción (−0.03). La hora de la tarde suma +0.12 y el domingo resta ≈−0.19
  (`pickup_dayofweek`, `dia_sin`, `es_fin_de_semana`, `dia_cos`). El modelo predice
  13.8 min, coherente con lo que sabe al recojo. El viaje real recorrió 3.32 millas,
  casi el doble de la mediana de la ruta, a una velocidad media de 3.6 mph: indica un
  desvío o detenciones prolongadas que ninguna variable disponible al recojo puede
  anticipar. Es el tipo de error que explica la limitación principal del modelo.

## Cobertura geográfica y errores

Estas tablas se calculan sobre todo el test de junio, no solo la muestra SHAP.
`error_medio_firmado` = predicción menos duración real: negativo indica subestimación
media en el grupo observado, positivo sobreestimación media. `fraccion_subestimada`
es la proporción de viajes con predicción inferior a la duración real. Un MAE alto
por sí solo no identifica la dirección del error. Se informa n y cobertura para
no interpretar grupos pequeños como evidencia concluyente.

### Borough de origen

| grupo | n | MAE | error_medio_firmado | fraccion_subestimada | porcentaje_test |
| --- | --- | --- | --- | --- | --- |
| Bronx | 16961 | 6.2017 | -0.649 | 0.4752 | 0.8446 |
| Brooklyn | 88463 | 5.107 | -0.6387 | 0.4871 | 4.4054 |
| EWR | 15 | 11.6567 | -7.7259 | 0.6 | 0.0007 |
| Manhattan | 1707492 | 3.3273 | -0.2479 | 0.456 | 85.0322 |
| Queens | 194919 | 6.0399 | -0.6227 | 0.477 | 9.7069 |
| Staten Island | 204 | 9.7786 | -3.7657 | 0.5735 | 0.0102 |

### Viajes con origen o destino en un aeropuerto

| grupo | n | MAE | error_medio_firmado | fraccion_subestimada | porcentaje_test |
| --- | --- | --- | --- | --- | --- |
| False | 1838027 | 3.4338 | -0.2693 | 0.4584 | 91.5327 |
| True | 170027 | 6.5061 | -0.6946 | 0.4731 | 8.4673 |

### Disponibilidad del número de pasajeros

| grupo | n | MAE | error_medio_firmado | fraccion_subestimada | porcentaje_test |
| --- | --- | --- | --- | --- | --- |
| False | 1450615 | 3.7699 | -0.3282 | 0.4573 | 72.2398 |
| True | 557439 | 3.4964 | -0.2456 | 0.4655 | 27.7602 |

Las diferencias entre estos grupos son descriptivas y pueden reflejar diferencias
en rutas, horarios o duración. No prueban discriminación causal ni la causa de los
datos faltantes. Lo que sí se verificó es su origen: en el archivo crudo de junio,
los registros sin `passenger_count` coinciden exactamente con los de
`payment_type = 0` (ver "Interpretación de las figuras").

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

Ejecutar `notebooks/06_interpretabilidad.ipynb` con `requirements.txt`.
Los resultados de esta ejecución están en `outputs/semana13/` y las figuras en
`reports/semana13/figures/`. El modelo y el preprocesador se guardan en el directorio
de Drive indicado por el notebook. El archivo `modelo_metadata.json` registra
configuración, versiones y hashes. Las tablas completas están en
`importancias.csv`, `errores_segmentos.csv` y `casos_locales.csv`.

La interpretación de las figuras está en la sección "Interpretación de las figuras",
contrastada con las figuras y tablas de esta ejecución. No se trasladan conclusiones
numéricas de la ejecución descartada.
