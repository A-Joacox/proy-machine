# Semana 13: interpretabilidad, sesgos y limitaciones

## Alcance y procedencia

Análisis nuevo desde la configuración de semana 12. No se leen ni reutilizan las
figuras o estadísticas del notebook 06 eliminado. Se usan registros reales de NYC
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
