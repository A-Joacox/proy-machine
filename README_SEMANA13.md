# Semana 13 

## Archivos de código

| Ruta | Propósito |
| --- | --- |
| `notebooks/06_interpretabilidad.ipynb` | Notebook completo para Google Colab; reemplaza el provisional de dos celdas. |
| `src/interpretabilidad.py` | Reconstrucción del modelo por bloques, checkpoint, verificaciones, SHAP, análisis e informe. |
| `requirements_semana13.txt` | Versiones comprobadas para esta ejecución. |
| `README_SEMANA13.md` | Instrucciones y alcance. |

Los resultados nuevos se generan en `outputs/semana13/` y `reports/semana13/`.

## Subir a GitHub

1. Extraer `Semana13_desde_semana12.zip` en la computadora.
2. Abrir `A-Joacox/proy-machine` y seleccionar la rama `semana-13`.
3. En la raíz del repositorio: **Add file → Upload files**.
4. Arrastrar las carpetas `src` y `notebooks`, y los archivos `requirements_semana13.txt` y `README_SEMANA13.md`.
5. Verificar que los archivos conserven las cuatro rutas de la tabla. Se reemplaza únicamente el notebook provisional; no se eliminan los demás archivos del repositorio.
6. Commit: `Semana 13: implementar reconstrucción e interpretabilidad desde semana 12`.
7. Confirmar el commit en la rama `semana-13`.

## Abrir en Colab

[Abrir el notebook de la rama semana-13](https://colab.research.google.com/github/A-Joacox/proy-machine/blob/semana-13/notebooks/06_interpretabilidad.ipynb)

El enlace funcionará después de subir los archivos. En Colab, guardar una copia en Drive para conservar el notebook con las salidas. Elegir CPU y ejecutar las secciones en orden.

1. Obtener el código de `semana-13`. Se usa `/content/proy-machine-s13`; la carpeta anterior no se modifica.
2. Instalar las dependencias. Si Colab solicita reiniciar, reiniciar y ejecutar desde la sección 1.
3. Comprobar los hashes de la base de semana 12 y los recursos.
4. Autorizar Drive. El checkpoint se guarda en `MyDrive/ML_NYC_semana13_desde_semana12/modelo/`.
5. Reconstruir y guardar el modelo. Esta es la fase larga; no se conoce de antemano su duración en el Colab del usuario.
6. Verificar las métricas en junio. Si el control falla, detenerse y revisar; no ajustar el modelo sobre junio.
7. Calcular tablas de errores sobre el test completo.
8. Calcular SHAP sobre 2000 viajes reales, seleccionados antes del análisis con semilla 42.
9. Generar y revisar figuras e informe. Los resultados se copian también a Drive.
10. Descargar el ZIP de resultados.

El modelo se guarda al terminar de entrenar. Si la sesión termina antes de ese punto, habrá que volver a entrenar. Después del guardado, la sección 5 carga el checkpoint compatible. La matriz temporal de entrenamiento está en disco local de Colab, no en Drive ni en GitHub.

## Commit de resultados

1. Descargar el notebook ejecutado desde **Archivo → Descargar → Descargar .ipynb**.
2. Extraer `semana13_resultados.zip`.
3. En GitHub, en la rama `semana-13`, subir las carpetas extraídas `outputs` y `reports`, respetando sus rutas.
4. Actualizar `notebooks/06_interpretabilidad.ipynb` con el notebook ejecutado (puede hacerse en la misma carga colocando el archivo dentro de una carpeta local `notebooks`).
5. Revisar que solo se incluyan el notebook y resultados nuevos; no subir Parquet crudos ni el checkpoint joblib.
6. Commit: `Semana 13: añadir resultados SHAP, sesgos y limitaciones`.
7. Revisar la rama completa antes de abrir el pull request a `main`.

Los resultados no se guardan en GitHub automáticamente desde Colab. El código no usa tokens ni ejecuta push.

## Metodología y comprobaciones

- Base: archivos del ZIP de semana 12; hashes verificados al inicio.
- Enero-mayo: 50 % por mes, semilla 42; entrenamiento esperado de 9,319,633 viajes.
- Junio: mismo muestreo; test esperado de 2,008,054 viajes.
- LightGBM: mismos hiperparámetros y 1710 árboles de `config_modelo_final.json`.
- Datos descargados desde los enlaces TLC del proyecto; no hay datos ficticios.
- Preprocesamiento: float64, mismas features y codificaciones; `TargetEncoder.fit_transform` conserva el cross-fitting global. OneHotEncoder usa las categorías observadas en filas reales. Se comparan ambas implementaciones en 3000 filas reales antes de entrenar.
- Tolerancias de control previas: diferencia MAE ≤ 0.02 minutos; RMSLE ≤ 0.002. No son garantías de identidad de modelos ni criterios para optimizar sobre test.
- SHAP: `tree_path_dependent`, salida raw en log1p, chequeo explícito de aditividad. No interpretar las contribuciones como minutos ni efectos causales.
- Métricas por grupo: n, cobertura, MAE, error firmado y fracción de subestimación. No se infiere sesgo sistemático solo a partir de MAE alto o un caso aislado.
- No se atribuyen errores a tráfico o clima no medidos, ni se afirma relación entre faltantes y payment_type sin analizarla.

## Validación técnica realizada antes de la entrega

Se descargaron registros reales de enero y junio de 2025. Se comprobaron equivalencia de matrices de entrenamiento e inferencia, nombres y orden de columnas, predicciones de dos modelos de prueba equivalentes, guardado y recarga del checkpoint, conteos por segmento, generación de figuras y aditividad SHAP. Se validaron sintaxis y formato del notebook. Estas comprobaciones usaron muestras reales y no constituyen la ejecución completa del modelo final.

La reconstrucción completa de 9.3 millones de viajes y sus resultados quedan pendientes de ejecución en Colab. Las salidas de pruebas técnicas no se incluyen en este paquete.

## Referencias técnicas

- [TargetEncoder y cross-fitting](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.TargetEncoder.html)
- [TreeExplainer](https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html)
- [LightGBM Booster](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.Booster.html)
