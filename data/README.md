# data/

Los archivos Parquet NO se versionan en el repositorio por su tamaño (ver `.gitignore`).

El notebook `notebooks/01_exploracion_inicial.ipynb` descarga automáticamente
`yellow_tripdata_2025-01.parquet` (~60 MB) desde la TLC a esta carpeta.

Descarga manual alternativa:
https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2025-01.parquet

`Taxi_Zone_Lookup.csv` (sí versionado, 12 KB) mapea `LocationID` → borough y nombre de zona.
Fuente: https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv

Fuente oficial: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
