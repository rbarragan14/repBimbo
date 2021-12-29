# Modelo RRHH Accidentalidad

Modelo de regresión logística para predecir la accidentalidad.

## Descripción de los Datos

Los datos ya se encuentran curados y preparados en la instancia de Vantage en las tablas/vistas `ACC_TRAIN`, `ACC_TEST` y `ACC_PREDICTIONS`.

## Entrenamiento

El entrenamiento se realiza en el archivo [training.py](./model_modules/training.py)

## Evaluación

La evaluación se realiza en el archivo [evaluation.py](./model_modules/evaluation.py)

## Scoring

El scoring se realiza en el archivo [scoring.py](./model_modules/scoring.py)