# Modelo RRHH Accidentalidad

Modelo de regresión logística para predecir la accidentalidad.

## Descripción de los Datos

Los datos ya se encuentran curados y preparados en la instancia de Vantage en las tablas/vistas `ACC_TRAIN`, `ACC_TEST` y `ACC_PREDICTIONS`.

Como referencia, los datasets a generar y requeridos para ejecutar el modelo son:

Entrenamiento
```json
{
    "schema": "aoa_data",
    "table": "ACC_TRAIN"
}
```
Evaluación

```json
{
    "schema": "aoa_data",
    "table": "ACC_TRAIN",
    "predictions": "ACC_PREDICTIONS"
}
```

Scoring por lotes
```json
{
    "schema": "aoa_data",
    "table": "ACC_TRAIN",
    "predictions": "ACC_PREDICTIONS"
}
 ```

## Entrenamiento

El entrenamiento se realiza en el archivo [training.py](./model_modules/training.py) y genera los siguientes artefactos

- model.joblib            (pipeline de sklearn con el modelo de regresión logística)
- model.pmml              (versión en formato pmml del pipeline de sklearn)
- data_stats.json         (estadísticas de los datos utilizado para la monitorización)
- feature_importance.png  (gráfica de las __feature importance__)

## Evaluación

La evaluación está definida en el método `evaluate` en el archivo [evaluation.py](./model_modules/evaluation.py) y retorna las siguientes métricas

- AUC value
- Accuracy
- Recall
- Precision
- f1-score

Además en cada evaluación genera las siguientes gráficas

- Curva ROC
- Matriz de Confusión
- Feature Importance

## Scoring

El scoring se realiza en el archivo [scoring.py](./model_modules/scoring.py)

Este modelo soporta tres tipos de scoring

 - Por lotes (Batch)
 - RESTful
 - In-Vantage (IVSM)

El scoring del tipo __In-Vantage__ está soportado a través del archivo PMML del modelo generado en la fase de entrenamiento. Dado que la evaluación es un scoring en batch más una comparación, la lógica del scoring ya se ha validado previamente en la fase de evaluación. Los resultados del scoring en batch se guardan en la tabla de predicciones definida en la plantilla de datos (dataset template) en la sección de `scoring`.

La siguiente tabla debe existir previamente a la adición de prediciones

```sql
CREATE MULTISET TABLE IVSM_ACC_PREDICTIONS, FALLBACK ,
     NO BEFORE JOURNAL,
     NO AFTER JOURNAL,
     CHECKSUM = DEFAULT,
     DEFAULT MERGEBLOCKRATIO,
     MAP = TD_MAP1
     (
        job_id VARCHAR(255),
        id BIGINT, 
        score_result CLOB(2097088000) CHARACTER SET LATIN
     )
     PRIMARY INDEX ( job_id );
```

Y la siguiente vista debe existir para extraer la predicción específica del json resultante de IVSM.

```sql
CREATE VIEW IVSM_ACC_PREDICTIONS_V AS 
    SELECT job_id, id, CAST(CAST(score_result AS JSON).JSONExtractValue('$.predicted_ptarget') AS INT) as Prediccion 
    FROM IVSM_ACC_PREDICTIONS;
```

El scoring en modo RESTful está soportado a través de la clase `ModelScorer` que implementa el método `predict` que es llamado por el RESTful Serving Engine. Una petición de ejemplo sería:  

    curl -X POST http://<service-name>/predict \
        -H "Content-Type: application/json" \
        -d '{
      "data": {
        "ndarray": [
          0.021,
          0.186,
          0.088,
          0.192396525,
          0.025615116,
          0,
          0.25346642,
          0.191203309,
          0.37474141,
          0,
          1,
          0,
          1,
          0,
          1,
          1,
          0,
          0
        ]
      }
    }' 

## Notebooks

En este ejemplo se incluyen diversos notebooks donde se muestran diversos aspectos del modelo

- [ModeloAccidentalidad.ipynb](./notebooks/ModeloAccidentalidad.ipynb): Descripción del proceso de generación del modelo.
- [CargaDatos.ipynb](./notebooks/CargaDatos.ipynb): Ejemplo de carga y preparación de los datos del modelo.
- [ComprobarBatch.ipynb](./notebooks/ComprobarBatch.ipynb): Ejemplo que permite comprobar las predicciones realizadas en el modo batch.
- [InVantageScoring.ipynb](./notebooks/InVantageScoring.ipynb): Ejemplo que permite comprobar las predicciones realizadas en el modo IVSM.
