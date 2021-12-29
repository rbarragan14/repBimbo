from sklearn.pipeline import Pipeline
from nyoka import skl_to_pmml
from teradataml import create_context
from teradataml.dataframe.dataframe import DataFrame
from sklearn.linear_model import LogisticRegression
from aoa.stats import stats
from aoa.util.artefacts import save_plot
import matplotlib.pyplot as plt
import joblib
import os


def train(data_conf, model_conf, **kwargs):
    hyperparams = model_conf["hyperParameters"]

    create_context(
        host=os.environ["AOA_CONN_HOST"],
        username=os.environ["AOA_CONN_USERNAME"],
        password=os.environ["AOA_CONN_PASSWORD"],
        database=data_conf["schema"])

    feature_names = ['estado_civil_jefe_CASADO_accidentes',
                     'nivel_Operativo_accidentes',
                     'antiguedad_empresa_accidentes',
                     'masculino_planta',
                     'Severidad_con_Seguras_ci_dSegura',
                     'total_reportes_accidentes',
                     'Severidad_con_Seguras_ci_cBajo',
                     'Supervisor_planta',
                     'edad_planta',
                     'pais_COSTA_RICA_value_1_0',
                     'mes_anterior_value_1',
                     'accidentes_total_value_1',
                     'pais_COLOMBIA_value_1_0',
                     'pais_HONDURAS_value_1_0',
                     'lugar_de_trabajo_PLANTA_value_1_0',
                     'pais_VENEZUELA_value_1_0',
                     'pais_EL_SALVADOR_value_1_0',
                     'pais_PANAMA_value_1_0']

    target_name = 'ptarget'

    # read training dataset from Teradata and convert to pandas
    train_tdf = DataFrame(data_conf["table"])
    train_df = train_tdf.select([feature_names + [target_name]])
    train_df = train_df.to_pandas()

    # split data into X and y
    X_train = train_df.drop(target_name, axis=1)
    y_train = train_df[target_name]

    print("Starting training...")

    # fit model to training data
    model = Pipeline([('model', LogisticRegression(penalty=hyperparams["penalty"], random_state=hyperparams["random_state"]))])
    # Logistic Regression saves feature names but lets store on pipeline for easy access later
    model.feature_names = feature_names
    model.target_name = target_name

    model.fit(X_train, y_train)

    print("Finished training")

    # export model artefacts
    joblib.dump(model, "artifacts/output/model.joblib")
    skl_to_pmml(pipeline=model, col_names=feature_names, target_name=target_name, pmml_f_name="artifacts/output/model.pmml")

    print("Saved trained model")

    # record training statistics   
    importance_values = model[0].coef_[0]
    feature_importance = {feature_names[key]: value for (key, value) in enumerate(importance_values)}
    
    plt.bar(range(len(importance_values)), importance_values)
    plt.xticks(ticks = range(len(importance_values)), labels = feature_names, rotation = 'vertical')
    save_plot('Feature Importance')
    
    stats.record_training_stats(train_tdf,
                       features=feature_names,
                       predictors=[target_name],
                       categorical=[target_name],
                       importance=feature_importance,
                       category_labels={target_name: {0: "Falso", 1: "Verdadero"}})

    
    print("All done!")
