from sklearn.pipeline import Pipeline
from nyoka import skl_to_pmml
from teradataml import create_context
from teradataml.dataframe.dataframe import DataFrame
from sklearn.linear_model import LogisticRegression
import joblib
import os

def train(data_conf, model_conf, **kwargs):
    """Python train method called by AOA framework

    Parameters:
    data_conf (dict): The dataset metadata
    model_conf (dict): The model configuration to use

    Returns:
    None:No return

    """

    create_context(
            host=os.environ["AOA_CONN_HOST"], 
            username=os.environ["AOA_CONN_USERNAME"], 
            password=os.environ["AOA_CONN_PASSWORD"],
            database=data_conf["schema"]
    )
    
    hyperparams = model_conf["hyperParameters"]

    # load data & engineer
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
    
    train_df = DataFrame('ACC_TRAIN')
    train_df = train_df.select([feature_names + [target_name]])
    train_df = train_df.to_pandas()

    X_train = train_df.drop(target_name, axis=1)
    y_train = train_df[target_name]

    print("Starting training...")

    # fit model to training data
    
    model = Pipeline([('model', LogisticRegression(penalty=hyperparams["penalty"], random_state=hyperparams["random_state"]))])
    model.fit(X_train, y_train)

    print("Finished training")

    # export model artefacts to models/ folder
    
    import matplotlib.pyplot as plt
    importance_values = model[0].coef_[0]
    feature_importance = {feature_names[key]: value for (key, value) in enumerate(importance_values)}
    # summarize feature importance
    for i,v in feature_importance.items():
        print('Feature: %s, Score: %.5f' % (i,v))
    # plot feature importance
    plt.bar(range(len(importance_values)), importance_values)
    plt.xticks(ticks = range(len(importance_values)), labels = feature_names, rotation = 'vertical')
    plt.show()
    
    joblib.dump(model, "artifacts/output/model.joblib")

    print("Saved trained model")