from sklearn import metrics
from teradataml import create_context
from teradataml.dataframe.dataframe import DataFrame
from teradataml.dataframe.copy_to import copy_to_sql
from aoa.stats import stats
from aoa.util.artefacts import save_plot
import matplotlib.pyplot as plt
import os
import joblib
import json
import pandas as pd


def save_plot(title):
    import matplotlib.pyplot as plt

    plt.title(title)
    fig = plt.gcf()
    filename = title.replace(" ", "_").lower()
    fig.savefig('artifacts/output/{}'.format(filename), dpi=500)
    plt.clf()


def evaluate(data_conf, model_conf, **kwargs):
    model = joblib.load('artifacts/input/model.joblib')

    create_context(host=os.environ["AOA_CONN_HOST"],
                   username=os.environ["AOA_CONN_USERNAME"],
                   password=os.environ["AOA_CONN_PASSWORD"],
                   database=data_conf["schema"] if "schema" in data_conf and data_conf["schema"] != "" else None)

    # Read test dataset from Teradata
    # As this is for demo purposes, we simulate the test dataset changing between executions
    # by introducing a random sample. Note that the sampling is performed in Teradata!
    test_tdf = DataFrame(data_conf["table"]).sample(frac=0.8)
    test_df = test_tdf.to_pandas()

    X_test = test_df[model.feature_names]
    y_test = test_df[model.target_name]

    print("Starting evaluation...")

    y_pred = model.predict(test_df[model.feature_names])
    y_prob = model.predict_proba(test_df[model.feature_names])[:,1]
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_prob)
    
    y_pred_tdf = pd.DataFrame(y_pred, columns=[model.target_name])
    y_pred_tdf["id"] = test_df["id"].values
    
    print("Finished evaluation")

    evaluation = {
        'AUC value': '{:.2f}'.format(metrics.auc(fpr, tpr)),
        'Accuracy': '{:.2f}'.format(metrics.accuracy_score(y_test, y_pred)),
        'Recall': '{:.2f}'.format(metrics.recall_score(y_test, y_pred)),
        'Precision': '{:.2f}'.format(metrics.precision_score(y_test, y_pred)),
        'f1-score': '{:.2f}'.format(metrics.f1_score(y_test, y_pred))
    }

    print("Storing metrics and plots...")
    
    with open("artifacts/output/metrics.json", "w+") as f:
        json.dump(evaluation, f)

    metrics.ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    save_plot('Confusion Matrix')

    metrics.RocCurveDisplay.from_predictions(y_test, y_prob)
    save_plot('ROC Curve')

    importance_values = model[0].coef_[0]
    feature_importance = {model.feature_names[key]: value for (key, value) in enumerate(importance_values)}
    plt.bar(range(len(importance_values)), importance_values)
    plt.xticks(ticks = range(len(importance_values)), labels = model.feature_names, rotation = 'vertical')
    save_plot('Feature Importance')

    predictions_table = "{}_tmp".format(data_conf["predictions"]).lower()
    copy_to_sql(df=y_pred_tdf, table_name=predictions_table, index=False, if_exists="replace", temporary=False)

    stats.record_evaluation_stats(test_tdf, DataFrame(predictions_table), feature_importance)

    print("All done!")
