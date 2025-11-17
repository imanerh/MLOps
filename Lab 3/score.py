import json
import pandas as pd
import joblib
import os
from azureml.core.model import Model
import mlflow

def init():
    global model
    model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), "downloaded_model")
    model = mlflow.sklearn.load_model(model_path)

def run(raw_data):
    try:
        data = json.loads(raw_data)
        df = pd.DataFrame(data['data'])
        predictions = model.predict(df)
        return json.dumps({'predictions': predictions.tolist()})
    except Exception as e:
        return json.dumps({'error': str(e)})