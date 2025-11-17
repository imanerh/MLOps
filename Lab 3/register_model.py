from azureml.core import Workspace, Model
import mlflow
import mlflow.sklearn
import os
import shutil

# Load workspace
ws = Workspace.from_config()
print(f"Connected to workspace: {ws.name}")

# MLFlow model details
RUN_ID = "2a50f8bbefcb49ad8fbd959cb8ac53fe" 
MODEL_NAME = "wine-quality-rf"
MODEL_PATH = f"runs:/{RUN_ID}/random_forest_model"

print(f"Downloading model from MLFlow...")

# Download the model locally first
local_model_path = "downloaded_model"
if os.path.exists(local_model_path):
    shutil.rmtree(local_model_path)

mlflow.sklearn.save_model(
    mlflow.sklearn.load_model(MODEL_PATH),
    local_model_path
)

print(f"Model downloaded to {local_model_path}")

# Register model in Azure ML
print(f"Registering model in Azure ML...")

model = Model.register(
    workspace=ws,
    model_name=MODEL_NAME,
    model_path=local_model_path,
    description="Wine Quality Random Forest Model",
    tags={
        'framework': 'scikit-learn',
        'type': 'RandomForest',
        'dataset': 'Wine Quality',
        'mlflow_run_id': RUN_ID
    }
)

print(f"\nModel registered successfully!")
print(f"Model Name: {model.name}")
print(f"Model ID: {model.id}")
print(f"Model Version: {model.version}")