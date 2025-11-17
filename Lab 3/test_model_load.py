import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np

RUN_ID = "2a50f8bbefcb49ad8fbd959cb8ac53fe"

# Load the model
model = mlflow.sklearn.load_model(f"runs:/{RUN_ID}/random_forest_model")

# Create a sample input
sample_data = pd.DataFrame({
    'fixed acidity': [7.4],
    'volatile acidity': [0.70],
    'citric acid': [0.00],
    'residual sugar': [1.9],
    'chlorides': [0.076],
    'free sulfur dioxide': [11.0],
    'total sulfur dioxide': [34.0],
    'density': [0.9978],
    'pH': [3.51],
    'sulphates': [0.56],
    'alcohol': [9.4]
})

# Make prediction
prediction = model.predict(sample_data)
print(f"Prediction: {prediction[0]}")
print("Model loaded successfully!")