# GO2009-24ScriptDeInferênciaAzureMl
import json
import numpy as np
import joblib

def init():
    global model
    model_path = Model.get_model_path('iris-rf-model')
    model = joblib.load(model_path)

def run(raw_data):
    data = json.loads(raw_data)['data']
    predictions = model.predict(data)
    return predictions.tolist()
