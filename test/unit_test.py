import pickle
import pandas as pd
import pytest
import numpy as np
import requests
from fastapi.testclient import TestClient

from LY_France_API_112024 import app

URL = "http://127.0.0.1:8000/"
client = TestClient(app)

model = pickle.load(open("model_LGBM.pkl", "rb"))
df = pd.read_csv("test/data/data_sample.csv")
df = df[df['TARGET'].notnull()]
feats = [f for f in df.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index']]
data_to_test = df[feats]


def test_value_prediction_model(model = model, data = data_to_test):
    prediction = model.predict(data)
    expected_array = np.array([1.0, 0.0])
    assert (prediction == expected_array).all()

def test_response_api():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Credit Scoring API"}
