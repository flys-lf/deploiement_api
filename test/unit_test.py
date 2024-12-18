# import requests
import pickle
import pandas as pd
import pytest
import numpy as np

# URL = "http://127.0.0.1:8000/"
# URL = "https://apitestscoring-bre0d5dbasdsewhw.francecentral-01.azurewebsites.net/predict"
model = pickle.load(open("model_LGBM_Tuned_500cols.pkl", "rb"))
model = model._final_estimator

df = pd.read_csv("test/data/data_sample.csv")
df = df[df['TARGET'].notnull()]
feats = [f for f in df.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index']]
data_to_test = df[feats]


def test_value_prediction_model(model = model, data = data_to_test):
    prediction = model.predict(data)
    expected_array = np.array([1.0, 0.0])
    assert (prediction == expected_array).all()

# def test_dummy_get(url_to_test=URL):
#     rep = requests.get(url_to_test)
#     rep_statut = rep.status_code
#     rep = rep.text
#     result = {"text": rep, "statut": rep_statut}
#     assert rep_statut == 200
#     assert rep == """{"message":"Credit Scoring API"}"""

# requests.post(url=f"{URL}predict", json={"text": "test"})


# def test_type_response(url=f"{URL}predict", data = data_to_test):
#     payload = data.to_json()
#     response = requests.post(url, data= payload)
#     result = response.json()
#     assert isinstance(result, dict)

# def test_value_prediction_api(url=f"{URL}/predict", data = data_to_test):
#     payload = data.to_json()
#     response = requests.post(url, data= payload)
#     result = response.json()
#     prediction_df = pd.DataFrame.from_dict(result["prediction"])
#     expected_series = pd.Series([np.float64(1.0), np.float64(0.0)])
#     prediction_series = prediction_df['y_pred'].reset_index(drop=True)
#     assert prediction_series.equals(expected_series)
