#--------------------------------------------------------
# Vous exposerez votre modèle de prédiction sous forme d’une API qui permet de calculer la probabilité de défaut du client, 
# ainsi que sa classe (accepté ou refusé) en fonction du seuil optimisé d’un point de vue métier.

# FAST API
import pickle
import uvicorn
from fastapi import FastAPI, Request
import pandas as pd

MODEL_FILE = 'model_LGBM_Tuned_500cols.pkl'
model = pickle.load(open(f"{MODEL_FILE}", "rb"))

app = FastAPI()

@app.get('/')
def index():
    return {'message': 'Credit Scoring API'}

@app.post('/predict')
async def predict(request: Request):
    result = await request.json()
    df = pd.DataFrame.from_dict(result)
    
    # Use the loaded model to make predictions on the DataFrame
    prediction = model.predict(df)
    probability = model.predict_proba(df)
    print(prediction)
    print(probability)

    prediction_df = pd.DataFrame(prediction, columns=['y_pred'])
    probability_df = pd.DataFrame(probability, columns=['proba_classe_0', 'proba_classe_1'])

    # Return the predictions as a JSON response
    return {"prediction": prediction_df, "probability": probability_df}
