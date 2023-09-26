import sys
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
import pandas as pd
import numpy as np

# On importe les fonctions liées au modèle
from app.model.model import application_model, feature_importance_client, get_threshold, get_model, get_feature_importance_model
from app.model.model import get_impact_threshold_risque, get_impact_threshold_clients, application_model_score


project_path = r'C:\Users\33664\Desktop\Data scientist formation\[Projets]\Projet test'
sys.path.append(project_path)


app = FastAPI()

class DataframeIn(BaseModel):
    data:dict
class DataframeInThresholdIn(BaseModel):
    data:dict
    thresh:float
class DataframeOut(BaseModel):
    data:dict
class FloatOut(BaseModel):
    data:float

@app.get("/")

def home():
    return {"health_check": "OK"}

@app.post("/get_threshold", response_model=FloatOut)
def predict():

    return {"data": get_threshold()}

@app.post("/shap_model", response_model=DataframeOut)
def predict():
    prepared_df = {
        "data" : get_feature_importance_model().to_dict('list')
    }
    return prepared_df

@app.post("/risque_th_model", response_model=DataframeOut)
def predict():
    prepared_risque = {
        "data" : get_impact_threshold_risque().to_dict('list')
    }
    return prepared_risque

@app.post("/client_th_model", response_model=DataframeOut)
def predict():
    prepared_clients = {
        "data" : get_impact_threshold_clients().to_dict('list')
        }
    return prepared_clients


@app.post("/prediction", response_model=DataframeOut)
def predict(payload: DataframeInThresholdIn):

    # On transforme le dictionnaire en Dataframe
    input_df = pd.DataFrame(payload.data)

    input_df = input_df.replace(-.0123, np.nan)

    prepared_df = application_model(input_df, payload.thresh)
    
    prepared_df = prepared_df.fillna(-.0123)
   
    prepared_dict = {
        "data": prepared_df.to_dict('list')
    }
    # Check if the "data" field is present in the response
    if "data" not in prepared_dict:
        raise HTTPException(status_code=500, detail="Response is missing the 'data' field.")

    # Check if the "data" field is of the correct type (dict)
    if not isinstance(prepared_dict["data"], dict):
        raise HTTPException(status_code=500, detail="The 'data' field should be of type dict.")

    return prepared_dict 

@app.post("/scoring", response_model=DataframeOut)
def predict(payload: DataframeInThresholdIn):

    # On transforme le dictionnaire en Dataframe
    input_df = pd.DataFrame(payload.data)

    prepared_df = application_model_score(input_df, payload.thresh)
   
    prepared_dict = {
        "data": prepared_df.to_dict('list')
    }
    # Check if the "data" field is present in the response
    if "data" not in prepared_dict:
        raise HTTPException(status_code=500, detail="Response is missing the 'data' field.")

    # Check if the "data" field is of the correct type (dict)
    if not isinstance(prepared_dict["data"], dict):
        raise HTTPException(status_code=500, detail="The 'data' field should be of type dict.")

    return prepared_dict 

@app.post("/importance_client", response_model=DataframeOut)
def predict(payload: DataframeIn):
# On transforme le dictionnaire en Dataframe
    input_df = pd.DataFrame(payload.data)

    input_df = input_df.replace(-.0123, np.nan)

    prepared_df = feature_importance_client(input_df)
    
    prepared_df = prepared_df.fillna(-.0123)

    prepared_dict = {
        "data": prepared_df.to_dict('list')
    }
    # Check if the "data" field is present in the response
    if "data" not in prepared_dict:
        raise HTTPException(status_code=500, detail="Response is missing the 'data' field.")
    
    # Check if the "data" field is of the correct type (dict)
    if not isinstance(prepared_dict["data"], dict):
        raise HTTPException(status_code=500, detail="The 'data' field should be of type dict.")
    
    return prepared_dict 

    
# Run the app
if __name__ == '__main__':
    app.run(debug=False)