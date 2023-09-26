import pandas as pd
from pathlib import Path
import numpy as np
import shap 
import joblib
import itertools
from mlflow.sklearn import load_model

BASE_DIR = Path(__file__).resolve(strict=True).parent

# On charge le modèle 
model = load_model(f"{BASE_DIR}/banking_model_20230915203320")


def get_impact_threshold_risque():
    ###
    #   Fonction retournant les études d'impact sur les risques selon threshold
    ###
    risque = pd.read_csv(f"{BASE_DIR}/Risque assessment.csv")

    return risque

def get_impact_threshold_clients():
    ###
    #   Fonction retournant les études d'impact sur le pourcentage de client selon threshold
    ###
    clients = pd.read_csv(f"{BASE_DIR}/Potentiel clients.csv")

    return clients

def get_feature_importance_model():
    ###
    #   Fonction retournant les valeurs SHAPs du modèle
    ###
    feature_importance = pd.read_csv(f"{BASE_DIR}/shap_values_model.csv")

    return feature_importance

def get_threshold():
    ###
    #    Fonction retournant le threshold optimal utilisé pour le modèle
    ###
    # On charge le seuil
    with open(f"{BASE_DIR}/banking_model_seuil_20230915203320.pkl", 'rb') as seuil:
        threshold = 1 - joblib.load(seuil)

    return threshold

def get_model():
    ###
    #    Fonction retournant le modèle utilisé 
    ###
    return model

def application_model(df, threshold_app):
    try:
        ###
        #    Fonction permettant d'appliquer le modèle au dataframe
        ###

        # On prédit les probabilité selon le modèle
        # Prédiction d'avoir un prêt
        df["proba_pred_pret"] = np.round(model.predict_proba(df)[:, 0], 2)
        # Prédiction de ne pas avoir un prêt
        df["proba_pred_non_pret"] = np.round(1 - df["proba_pred_pret"], 2)

        # Résultat selon le threshold du modèle établit. 
        # Si au dessus du threshold, la valeur = 0, ce qui correspond à l'obtention d'un prêt
        df["prediction"] = np.where(df["proba_pred_pret"] > threshold_app, 0, 1)
    
        df["prediction_pret"] = np.where(df["prediction"] == 1, "Non pret", "Pret")
    
        return df
    
    except Exception as e:
        
        # Si une erreur est présente, on retourne le dataframe d'origin
        print("Erreur dans le traitement")
    
        return df

def application_model_score(df, threshold_app):
    
    try:
        # On calcul un score selon si le client a obtenu un prêt ou pas
        # Ce score donne une autre appréciation des probabilités, plus parlant pour un consommateur
        min_value = threshold_app  
        max_value = threshold_app
    
        df.loc[df.prediction_pret == "Pret", "score"] = (df["proba_pred_pret"] - min_value) / (1 - min_value)
        df.loc[df.prediction_pret == "Non pret", "score"] = (1 - (df["proba_pred_pret"] / max_value)) * - 1
        df.loc[:, "score"] = round(df.loc[:, "score"], 4)
    
        lettres = ['a', 'b', 'c', 'd', 'e', 'f']
        signes = ['++', '+', '-', '--']

        # On génére 2 dictionnaires
        bon_clients = {}
        mauvais_clients = {}

        point_decr = (100 / ((len(lettres) * len(signes)) / 2)) / 100

        point = 1
        for l, s in itertools.product(lettres[:3], signes):
            key = l + s
            bon_clients[key] = round(point, 2)
            point -= point_decr
    
        point = 0
        for l, s in itertools.product(lettres[3:], signes):
            key = l + s
            mauvais_clients[key] = round(point, 2)*-1
            point += point_decr

        condition1 = df.prediction_pret == "Pret"
        condition2 = df.prediction_pret == "Non pret"

        for keys, items in bon_clients.items():
            df.loc[(condition1) & (df.score <= items), "score_note"] = keys

        for keys, items in mauvais_clients.items():
            df.loc[(condition2) & (df.score <= items), "score_note"] = keys


        df.loc[:, "score"] = round(df.loc[:, "score"]*100, 2)
    
        return df
    
    except Exception as e:
        
        # Si une erreur est présente, on retourne le dataframe d'origin
        print("Erreur dans le traitement")
    
        return df


def feature_importance_client(df): 
    try:
        ###
        #    Fonction permettant de mesurer l'importance des features pour le client
        ###
    
        # On charge l'objet explainer du modèle
        with open(f"{BASE_DIR}/explainer_model.pkl", 'rb') as explainer_file:
            explainer_model = joblib.load(explainer_file)

        features = model.named_steps["select_columns"].transform(df).columns
    
        x_train_preprocessed = model[:-1].transform(df[features])
    
        selected_cols = df[features].columns[model.named_steps["feature_selection"].get_support()]
    
        x_train_preprocessed = pd.DataFrame(x_train_preprocessed, columns = selected_cols)

        shap_values = explainer_model(x_train_preprocessed)
    
        df_sk_shape = pd.DataFrame({'SK_ID_CURR': df["SK_ID_CURR"].values, 'VALEUR_TOTALE': shap_values.values.sum(axis=1)})

        df_feat_shape = pd.DataFrame(shap_values.values, columns = selected_cols)

        df_shape_score = pd.concat([df_sk_shape, df_feat_shape], axis = 1)
    
        return df_shape_score

    except Exception as e:
        
        # Si une erreur est présente, on retourne le dataframe d'origin
        print("Erreur dans le traitement")
    
        return df