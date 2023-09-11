import dash
from dash import Dash, html, dash_table, dcc, callback, Output, Input, State, ctx, Patch
import dash_bootstrap_components as dbc
import pandas as pd
import plotly
import plotly.express as px
import plotly.subplots as sp
import copy
import base64
import io
import numpy as np
import itertools
import mlflow
import shap
import pickle
import time
from mlflow.sklearn import load_model


# On load le modèle 
model = load_model("banking_model_20230901135647")

proba_threshold = 0.42

# on charge l'objet explainer du modèle
with open('explainer_model.pkl', 'rb') as explainer_file:
    explainer_model = pickle.load(explainer_file)
# on charge les valeurs du modèle
feature_importance = pd.read_csv("shap_values_model.csv") 

def preparation_file(df, model):
    
    if "DAYS_BIRTH" in df.columns:
        df["ANNEES_AGE"] = (abs(df.DAYS_BIRTH) / 365.25)
        # On élimine l' ancienne variable
        df.drop(["DAYS_BIRTH"], axis = 1, inplace = True)
        
    if "ANNEES_LAST_PHONE_CHANGE" in df.columns:
        df["ANNEES_LAST_PHONE_CHANGE"] = round((abs(df.DAYS_LAST_PHONE_CHANGE) / 365.25), 2)
        # On élimine l' ancienne variable
        df.drop(["DAYS_LAST_PHONE_CHANGE"], axis = 1, inplace = True)
    
    # On corrige les erreurs 
    infinity_indices = np.where(np.isinf(df))
    for row_ind, col_ind in zip(*infinity_indices):
        df.iloc[row_ind, col_ind] = df.iloc[:,col_ind].median()
    
    # On récupère les features du modèle
    features = model.named_steps["select_columns"].columns
    
    return  df[features]

def scoring_pret(df, threshold):
    
    # On calcul un score selon si le client a obtenu un prêt ou pas
    # Ce score donne une autre appréciation des probabilités, plus parlant pour un consommateur
    min_value = 1 - threshold  # Minimum value of proba_pred_pret
    max_value = 1 - threshold
    
    df.loc[df.prediction_pret == "Pret", "score"] = (df["proba_pred_pret"] - min_value) / (1 - min_value)
    df.loc[df.prediction_pret == "Non pret", "score"] = (1 - (df["proba_pred_pret"]) / (max_value)) * - 1
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

def application_model(df, model, threshold):
    
    result_df = copy.deepcopy(df)
 
    # On prédit les probabilité selon le modèle
    # Prédiction d'avoir un prêt
    result_df["proba_pred_pret"] = np.round(model.predict_proba(result_df)[:, 0], 2)
    # Prédiction de ne pas avoir un prêt
    result_df["proba_pred_non_pret"] = np.round(1 - result_df["proba_pred_pret"], 2)
    
    # Résultat selon le threshold du modèle établit. 
    # Si au dessus, la valeur = 1, ce qui correspond à la non obtention d'un prêt
    result_df["prediction"] = np.where(result_df["proba_pred_non_pret"] >= threshold, 1, 0)
    
    result_df["prediction_pret"] = np.where(result_df["prediction"] == 1, "Non pret", "Pret")
    result_df = scoring_pret(result_df, threshold)

    return result_df

def feature_importance_client(df, model, explainer_model): 
    
    features = model.named_steps["select_columns"].columns
    
    x_train_preprocessed = model[:-1].transform(df[features])
    
    selected_cols = df[features].columns[model.named_steps["feature_selection"].get_support()]
    
    x_train_preprocessed = pd.DataFrame(x_train_preprocessed, columns = selected_cols)
    
    shap_values = explainer_model(x_train_preprocessed)
    
    df_sk_shape = pd.DataFrame({'SK_ID_CURR': df["SK_ID_CURR"].values, 'value_total': shap_values.values.sum(axis=1)})

    df_feat_shape = pd.DataFrame(shap_values.values, columns = selected_cols)

    df_shape_score = pd.concat([df_sk_shape, df_feat_shape], axis = 1)
    
    return df_shape_score


def figure_feature_client_dash(df_shape, df_client, nb_variable = 10, color_point = "plotly", size = 18):
    
    # on ajuste la couleur des barres du graphique
    if color_point == "colorblind":
        color_bar = "#648FFF"
        color_bar_selected = "#FFB000"
    else :
        color_bar = "blue"
        color_bar_selected = "red"
     
    figure_height = (df_shape.shape[0] * size)
    
    for i, client in df_shape.iterrows():

        figure_height = 200 + (len(client[2:]) * size+1)
        
        # on trie les variables selon leur influence sur le résultat et on ne garde que les nb_variables premières valeurs
        index = abs(client[2:]).sort_values(ascending=True)[-nb_variable:].index
        
        nomVar = str(f"SOMME DES VARIABLES RESTANTES ({(len(client) - 2 - nb_variable)})")
        # Ajout du score des variables non calculées
        client.loc[nomVar] = client[~client.index.isin(index)][2:].sum()
        
        # on ajoute sous la forme d'un index
        index = index.insert(0, pd.Index([nomVar]))
        
        
        # On fait -1 pour inverser la contribution
        # Négatif : Contribution négative à l'obtention du prêt
        # Positif : Contribution positif à l'obtention du prêt
        client[index] = client[index] * -1 
        
        # Pour afficher les valeurs sur le graphiques
        texte_liste = round(client[index], 2).astype(str).values
        # Permet de standardiser la taille du nom des variables
        texte_variable = index[1:].str[:20]
        
        texte_valeur = df_client[index[1:]].values[0]

        # on format
        texte_valeur = [round(x) if abs(x) > 5 else round(x, 2) for x in texte_valeur]

        
        # On ajoute la valeur de la variable au nom de la variable
        combined_texte = [f"{x} ({y})" for x, y in zip(texte_variable, texte_valeur)]

        combined_texte.insert(0, pd.Index([nomVar]))

        figure_var = px.bar(client[index], 
                        y=index, 
                        x=round(client[index], 2).values,
                        text=texte_liste,  
                        height=figure_height)

        # on met une couleur selon si la variable a une influence positive ou négative sur le score.
        colors = [color_bar_selected if client[col] <= 0 else color_bar for col in client[index].index]
        figure_var.update_traces(marker_color=colors)
        
        # On fait -1 pour inverser la contribution
        # Négatif : Contribution négative à l'obtention du prêt
        # Positif : Contribution positif à l'obtention du prêt
        shape_global = {'nom': ["Résultat global"], 'valeur': round(client[1] * -1, 2)}
        figure_global = px.bar(shape_global,
                        x='valeur',
                        text='valeur',
                        height=200)
        colors = [color_bar_selected if shape_global["valeur"] <= 0 else color_bar]
        figure_global.update_traces(marker_color=colors)
        
        # On crée un subplot de 2 lignes, avec une différence de taille entre les 2 figures
        fig_general = sp.make_subplots(rows=2, cols=1, row_heights=[9, 1], shared_xaxes=True,
                              subplot_titles=(f"Les 10 variables contribuant le plus <br>à la prédiction du client : {round(client[0])} ", 
                                              f"Valeur globale : {'Pas de prêt' if shape_global['valeur'] <= 0 else 'Prêt'}"),
                              vertical_spacing=0.05)
        
        fig_general.update_annotations(font=dict(size=8+size, color="black"))

        # On ajoute la première figure en haut
        for trace in figure_var.data:
            fig_general.add_trace(trace, row=1, col=1)

        # On ajoute la seconde en bas
        for trace in figure_global.data:
            fig_general.add_trace(trace, row=2, col=1)  

        fig_general.update_yaxes(ticktext=combined_texte, tickvals=list(range(nb_variable+1)), row=1, col=1)
        fig_general.update_yaxes(ticktext=[shape_global["nom"]], tickvals=[0], row=2, col=1)
        fig_general.update_layout(height=figure_height)

        fig_general.update_layout(
            title=dict(
                font=dict(size=8+size, color="black"), x=0.5, xanchor='center'),
            
            font=dict(size=size, color="black"),
            yaxis=dict(title = "",#"Variable(s) d'intérêt(s)",
                tickmode='array', tickvals=list(range(nb_variable + 1)),  
                dtick=1, automargin=False),
            margin=dict(l= 200 - ((1 - size / 36) * 30), r=20, t=150, b=20),# on va de 50 à 200 (18 à 36 en taille)
        )
            
        fig_general.add_shape(type="line", x0=0, x1=0, y0=-0.5, y1=nb_variable+0.5,
                            line=dict(color="black", width=2), row=1, col=1)
        fig_general.add_shape(type="line", x0=0, x1=0, y0=-0.5, y1=0.5,
                            line=dict(color="black", width=2), row=2, col=1)

    return fig_general
        
def return_dataframe_client(df, list_client):
    return df.loc[df.SK_ID_CURR.isin(list_client)]


def feature_importance_threshold(df, feats, importance, variance_importance = 0.9):

    # On récupère le total
    variance_total = df[importance].sum()
    
    # Ce qui va nous permettre d'identifier les variables à afficher
    variance_threshold = variance_total * variance_importance
    
    # On trie
    df.sort_values(importance, ascending = False, inplace = True)
    
    # On mesure la somme continue de l'importance, en partant de la variable la plus importante
    cumsum = df[importance].cumsum()
    
    # On ne sélectionne que les features
    cols_num = [cols_num for cols_num, val in zip(cumsum.index, cumsum) if val <= variance_threshold]
    
    sorted_df = copy.copy(df.loc[df.index.isin(cols_num), :]) 
    
    return sorted_df

def figure_feature_importance_dash(df, feats, importance, size, variance_importance = 0.9, color_point = "plotly", list_feat = None):
    ## 
    #    Fonction retournant un object de type figure
    #    On filtre les variables à afficher selon l'importance qu'ils ont sur le modèle (du plus important au moins)
    #    variance_importance = 0.5 : On affiche les variables qui ont 50 % du poids du modèle 
    #    variance_importance = 1 : On affiche l'ensemble des variables qui contribue au modèle
    ##
    
    # on ajuste la couleur des barres du graphique
    if color_point == "colorblind":
        color_bar = "#648FFF"
        color_bar_selected = "#FFB000"
    else :
        color_bar = "blue"
        color_bar_selected = "red"
        
    mask = feature_importance_threshold(df, feats, importance, variance_importance)
    

    mask.sort_values(by=importance, ascending = True, inplace = True)
    
    # Permet de standardiser la taille du nom des variables, selon la taille du texte
    mask["truncated"] = mask[feats].str[:(60 - size)]
    
    figure_height = 200 + (mask.shape[0] * (size+1))
    
    # On instancie un objet px.bar avec les features filtrées
    figure = px.bar(mask, 
                    y=feats, 
                    x=importance,
                    height=figure_height, 
                   )
        
    # Si on a une liste de features sélectionnées, on les met en avant
    
    if list_feat:
        colors = [color_bar_selected if f in list_feat else color_bar for f in mask[feats]]
        figure.update_traces(marker_color=colors)
    else : 
        figure.update_traces(marker_color=color_bar)
        
    figure.update_yaxes(ticktext=mask["truncated"], tickvals=list(range(len(mask["truncated"]))))
    
    figure.update_layout(
        title=dict(
            text = (
                    f"{mask.shape[0]} variable{'(s)' if mask.shape[0] > 1 else ''} "
                    f"contribue{'nt' if mask.shape[0] > 1 else ''} <br>à expliquer "
                    f"{round(variance_importance * 100)} % du modèle"
                ),
            font=dict(size=8+size, color="black"),
            x=0.5,  
            xanchor='center',
        ),
        margin=dict(l=200 - ((1 - size / 36) * 30), r=20, t=150, b=20),# on va de 50 à 200 (18 à 36 en taille)
        font=dict(size=size, color="black"),
        xaxis=dict(title = "Importance (valeur absolue)"),
        yaxis=dict(title = "",
            tickmode='array',  
            tickvals=list(range(mask.shape[0])),  
            dtick=1,  
            automargin=False),
    )
    # On retourne la figure
    return figure



def figure_score_dash(df, scoring_var, selected_client, color_discrete_map, point_size, font_size):
    
    if scoring_var == "score" :
        threshold = 0
        textH = threshold + 5
        textB = threshold - 5

    elif scoring_var == "proba_pred_pret":
        threshold = 1 - proba_threshold
        textH = threshold + 0.05
        textB = threshold - 0.05

    if scoring_var in ["score", "proba_pred_pret"]:
        figure_score = px.strip(df, y=scoring_var,
                    # On fait figurer le nom des clients et leur note
                    hover_data=["SK_ID_CURR", "score_note"],  
                    # On établit la coloration selon la condition de prêt 
                    color=df["prediction_pret"],  
                    color_discrete_map=color_discrete_map
                    # on actualise les valeurs de l'axe des x
                    ).update_xaxes(range=[-0.5, 0.5], tickvals=[0]  

                    # On ajoute une ligne horizontale
                    ).add_shape(type="line", x0=-1, x1=1, y0=threshold, y1=threshold,
                        line=dict(color="red", width=1)

                    # On ajoute un texte au dessus de la ligne
                    ).add_annotation(text="Threshold", x=-0.4, y=textH, showarrow=False,
                        font=dict(color="red", size=20)

                    # On ajoute un texte au dessus de la ligne              
                    ).add_annotation(text="Prêt", x=0.4, y=textH, showarrow=False,
                        font=dict(color="black", size=20)

                    # On ajoute un texte en dessous de la ligne
                    ).add_annotation(text="Non Prêt", x=0.4, y=textB, showarrow=False,
                        font=dict(color="black", size=20)

                    # On regroupe tout sur un seul tick
                    ).update_traces(offsetgroup=0.5)
        
        figure_score.update_traces(marker = dict(size=point_size))
        
        if selected_client is not None :
            selected_data_p = df[(df["SK_ID_CURR"].isin(selected_client)) & (df["prediction_pret"] == "Pret")]
            selected_data_np = df[(df["SK_ID_CURR"].isin(selected_client)) & (df["prediction_pret"] == "Non pret")]
        
            for selected_data in [selected_data_p, selected_data_np]:
                if not selected_data.empty:
                    trace = px.strip(selected_data, y=scoring_var, color=selected_data["prediction_pret"], 
                         color_discrete_map=color_discrete_map, hover_data=["SK_ID_CURR", "score_note"]
                        ).update_traces(marker_size=point_size+18, name="Sélectionné(s)", 
                                        marker_line_color="black", marker_line_width=2)
                else :
                    trace = px.strip()
                    
                figure_score.add_trace(trace.data[0]).update_traces(offsetgroup=0.5)
                

    else : 
        nb_pret = df[df.prediction_pret == 'Pret'].shape[0]
        nb_non_pret = df[df.prediction_pret == 'Non pret'].shape[0]
        nb_max = df["score_note"].value_counts().max()

        figure_score = px.histogram(df.sort_values("score_note"), x = "score_note",
                    color=df.sort_values("score_note")["prediction_pret"],  
                    color_discrete_map=color_discrete_map                       
                        # On ajoute un texte au dessus de la ligne              
                    ).add_annotation(text=f"Prêt (n={nb_pret})", x=1, y=nb_max, showarrow=False,
                        font=dict(color="black", size=20)

                    # On ajoute un texte en dessous de la ligne
                    ).add_annotation(text=f"Non prêt (n={nb_non_pret})", x=20, y=nb_max, showarrow=False,
                        font=dict(color="black", size=20))

        figure_score.update_xaxes(title=dict(
                    text = "Distribution des clients selont leur score note",
                    font=dict(size=20, color="black"),
                ))
        figure_score.update_yaxes(title=dict(
                    text = "Comptage (nombre de client(s))",
                    font=dict(size=20, color="black"),
                ))

    figure_score.update_layout(font=dict(size=font_size, color="black"))
    
    return figure_score


def figure_variable_dash(df, selected_variable, selected_client, selected_affichage, color_discrete_map, point_size, font_size):
    # Initialisation
    figure_variables = px.scatter()
    
    # On unpack
    x_variable, y_variable = selected_variable
    # Si une ou deux variables sont sélectionnées, on affiche soit un strip, soit un scatterplot
    if x_variable is not None and df is not None:
        hover_data = ["SK_ID_CURR", "score_note"]
        color_column = df["prediction_pret"]

        if y_variable is None:
            if selected_affichage == "strip":
                figure_variables = px.strip(df, y=x_variable, 
                                            hover_data=hover_data, 
                                            color=color_column, 
                                            color_discrete_map=color_discrete_map).update_traces(offsetgroup=0.5)
            elif selected_affichage == "boxplot":
                figure_variables = px.box(df, y=x_variable, 
                                          hover_data=hover_data, 
                                          color=color_column, 
                                          color_discrete_map=color_discrete_map)
        elif y_variable is not None:
            if selected_affichage == "scatter":
                figure_variables = px.scatter(df, x=x_variable, y=y_variable, 
                                              hover_data=hover_data, 
                                              color=color_column, 
                                              color_discrete_map=color_discrete_map)
            elif selected_affichage == "density":
                figure_variables = px.density_contour(df, x=x_variable, y=y_variable, 
                                                      hover_data=hover_data, 
                                                      color=color_column, 
                                                      color_discrete_map=color_discrete_map)
                
        if figure_variables:
            figure_variables.update_layout(font=dict(size=font_size, color="black"))
            if selected_affichage != "density":
                figure_variables.update_traces(marker=dict(size=point_size))
                    
        if selected_client is not None :
            selected_data_p = df[(df["SK_ID_CURR"].isin(selected_client)) & (df["prediction_pret"] == "Pret")]
            selected_data_np = df[(df["SK_ID_CURR"].isin(selected_client)) & (df["prediction_pret"] == "Non pret")]
            
            for selected_data in [selected_data_p, selected_data_np]:
                
                if not selected_data.empty:
                    
                    if y_variable is None:
                        trace = px.strip(selected_data, y=x_variable, color=selected_data["prediction_pret"], 
                        color_discrete_map=color_discrete_map, hover_data=hover_data
                        ).update_traces(marker_size=point_size+18, name="Sélectionné(s)", 
                                        marker_line_color="black", marker_line_width=2)

                    elif y_variable is not None:
                        trace = px.scatter(selected_data, x=x_variable, y=y_variable, color=selected_data["prediction_pret"], 
                                           color_discrete_map=color_discrete_map, hover_data=hover_data
                                          ).update_traces(marker_size=point_size+18, name="Sélectionné(s)", 
                                                          marker_line_color="black", marker_line_width=2)
                else :
                    trace = px.strip()
                if (y_variable is not None) or (selected_affichage == "boxplot"):
                    figure_variables.add_trace(trace.data[0])
                else:
                    figure_variables.add_trace(trace.data[0]).update_traces(offsetgroup=0.5)
        

    return figure_variables

def personnalized_describe(df):
    ##
    #        Permet de mesure plusieurs métrics du dataframe
    ##
    liste_statistique = ["Nombre d'entrées", "Remplissage", "Moyenne", "Mediane", "Ecart-type", "Maximum", "Minimun"]
    results = df.describe().transpose()
    results['Remplissage'] = (df.count() / df.shape[0] * 100).round(2).astype(str) + " %"
    
    results.rename(columns={'50%': 'Mediane', 'count': "Nombre d'entrées", 'mean' : "Moyenne", 'std':"Ecart-type", 'max':"Maximum",'min':"Minimun"},
                   inplace=True)
    results = round(results, 2)
    results.rename_axis(index='Statistiques', inplace=True)
    
    return results[liste_statistique].transpose()

def test_stat(groupe1, groupe2):
    
    # On réalise un t-test
    t_statistic, p_value = stats.ttest_ind(groupe1.dropna(), groupe2.dropna())

    # On prend un alpa = 0.05
    alpha = 0.05  

    # On test s'il y a une différence significative ou non entre les 2 groupes
    if p_value < alpha:
        result = "La moyenne des deux groupes est statistiquement différente"
    else:
        result = "La moyenne des deux groupes n'est pas statistiquement différente"
        
    p_value = "<0.001" if p_value < 0.001 else "{:.3f}".format(p_value)

    result = f"p_value : {p_value}, Résultat : {result}" 

    return result  

##### Initialize the app - incorporate css
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)

server = app.server
app.title = 'Modèle de prédiction'

temps_actuel = time.time()
# Temps d'inactivité
temps_inactivite = 5 * 60
interval_check = 5*60*1000 # 5 minutes

initial_size = 18
initial_point_size = 8
initial_palette = "plotly"

# Define the layout for the home page
home_page = html.Div([ 
    # Titre dashboard
    html.Div(className='row', children="Utilisation du modèle pour prédire l'obtention d'un prêt",
             style={'textAlign': 'center', 'color': 'black', 'fontSize': 48}),
    html.Hr(style={'border-top': '4px solid black'}), # Ligne horizontale
    
    # Upload liste de clients et leur caractéristiques
    html.Div(className='row',
        style={'textAlign': 'center', 'color': 'black', 'fontSize': 24}, children=[
            html.Div(className='row', children = "Chargez votre fichier clients ou le fichier test.csv"),
            # Boutton pour upload données 
            dcc.Upload(id="upload-data",children=[
                html.Button("Upload un fichier .csv", style={'color': 'black'})], multiple=False),
            html.Div(className='row', children="ou"),
            # Boutton pour upload les données tests
            html.Button("Upload test.csv", id="test_file_button", n_clicks=0, style={'color': 'black', 'margin-right': '10px'}),
            # Boutton pour upload les données tests
            html.Button(html.Strong("Effacer données"), id="del_file_button", n_clicks=0, style={'color': 'black'}),
    ]),   
    html.Hr(style={'border-top': '1px solid black'}), # Ligne horizontale
    
        # Choix des clients

    html.Div(className='client-selection', children=[
            html.Div("Visualisez les résultats de prédiction de prêt sur l'ensemble de vos clients (ou sélectionnés):",
                     style={'textAlign': 'center', 'color': 'black', 'fontSize': 30}),
            dcc.Dropdown(
                    value=None,
                    style={'textAlign': 'left', 'color': 'black', 'fontSize': 15},
                    placeholder="Sélection client(s)",
                    multi=True,
                    id='dropdown_client'
            ),
        ],
    ),

    # On affiche les 2 tableaux
    html.Div([
        html.Div("Visualisation des données", style={'float': 'center', 'textAlign': 'center', 'color': 'black', 'fontSize': 24, 'width': '50%'}),
        html.Div("Score de prédiction de l'obtention d'un prêt", style={'float': 'center', 'textAlign': 'center', 'color': 'black', 'fontSize': 24, 'width': '50%'}),
    ], style={'display': 'flex', 'flex-direction': 'row', 'width': '100%'}),
    
    html.Div([
        html.Div(id="table_client", style={'width': '50%','margin-right': '10px'}),
        html.Div(id="table_prediction", style={'width': '50%', 'margin-left': '50px'}),
    ], style={'display': 'flex', 'flex-direction': 'row', 'width': '100%'}),
    
    html.Div([
            html.Button("⇓ Desciption des données ⇓", id="describe-button", n_clicks = 1, style = {'display': 'none'}),
            html.Div(id="table_description"),
        ],  style={'color': 'black'}),

    html.Hr(style={'border-top': '1px solid black'}), # Ligne horizontale

        # Upload liste de clients et leur caractéristiques
    html.Div(className='row',
        style={'textAlign': 'center', 'color': 'black', 'fontSize': 24}, children=[
            html.Div(className='row', children = "Télécharger les résultats de la prédiction"),
            # Boutton pour download les résultats 
            html.Button("Result_prediction_all.csv", id="download-data_all-button", n_clicks = 0, style={'color': 'black', 'margin-right':'10px'}),
            html.Button("Result_prediction_client.csv", id="download-data_client-button", n_clicks = 0, style={'color': 'black'}),
            dcc.Download(id="download-data"),
    ]),  

    html.Hr(style={'border-top': '1px solid black'}), # Ligne horizontale


    html.Div(className="figure_client", children=[
        html.Div(className='row', children="Options d'affichages",
            style={'textAlign': 'center', 'color': 'black', 'fontSize': 30}),
        # Option pour la taille des polices, des points et couleurs graphiques
        html.Div([
            html.Label("Taille police (axes et valeurs)", style={'textAlign': 'center'}),
            dcc.Slider(id='font-size-slider', min=18, max=36, step=2, value=initial_size,
                marks={i: str(i) for i in range(18, 37, 2)})
        ], style={'width': '50%', 'float': 'left', 'margin': '0 auto'}),

        html.Div([
            html.Label("Taille points (graphique)", style={'textAlign': 'center'}),
            dcc.Slider(id='point-size-slider', min=4, max=16, step=2, value=initial_point_size,
                marks={i: str(i) for i in range(4, 17, 2)})
        ], style={'width': '50%', 'float': 'right', 'margin': '0 auto'}),

        html.Div([
            html.Label("Palette de couleurs :"),
            dcc.RadioItems(id='color-palette-dropdown',
                style={'color': 'black', 'fontSize': 15},
                value=initial_palette, inline=True,
                options=[
                    {'label': 'Défaut', 'value': 'plotly'},
                    {'label': 'Daltonien', 'value': 'colorblind'}
                ]
            ),
        ], style={'textAlign': 'center', 'width': '100%'}),
        html.Div([
            html.Button("RESET affichage", id='reset-button', n_clicks=0),
        ], style={'textAlign': 'center', 'width': '100%'})
    ]),

        html.Hr(style={'border-top': '1px solid black'}), # Ligne horizontale


    # Choix des variables à représenter
        html.Div("Afficher la distribution des scores et des variables définissant les clients",
         style={'textAlign': 'center', 'color': 'black', 'fontSize': 30}),
        html.Div([
            html.Label("Sélectionnez une métrique de score :", 
                style={'textAlign': 'center', 'color': 'black', 'fontSize': 15}),
            dcc.RadioItems(id='dropdown_scoring', 
                style={'textAlign': 'center', 'color': 'black', 'fontSize': 15},
                options=[
                    {'label': 'Score', 'value': 'score'},
                    {'label': 'Proba_pred_pret', 'value': 'proba_pred_pret'},
                    {'label': 'Score_note', 'value': 'score_note'}],
                value='score', inline=True)],
            style={'width': '50%', 'float': 'left'}),

        html.Div([
            html.Div([
                dcc.Dropdown(
                    value=None,
                    style={'textAlign': 'center', 'color': 'black', 'fontSize': 15},
                    placeholder="Sélection variable (abscisse)",
                    multi=False,
                    id='dropdown_variable_x'
                )
            ], style={'width': '50%', 'display': 'inline-block'}),

            html.Div([
                dcc.Dropdown(
                    value=None,
                    style={'textAlign': 'center', 'color': 'black', 'fontSize': 15},
                    placeholder="Sélection variable (ordonné)",
                    multi=False,
                    id='dropdown_variable_y'
                )
            ], style={'width': '50%', 'display': 'inline-block'}),
             
            html.Button("⇐ Echanger ⇒", id='exchange-button', n_clicks=0),
            
            dcc.RadioItems(id='dropdown_fig_type', 
                style={'textAlign': 'center', 'color': 'black', 'fontSize': 15},
                options=[
                    {'label': 'Strip', 'value': 'strip'},
                    {'label': 'Boxplot', 'value': 'boxplot'}],
                value='strip', inline=True)], style={'width': '50%', 'float': 'right'}),

        html.Div([
            html.Div([
            dcc.Graph(id="graph_pred", style={'height': '600px', 'width': '50%', 'float': 'left'}),

            dcc.Graph(id="graph_variables", style={'height': '600px', 'width': '50%', 'float': 'right'}),

            ], style={'display': 'flex', 'flex-direction': 'row', 'width': '100%'}),
            html.Div(children="", id='statistique-variables', style={'width': '100%', 'float': 'right'}),
            html.Hr(style={'border-top': '0px solid black'}), # Ligne horizontale
        ], style={'display': 'flex', 'flex-direction': 'column', 'width': '100%'}),
         
        html.Hr(style={'border-top': '1px solid black'}), # Ligne horizontale
    
        html.Div(className='row', children="Information sur le modèle et les clients",
            style={'textAlign': 'center', 'color': 'black', 'fontSize': 30}),
        
        html.Div([ 
            html.Label("Importances des variables pour le modèle (défaut = 0.9)", 
            style={'textAlign': 'center', 'color': 'black', 'fontSize': 18}),
            
            dcc.Slider(id='feature-importance-slider', min=0, max=1, step=0.05, value=0.9,
                marks={i: str(round(i, 2)) for i in np.arange(0, 1.01, 0.05)}),
        ], style={'width': '50%', 'float': 'left'}),
        
        html.Div([
            dcc.Dropdown(
                style={'textAlign': 'center','color': 'black','fontSize': 15,  'float': 'center'},
                placeholder="Sélection client parmis les clients sélectionnés (1 client maximum)",
                value=None, multi=False, id='dropdown_client_var'
            ),
        ], style={'width': '50%', 'float': 'right', 'margin': '0 auto'}),
        
        html.Div([
            dcc.Graph(id="graph_model", style={'width': '50%', 'float': 'left'}),
        
            dcc.Graph(id="graph_client", style={'width': '50%', 'float': 'right'}),
        ], style={'display': 'flex', 'flex-direction': 'row', 'width': '100%'}),
        
    # Nous permet de suivre l'activité utilisateur, si pas d'activité, élimination des données
    dcc.Interval(
        id='activity-interval',
        interval=interval_check, 
        n_intervals=0),
    dcc.Store(id='clear-screen', data=False),
    dcc.Store(id='time_session', data=None),
    dcc.Store(id='initialize_figure_model', data=False),  
    dcc.Store(id='fichier_utilisateur', data=None),
    dcc.Store(id='fichier_utilisateur_prediction', data=None),    

]),


# si nécessaire
page_2 = html.Div([
    dcc.Link("Retour à la page principale", href="/"),

    html.P("En cours de construction"),
])


app.layout = html.Div([
    dcc.Location(id='url', refresh=False),

    # Page content
    html.Div(id='page-content'),
])

# Callback to update the page content based on the URL
@app.callback(
    Output('page-content', 'children'),
    Input('url', 'pathname')
)
def display_page(pathname):
    if pathname == '/page_2':
        return page_2
    else:
        return home_page
        

@app.callback(    
    Output(component_id="fichier_utilisateur", component_property="data"),
    Output(component_id="fichier_utilisateur_prediction", component_property="data"),
    
    Input(component_id="test_file_button", component_property="n_clicks"),
    Input(component_id="upload-data", component_property="contents"),  
    Input(component_id="clear-screen", component_property="data"),
    Input(component_id="del_file_button", component_property="n_clicks"),
    
    prevent_initial_call=True,
)

# Téléchargement des fichiers
def gestion_files(n_clicks_load_file, content, clear_statut, n_clicks_):
            
    # On charge un fichier csv 
    if content is not None: 
        content_type, content_string = content.split(",")
        decoded = base64.b64decode(content_string)
        file_df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        
        # Plusieurs étapes de préparation du fichier
        file_df = preparation_file(file_df, model)
        file_df = application_model(file_df, model, proba_threshold)
        file_df_client_score = feature_importance_client(file_df, model, explainer_model)
        
        # On prépare pour stocker sur l'app
        fichier_utilisateur = file_df.to_json(date_format='iso', orient='split')
        fichier_utilisateur_score = file_df_client_score.to_json(date_format='iso', orient='split')

    
    # On charge le fichier test.csv (en local, livré avec l'application)
    elif ctx.triggered_id == "test_file_button":
        file_df = pd.read_csv("test.csv")
        
        # On mesure les différentes métriques
        file_df = application_model(file_df, model, proba_threshold)
        file_df_client_score = feature_importance_client(file_df, model, explainer_model)
        fichier_utilisateur = file_df.to_json(date_format='iso', orient='split')
        fichier_utilisateur_score = file_df_client_score.to_json(date_format='iso', orient='split')
                
    # on efface les données
    elif clear_statut == True :
        fichier_utilisateur = None
        fichier_utilisateur_score = None
        
    # on efface les données si on appuie sur le boutton effacer les données
    elif ctx.triggered_id == "del_file_button":
        fichier_utilisateur = None
        fichier_utilisateur_score = None
    
    else:
        fichier_utilisateur = dash.no_update
        fichier_utilisateur_score = dash.no_update
    
    
    return fichier_utilisateur, fichier_utilisateur_score


@app.callback(

    Output(component_id="dropdown_client", component_property="options"),
    Output(component_id="dropdown_variable_x", component_property="options"),
    Output(component_id="dropdown_variable_y", component_property="options"),
    Output(component_id="dropdown_fig_type", component_property="options"),
    Output(component_id="dropdown_client_var", component_property="options"),
    
    Output(component_id="dropdown_client", component_property="value"),
    Output(component_id="dropdown_variable_x", component_property="value"),
    Output(component_id="dropdown_variable_y", component_property="value"),
    Output(component_id="dropdown_fig_type", component_property="value"),
    Output(component_id="exchange-button", component_property="n_clicks"),

    Input(component_id="dropdown_client", component_property="value"),
    Input(component_id="dropdown_variable_x", component_property="value"),
    Input(component_id="dropdown_variable_y", component_property="value"),
    Input(component_id="dropdown_fig_type", component_property="value"),
    Input(component_id="dropdown_client_var", component_property="value"),   
    Input(component_id="feature-importance-slider", component_property="value"),   
    Input(component_id="clear-screen", component_property="data"),
    Input(component_id="exchange-button", component_property="n_clicks"),
    Input(component_id="fichier_utilisateur", component_property="data"),  
    
    prevent_initial_call=True,
)


def update_dropdown_menu(selected_client, selected_variable_x, selected_variable_y, 
                         selected_affichage, selected_client_variable, variance_importance, 
                         clear_statut, n_click_echange, fichier_utilisateur):
    

    
    if ctx.triggered_id == "exchange-button":
        # Remise à 0 du boutton
        n_click_echange = 0
        # On empêche l'actualisation des autres
        (option_drop_clients, option_drop_var_graph, option_client_variable,
                selected_client, selected_affichage) = [dash.no_update,
        dash.no_update,dash.no_update,dash.no_update,dash.no_update]
        
        # on vérifie qu'on a bien 2 variables d'entrées
        if selected_variable_y is not None :
            temp = selected_variable_x
            selected_variable_x = selected_variable_y
            selected_variable_y = temp
            
            # On réactualise la liste du drop
            feat_imp_threshold = feature_importance_threshold(feature_importance, "Features", "Importance", variance_importance).Features.values
            option_drop_var_x=[{'label': str(var), 'value': var} for var in [col for col in feat_imp_threshold]]
            option_drop_var_y=[{'label': str(var), 'value': var} for var in 
                               [col for col in feat_imp_threshold if col != selected_variable_x]]
        else :
            selected_variable_x, selected_variable_y = dash.no_update, dash.no_update
            option_drop_var_x, option_drop_var_y = dash.no_update, dash.no_update
            
        return (option_drop_clients, option_drop_var_x, option_drop_var_y, option_drop_var_graph, option_client_variable,
                selected_client, selected_variable_x, selected_variable_y, selected_affichage, n_click_echange)
    
    # Par défaut, si aucun fichier de chargé
    if (fichier_utilisateur is None) or (clear_statut is True):
        return ([], [], [], [], [], None, None, None, None, 0)
        
    elif fichier_utilisateur is not None :
        
        file_df = pd.read_json(fichier_utilisateur, orient='split')
      
        # On met à jour la liste de dropdown
        # La liste de variable est dépendante de la liste de feature importance

        feat_imp_threshold = feature_importance_threshold(feature_importance, "Features", "Importance", variance_importance).Features.values
        # Liste de variables (dépend des features liés au threshold du modèle)
        option_drop_var_x=[{'label': str(var), 'value': var} for var in [col for col in feat_imp_threshold]]
        option_drop_var_y=[]
        # Liste des clients selon le fichier loadé
        option_drop_clients=[{'label': str(client), 'value': client} for client in file_df['SK_ID_CURR']]      

        # On revient à la condition initiale
        selected_client = None if selected_client == [] else selected_client

        # On transforme en liste
        if (type(selected_client) != list) & (selected_client is not None):
            selected_client = [selected_client]

            # On place en premier ceux qui ont été sélectionnés
        option_client_variable = []

        if selected_client is not None:
            option_client_variable.extend([{'label': str(client), 'value': client} for client in selected_client])
            option_client_variable.extend([{'label': str(client), 'value': client} for client in file_df['SK_ID_CURR'] if client not in selected_client])
        else :     
            option_client_variable.extend([{'label': str(client), 'value': client} for client in file_df['SK_ID_CURR']])

        # Si aucune variable n'est sélectionnée, nous n'affichons rien
        if selected_variable_x == None:
            
            # Pas de valeur en ordonnée 
            selected_variable_y = None
            
            option_drop_var_graph = []
            selected_affichage = None

        # Si une ou deux variables sont sélectionnées, on affiche soit un strip, soit un scatterplot
        elif selected_variable_x is not None:
            # On met à jour la liste de variable, sans sélectionné la variable en x
            option_drop_var_y=[{'label': str(var), 'value': var} for var in 
                               [col for col in feat_imp_threshold if col != selected_variable_x]]
            
            if selected_variable_y is None :
                # On met à jour les options d'affichages pour une variable
                option_drop_var_graph=[
                        {'label': 'Strip', 'value': 'strip'},
                        {'label': 'Boxplot', 'value': 'boxplot'}]
                if(selected_affichage not in ["strip", "boxplot"]):
                    selected_affichage = "strip"
                
            else :
                # On met à jour les options d'affichages pour deux variables
                option_drop_var_graph=[
                        {'label': 'Scatterplot', 'value': 'scatter'},
                        {'label': 'Densité', 'value': 'density'}]
                if(selected_affichage not in ["scatter", "density"]):
                    selected_affichage = "scatter" 

                
        return (option_drop_clients, option_drop_var_x, option_drop_var_y, option_drop_var_graph, option_client_variable,
                selected_client, selected_variable_x, selected_variable_y, selected_affichage, n_click_echange)
    
@app.callback(
    Output(component_id="table_client", component_property="children"),
    Output(component_id="table_prediction", component_property="children"),

    Input(component_id="dropdown_client", component_property="value"),  
    Input(component_id="fichier_utilisateur", component_property="data"),
    
    prevent_initial_call=True,
)


def affichage_tab_clients(selected_client, fichier_utilisateur):
            
    if fichier_utilisateur is not None :
        # On prend la colonne ID client et les dernières colonnes liées à la prédiction crédit
        cols = [0] + list(range(-6, 0))
        
        file_df = pd.read_json(fichier_utilisateur, orient='split')
        
        if selected_client is None :
            table1 = dash_table.DataTable(
                data=file_df.iloc[:, :100].to_dict('records'),
                page_size=10,
                style_table={'overflowX': 'auto'}
            )
            table2 = dash_table.DataTable(
                data=file_df.iloc[:, cols].to_dict('records'),
                page_size=10,
                style_table={'overflowX': 'auto'}
            )
        else :
            filtered_df = file_df[file_df['SK_ID_CURR'].isin(selected_client)]
            table1 = dash_table.DataTable(
                data=filtered_df.iloc[:, :100].to_dict('records'),
                page_size=10,
                style_table={'overflowX': 'auto'}
            )
            table2 = dash_table.DataTable(
                data=filtered_df.iloc[:, cols].to_dict('records'),
                page_size=10,
                style_table={'overflowX': 'auto'}
            )
    else : 
        table1 = dash_table.DataTable()
        table2 = dash_table.DataTable()

    return table1, table2

@app.callback(
    Output(component_id="table_description", component_property="children"),
    Output(component_id="describe-button", component_property="n_clicks"),
    Output(component_id="describe-button", component_property="children"),
    Output(component_id="describe-button", component_property="style"),
    
    Input(component_id="describe-button", component_property="n_clicks"),
    Input(component_id="fichier_utilisateur", component_property="data"),
    
    State(component_id="dropdown_client", component_property="value"),  
    
    prevent_initial_call=True,
)
# Cleaning, effacement
def affichage_description(n_clicks, fichier_utilisateur, selected_client):
    
    if fichier_utilisateur is not None :
        
        button_visibilite = {'display': 'inline-block'}
        
        if n_clicks%2 == 0:
            file_df = pd.read_json(fichier_utilisateur, orient='split')

            # On réalise la description
            results_describe = personnalized_describe(file_df.iloc[:, 1:]).reset_index()
            table3 = dash_table.DataTable(
                data=results_describe.to_dict('records'),
                page_size=10,
                style_table={'overflowX': 'auto'}
            )
            texte_button = "⇑ Cacher la description ⇑"

        else :
            texte_button = "⇓ Desciption des données ⇓"
            table3 = dash_table.DataTable()
            n_clicks = 1
    else : 
        table3 = dash_table.DataTable()
        n_clicks = 1
        button_visibilite = {'display': 'none'}
        texte_button = dash.no_update

    return table3, n_clicks, texte_button, button_visibilite
    
    
@app.callback(
    Output(component_id="download-data", component_property="data"),
    
    Input(component_id="download-data_all-button", component_property="n_clicks"),
    Input(component_id="download-data_client-button", component_property="n_clicks"),
    
    State(component_id="dropdown_client", component_property="value"),
    State(component_id="fichier_utilisateur", component_property="data"),
    State(component_id="fichier_utilisateur_prediction", component_property="data"),
    prevent_initial_call=True,
)

def download_files(n_clicks_dl_file_all, n_clicks_dl_file_clients, selected_client, fichier_utilisateur, fichier_utilisateur_prediction):
    
    file_sent = None
    
    if fichier_utilisateur is not None :
        
        file_df = pd.read_json(fichier_utilisateur, orient='split')
        file_df_client_score = pd.read_json(fichier_utilisateur_prediction, orient='split')

        list_var_dl = ["SK_ID_CURR", "score", "score_note"]
        # On ajoute la contribution des variables au score (% selon la contribution totale)
        # Négative ou positive, selon l'outcome (Négatif si contribue à ne pas obtenir un prêt, positif inversement)

        # On mesure la contribution en pourcentage
        contribution_pourcentage = round(file_df_client_score.apply(lambda x : ((x[2:]) / abs(x[2:]).sum()) * - 100, axis = 1), 2)
        
        # On assemble les deux types d'informations
        dl_file = pd.concat([file_df[list_var_dl], contribution_pourcentage],  axis = 1)

        # On télécharge l'ensemble des données clients
        if ctx.triggered_id == "download-data_all-button":
            file_sent = dcc.send_data_frame(dl_file.to_csv, "Result_prediction_all.csv", index=False)
        
        # On télécharge les données des clients sélectionnés
        elif ctx.triggered_id == "download-data_client-button" :
            # Safegard pour éviter toute erreur de téléchargement
            if selected_client is not None:
                file_sent = dcc.send_data_frame(dl_file.loc[dl_file.SK_ID_CURR.isin(selected_client), :].to_csv, 
                                                "Result_prediction_client.csv", index=False)
            
    return file_sent



@app.callback(
    Output(component_id="graph_pred", component_property="figure"),
    Output(component_id="graph_variables", component_property="figure"),
    Output(component_id="graph_model", component_property="figure"),
    Output(component_id="graph_client", component_property="figure"),
    Output(component_id="initialize_figure_model", component_property="data"),
    
    Input(component_id="dropdown_scoring", component_property="value"),
    Input(component_id="dropdown_client", component_property="value"),
    Input(component_id="dropdown_variable_x", component_property="value"),
    Input(component_id="dropdown_variable_y", component_property="value"),
    Input(component_id="dropdown_fig_type", component_property="value"),
    Input(component_id="dropdown_client_var", component_property="value"),  
    Input(component_id="feature-importance-slider", component_property="value"),
    Input(component_id="color-palette-dropdown", component_property="value"),
    Input(component_id="point-size-slider", component_property="value"),
    Input(component_id="font-size-slider", component_property="value"), 
    Input(component_id="fichier_utilisateur", component_property="data"),
    Input(component_id="fichier_utilisateur_prediction", component_property="data"),
    
    State(component_id="initialize_figure_model", component_property="data"),
    #prevent_initial_call=True,
)

def figures_callback(scoring_choisie, selected_client, selected_variable_x, selected_variable_y,
                     selected_affichage, selected_client_variable, variance_importance,
                     color_point, point_size, font_size,
                    fichier_utilisateur, fichier_utilisateur_prediction, initialize_graph_model):    
    
    var_x_y = [selected_variable_x, selected_variable_y]
        
    # On défini des couleurs selon la condition
    if color_point == "plotly" :
        color_discrete_map = {"Pret": "blue", "Non pret": "red"}
    
    #Développé par IBM, source : https://davidmathlogic.com/colorblind/)
    elif color_point == "colorblind" :
        color_discrete_map = {"Pret": '#648FFF', "Non pret": '#FFB000'}
        
    # On prépare le graphique sur les features importances du modèle    
    if (initialize_graph_model == False) or (ctx.triggered_id == "feature-importance-slider"):        
        figure_feat_imp = figure_feature_importance_dash(feature_importance, "Features", "Importance", 
                                                         font_size, variance_importance, color_point, var_x_y)
        initialize_graph_model = True
    else : 
        figure_feat_imp = dash.no_update
    
    trigger_patch = ["color-palette-dropdown", "point-size-slider", "font-size-slider"]    
    
    if fichier_utilisateur is not None :
        file_df = pd.read_json(fichier_utilisateur, orient='split')
        # Seulement is on a initialisé une première fois les graphiques
        if ctx.triggered_id in trigger_patch :

            # On utilise patch pour ne devoir changer qu'un aspect de l'image
            patched_figure_score = Patch()
            patched_figure_variable = Patch()
            patched_figure_features_model = Patch()
            patched_figure_features_client = Patch()

            if ctx.triggered_id == "color-palette-dropdown":
                for patched in [patched_figure_score, patched_figure_variable, patched_figure_features_model, patched_figure_features_client]:

                    patched['data'][0]['marker']['color'] = color_discrete_map["Non pret"]
                    patched['data'][1]['marker']['color'] = color_discrete_map["Pret"]
                    patched['data'][3]['marker']['color'] = color_discrete_map["Non pret"]
                    patched['data'][2]['marker']['color'] = color_discrete_map["Pret"]

            elif ctx.triggered_id == "point-size-slider":
                for patched in [patched_figure_score, patched_figure_variable]:
                    for i in range(4):
                        patched['data'][i]['marker']['size'] = point_size if i < 2 else point_size + 18

            elif ctx.triggered_id == "font-size-slider":
                for patched in [patched_figure_score, patched_figure_variable, patched_figure_features_model, patched_figure_features_client]:
                    patched['layout']['font']['size'] = font_size
                    patched['layout']['title']['font']['size'] = font_size + 8 
                    if patched in [patched_figure_features_model, patched_figure_features_client]:
                        patched['layout']['margin']['l'] = 200 - ((1 - font_size / 36) * 30) # on va de 50 à 200 (18 à 36 en taille)  
                         
                patched_figure_features_client['layout']['annotations'][0]['font']['size'] = font_size + 8
                patched_figure_features_client['layout']['annotations'][1]['font']['size'] = font_size + 8

            return patched_figure_score, patched_figure_variable, patched_figure_features_model, patched_figure_features_client, initialize_graph_model 
        
        figure_score = dash.no_update
        figure_variable = dash.no_update
        figure_feature_importance_client = dash.no_update
        
        # fichier_utilisateur nous permet d'actualiser au chargement des données
        if ctx.triggered_id in ["dropdown_scoring", "fichier_utilisateur"]:
            figure_score = figure_score_dash(file_df, scoring_choisie, selected_client, color_discrete_map, point_size, font_size)
            
        elif ctx.triggered_id in ["dropdown_variable_x", "dropdown_variable_y", "dropdown_fig_type"]:
            figure_variable = figure_variable_dash(file_df, var_x_y, selected_client, selected_affichage, color_discrete_map, point_size, font_size)
            # On met à jour l'autre figure que dans la condition où une variable a été sélectionnée
            if ctx.triggered_id in ["dropdown_variable_x", "dropdown_variable_y"]:
                figure_feat_imp = figure_feature_importance_dash(feature_importance, "Features", "Importance", 
                                                         font_size, variance_importance, color_point, var_x_y)
                
        elif (ctx.triggered_id == "dropdown_client_var"):
            # on met à jour la figure car changement de client
            if selected_client_variable is not None :
                file_df_client_score = pd.read_json(fichier_utilisateur_prediction, orient='split')
                df1 = file_df_client_score[file_df_client_score.SK_ID_CURR == selected_client_variable]
                df2 = file_df[file_df.SK_ID_CURR == selected_client_variable]
                figure_feature_importance_client = figure_feature_client_dash(df1, df2, nb_variable = 10, color_point = color_point, size = font_size)            

            # on revient à aucun affichage
            else : 
                figure_feature_importance_client = px.bar() 

        elif (ctx.triggered_id == "dropdown_client") :
            figure_score = figure_score_dash(file_df, scoring_choisie, selected_client, color_discrete_map, point_size, font_size)
            figure_variable = figure_variable_dash(file_df, var_x_y, selected_client, selected_affichage, color_discrete_map, point_size, font_size)
        
            
    else :
        figure_score = px.scatter() 
        figure_variable = px.scatter()
        figure_feature_importance_client = px.bar()  
        
        if ctx.triggered_id in ["color-palette-dropdown", "font-size-slider"]     :
            figure_feat_imp = figure_feature_importance_dash(feature_importance, "Features", "Importance", 
                                                         font_size, variance_importance, color_point, var_x_y)

    return figure_score, figure_variable, figure_feat_imp, figure_feature_importance_client, initialize_graph_model


@app.callback(
    Output(component_id="statistique-variables", component_property="children"),
    
    Input(component_id="dropdown_variable_x", component_property="value"),
    Input(component_id="dropdown_variable_y", component_property="value"),
    
    State(component_id="fichier_utilisateur", component_property="data"),
)

def statistique_affichage(var_x, var_y, fichier_utilisateur):
    resultX = ""
    resultY = ""
    children = []
    
    if var_x is not None :
        file_df = pd.read_json(fichier_utilisateur, orient='split')
        resultX = test_stat(file_df.loc[file_df.prediction_pret == "Pret", var_x], file_df.loc[file_df.prediction_pret == "Non pret", var_x])
        resultX = f"Variable : {var_x}, {resultX}"  
        children.append(html.P(resultX))
        if var_y is not None :
            resultY = test_stat(file_df.loc[file_df.prediction_pret == "Pret", var_y], file_df.loc[file_df.prediction_pret == "Non pret", var_y]) 
            resultY = f"Variable : {var_y}, {resultY}"
            children.append(html.P(resultY))
        children.append(html.P())
    return children 
    

@app.callback(
    Output(component_id="color-palette-dropdown", component_property="value"),
    Output(component_id="point-size-slider", component_property="value"),
    Output(component_id="font-size-slider", component_property="value"),
    Input(component_id="reset-button", component_property="n_clicks"),
    Input(component_id="del_file_button", component_property="n_clicks"),
)
     
def reset_affichage(n_clicks_reset, n_click_del):
   # On reset les valeurs 
    return initial_palette,initial_point_size, initial_size

# Principaux bouttons ou options 
@app.callback(
    Output(component_id="clear-screen", component_property="data"),
    Output(component_id="time_session", component_property="data"),
              
    Input(component_id="test_file_button", component_property="n_clicks"),
    Input(component_id="exchange-button", component_property="n_clicks"),
    Input(component_id="download-data_all-button", component_property="n_clicks"),
    Input(component_id="download-data_client-button", component_property="n_clicks"),
    Input(component_id="dropdown_client_var", component_property="value"),
    Input(component_id="dropdown_client", component_property="value"),
    Input(component_id="dropdown_variable_x", component_property="value"),
    Input(component_id="dropdown_variable_y", component_property="value"),
    Input(component_id="dropdown_fig_type", component_property="value"),
    Input(component_id="dropdown_scoring", component_property="value"),
    Input(component_id="feature-importance-slider", component_property="value"),
    Input(component_id="color-palette-dropdown", component_property="value"),
    Input(component_id="point-size-slider", component_property="value"),
    Input(component_id="font-size-slider", component_property="value"), 
    Input(component_id="fichier_utilisateur", component_property="data"),    
    Input(component_id="del_file_button", component_property="n_clicks"),
    Input(component_id="reset-button", component_property="n_clicks"),
    
    Input(component_id="time_session", component_property="data"),
    Input(component_id="activity-interval", component_property="n_intervals"), 
)

# On ne prévient pas à l'initialisation, pour avoir un premier temps de référence
# Permet de surveiller l'activité utilisateur
# Permet d'avoir des sessions courtes

def inactivity(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11, arg12, arg13, arg14, arg15, arg16, arg17, temps_actuel, n_interval):
    
    # Initialisation
    if temps_actuel == None:
        temps_actuel = time.time()   
        
    elapsed_time = time.time() - temps_actuel
    clear_statut = False

    # Si il y a une activité utilisateur, on consigne le temps
    # On ne modifie pas le temps quand c'est activity-interval qui intervient
    if ctx.triggered_id != "activity-interval" and elapsed_time < (temps_inactivite):
        temps_actuel = time.time()
        
        return clear_statut, temps_actuel
    
    # Si l'utilisateur n'a rien réalisé dans les 5 - 10 dernières minutes, les données sont réinitialisées
    elif ctx.triggered_id == "activity-interval" :
        temps_actuel = time.time()
        
        if elapsed_time < temps_inactivite:
            
            return clear_statut, temps_actuel
        
        else:
            print("Temps expiré, It's clearing time")
            clear_statut = True

            return clear_statut, temps_actuel
        
# Run the app
if __name__ == '__main__':
    app.run(debug=False)