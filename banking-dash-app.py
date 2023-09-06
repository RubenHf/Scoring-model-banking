from dash import Dash, html, dash_table, dcc, callback, Output, Input, State
import pandas as pd
import plotly.express as px
import copy
import base64
import io
import numpy as np
import itertools
from mlflow.sklearn import load_model

proba_threshold = 0.42

force = np.random.rand(99)

feature_importance  = [[v, round(r, 2)] for v, r in zip(var, force)]
feature_importance = pd.DataFrame(feature_importance, columns = ["Features", "Importance"]).sort_values("Importance", ascending = True)

file_df = None

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
    
    return df[df.index.isin(cols_num)]

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
    
    # Permet de standardiser la taille du nom des variables, selon la taille du texte
    mask["truncated"] = mask[feats].str[:(60 - size)]
    
    figure_height = 200 + (mask.shape[0] * (size+1))
    
    # On instancie un objet px.bar avec les features filtrées
    figure = px.bar(mask.sort_values(importance, ascending = True), 
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
                    f"contribue{'nt' if mask.shape[0] > 1 else ''} à expliquer "
                    f"{variance_importance * 100} % du modèle"
                ),
            font=dict(size=8+size, color="black"),
            x=0.5,  # Center the title horizontally
            xanchor='center'
        ),
        font=dict(size=size, color="black"),
        yaxis=dict(
            tickmode='array',  
            tickvals=list(range(mask.shape[0])),  
            dtick=1,  
            automargin=True)
    
    )
    # On retourne la figure
    return figure

##### Initialize the app - incorporate css
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = Dash(__name__, external_stylesheets=external_stylesheets)


# App layout
app.layout = html.Div([
    
    # Titre dashboard
    html.Div(className='row', children="Utilisation du modèle pour prédire l'obtention d'un prêt",
             style={'textAlign': 'center', 'color': 'black', 'fontSize': 30}),
    html.Hr(), # Ligne horizontale
    
    # Upload liste de clients et leur caractéristiques
    html.Div(className='row',
        style={'textAlign': 'center', 'color': 'black', 'fontSize': 24}, children=[
            html.Div(className='row', children = "Chargez votre fichier clients ou le fichier test.csv"),
            # Boutton pour upload données 
            dcc.Upload(id="upload-data",children=[
                html.Button("Upload un fichier .csv", style={'color': 'black'})], multiple=False),
            html.Div(className='row', children="ou"),
            # Boutton pour upload les données tests
            html.Button("Upload test.csv", id="test_file_button", n_clicks=0, style={'color': 'black'}),
    ]),   

    html.Hr(), # Ligne horizontale
    
        # Choix des clients
    html.Div(className='row', children=[
        html.Div(className='client-selection', children=[
                html.Div("Choix d'un client :",
                         style={'textAlign': 'left', 'color': 'black', 'fontSize': 18}),
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
        html.Div(id="table_client"),
        html.Div(className='row', children="Score de prédiction de l'obtention d'un prêt",
             style={'textAlign': 'center', 'color': 'black', 'fontSize': 30}),
        html.Div(id="table_prediction"),
        
            # Upload liste de clients et leur caractéristiques
        html.Div(className='row',
            style={'textAlign': 'center', 'color': 'black', 'fontSize': 24}, children=[
                html.Div(className='row', children = "Télécharger les fichiers de prédictions"),
                # Boutton pour download les résultats 
                html.Button("Result_prediction_all.csv", id="download-data_all-button", n_clicks = 0, style={'color': 'black'}),
                html.Button("Result_prediction_client.csv", id="download-data_client-button", n_clicks = 0, style={'color': 'black'}),
                dcc.Download(id="download-data"),
        ]),  
        
        html.Hr(), # Ligne horizontale
        # On affiche les différents graphs liés au résultat du modèle
        html.Div(className="figure_client", children=[
            # Option pour la taille des polices et des points
            html.Label("Taille police (axes et valeurs)"),
            dcc.Slider(id='font-size-slider', min=18, max=36, step=2, value=18,
                marks={i: str(i) for i in range(18, 37, 2)}),
            html.Label("Taille points (graphique)"),
            dcc.Slider(id='point-size-slider', min=4, max=16, step=2, value=8,
                marks={i: str(i) for i in range(4, 17, 2)}),
            # Option pour la palette de couleur
            html.Label("Sélectionnez une couleur de palette :"),
            dcc.RadioItems(id='color-palette-dropdown',
                style={'textAlign': 'left', 'color': 'black', 'fontSize': 15},
                options=[
                    {'label': 'Défaut', 'value': 'plotly'},
                    {'label': 'Daltonien', 'value': 'colorblind'}],
                value='plotly', inline=True),
            html.Div(className='row', children="Représentation des clients :",
             style={'textAlign': 'center', 'color': 'black', 'fontSize': 30}),
            html.Label("Sélectionnez une métrique de score :"),
            dcc.RadioItems(id='variables_graph', 
                style={'textAlign': 'left', 'color': 'black', 'fontSize': 15},
                options=[
                    {'label': 'Score', 'value': 'score'},
                    {'label': 'Proba_pred_pret', 'value': 'proba_pred_pret'},
                    {'label': 'Score_note', 'value': 'score_note'}],
                value='score', inline=True),
            dcc.Graph(id="graph_pred", style={'height': '600px', 'width': '100%', 'float': 'left'}),
            html.Label("Importances des variables pour le modèle (défaut = 0.9)"),
            dcc.Slider(id='feature-importance-slider', min=0, max=1, step=0.05, value=0.9,
                marks={i: str(round(i, 2)) for i in np.arange(0, 1.01, 0.05)}),
            dcc.Graph(id="graph_model", style={'width': '100%', 'float': 'left'}),
            #dcc.Graph(id="graph_client", style={'height': '600px', 'width': '50%', 'float': 'right'}),
            
        ]),
        # Choix des variables à représenter
        html.Div(className='variable-selection', 
            children=[
                html.Div("Afficher la distribution des variables définissant les clients",
                         style={'textAlign': 'center', 'color': 'black', 'fontSize': 18}),
                dcc.RadioItems(id='dropdown_fig_type', 
                    style={'textAlign': 'left', 'color': 'black', 'fontSize': 15},
                    options=[
                        {'label': 'Strip', 'value': 'strip'},
                        {'label': 'Boxplot', 'value': 'boxplot'}],
                    value='strip', inline=True),
                dcc.Dropdown(
                        value=None,
                        style={'textAlign': 'left', 'color': 'black', 'fontSize': 15},
                        placeholder="Sélection variable(s) (2 maximum)",
                        multi=True,
                        id='dropdown_variable'
                ),
                
                dcc.Graph(id="graph_variables", style={'height': '600px', 'width': '100%', 'float': 'left'}),
                
            ],
        ),
    ]),  
])


@app.callback(
    Output(component_id="table_client", component_property="children"),
    Output(component_id="table_prediction", component_property="children"),
    Output(component_id="graph_pred", component_property="figure"),
    Output(component_id="dropdown_variable", component_property="options"),
    Output(component_id="dropdown_client", component_property="options"),
    Output(component_id="dropdown_fig_type", component_property="options"),
    Output(component_id="dropdown_fig_type", component_property="value"),
    Output(component_id="graph_variables", component_property="figure"),
    Output(component_id="graph_model", component_property="figure"),
    Output(component_id="test_file_button", component_property="n_clicks"),
    Output(component_id="download-data", component_property="data"),
    Output(component_id="download-data_all-button", component_property="n_clicks"),
    Output(component_id="download-data_client-button", component_property="n_clicks"),
    
    Input(component_id="test_file_button", component_property="n_clicks"),
    Input(component_id="dropdown_client", component_property="value"),
    Input(component_id="variables_graph", component_property="value"),
    Input(component_id="dropdown_variable", component_property="value"),
    Input(component_id="upload-data", component_property="contents"),    
    Input(component_id="font-size-slider", component_property="value"), 
    Input(component_id="point-size-slider", component_property="value"),
    Input(component_id="color-palette-dropdown", component_property="value"),
    Input(component_id="dropdown_fig_type", component_property="value"),
    Input(component_id="feature-importance-slider", component_property="value"),
    Input(component_id="download-data_all-button", component_property="n_clicks"),
    Input(component_id="download-data_client-button", component_property="n_clicks"),
    
    
)



def update_table_and_button(n_clicks_file, selected_client, scoring_choisie, selected_variable, content, font_size, point_size, color_point, selected_affichage, variance_importance, n_clicks_dl_file_all, n_clicks_dl_file_clients):
    
    # On défini des couleurs selon la condition
    if color_point == "plotly" :
        color_discrete_map = {"Pret": "blue", "Non pret": "red"}
    
    #Développé par IBM, source : https://davidmathlogic.com/colorblind/)
    elif color_point == "colorblind" :
        color_discrete_map = {"Pret": '#648FFF', "Non pret": '#FFB000'}
    
    global file_df
    # On charge le fichier 
    if content is not None: 
        content_type, content_string = content.split(",")
        decoded = base64.b64decode(content_string)
        file_df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
     
    # On charge le fichier test.csv
    if n_clicks_file == 1:
        file_df = pd.read_csv("test.csv")
        n_clicks_file = 0
        
    # Par défaut, si aucun fichier de chargé
    if file_df is None :
        table1, table2 = dash_table.DataTable(), dash_table.DataTable()
        figure_score, figure_variables, figure_feat_imp = px.scatter(), px.scatter(), px.bar()
        option_drop_var, option_drop_clients, option_drop_var_graph = [], [], []
        selected_affichage = None
        return table1, table2, figure_score, option_drop_var, option_drop_clients, option_drop_var_graph, selected_affichage, figure_variables, figure_feat_imp, None, n_clicks_dl_file
    
    elif file_df is not None :
        # On met à jour la liste de dropdown
        # La liste de variable est dépendante de la liste de feature importance
        
        feat_imp_threshold = feature_importance_threshold(feature_importance, "Features", "Importance", variance_importance).Features.values
        option_drop_var=[{'label': str(var), 'value': var} for var in [col for col in feat_imp_threshold]]
        #option_drop_var=[{'label': str(var), 'value': var} for var in [col for col in file_df.iloc[:, 1:100].columns]]
        option_drop_clients=[{'label': str(client), 'value': client} for client in file_df['SK_ID_CURR']]
        cols = [0] + list(range(-6, 0))

        if selected_client == []:
            selected_client = None

        if (type(selected_client) != list) & (selected_client is not None):
            selected_client = [selected_client]

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
        if scoring_choisie == "score" :
            threshold = 0
            textH = threshold + 5
            textB = threshold - 5

        elif scoring_choisie == "proba_pred_pret":
            threshold = 1 - proba_threshold
            textH = threshold + 0.05
            textB = threshold - 0.05
            
        if scoring_choisie in ["score", "proba_pred_pret"]:
            figure_score = px.strip(file_df, y=scoring_choisie,
                        # On fait figurer le nom des clients et leur note
                        hover_data=["SK_ID_CURR", "score_note"],  
                        # On établit la coloration selon la condition de prêt 
                        color=file_df["prediction_pret"],  
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
            
        else : 
            nb_pret = file_df[file_df.prediction_pret == 'Pret'].shape[0]
            nb_non_pret = file_df[file_df.prediction_pret == 'Non pret'].shape[0]

            figure_score = px.histogram(file_df.sort_values("score_note"), x = "score_note",
                        color=file_df.sort_values("score_note")["prediction_pret"],  
                        color_discrete_map=color_discrete_map                       
                            # On ajoute un texte au dessus de la ligne              
                        ).add_annotation(text=f"Prêt (n={nb_pret})", x=3, y=15, showarrow=False,
                            font=dict(color="black", size=20)

                        # On ajoute un texte en dessous de la ligne
                        ).add_annotation(text=f"Non prêt (n={nb_non_pret})", x=13, y=15, showarrow=False,
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
        
        # On revient à la condition initiale
        selected_variable = None if selected_variable == [] else selected_variable

        # Si aucune variable n'est sélectionnée, nous n'affichons rien
        if selected_variable == None:
            figure_variables = px.scatter()
            option_drop_var_graph = []
            selected_affichage = None

        # Si une ou deux variables sont sélectionnées, on affiche soit un strip, soit un scatterplot
        elif (type(selected_variable) == list) :
            if len(selected_variable) == 1 :
                
                # On met à jour les options d'affichages pour une variable
                option_drop_var_graph=[
                        {'label': 'Strip', 'value': 'strip'},
                        {'label': 'Boxplot', 'value': 'boxplot'}]
                if(selected_affichage not in ["strip", "boxplot"]):
                    selected_affichage = "strip"
                
                # Type d'affichage
                if selected_affichage == "strip":
                    figure_variables = px.strip(file_df, y=selected_variable[0],
                        # On fait figurer le nom des clients et leur note
                        hover_data=["SK_ID_CURR", "score_note"],  
                        # On établit la coloration selon la condition de prêt 
                        color=file_df["prediction_pret"],  
                        color_discrete_map=color_discrete_map
                        # On regroupe tout sur un seul tick
                        ).update_traces(offsetgroup=0.5)
                    
                elif selected_affichage == "boxplot":
                    figure_variables = px.box(file_df, y=selected_variable[0],
                        # On fait figurer le nom des clients et leur note
                        hover_data=["SK_ID_CURR", "score_note"],  
                        # On établit la coloration selon la condition de prêt 
                        color=file_df["prediction_pret"],  
                        color_discrete_map=color_discrete_map
                        )
            else :
                
                # On met à jour les options d'affichages pour deux variables
                option_drop_var_graph=[
                        {'label': 'Scatterplot', 'value': 'scatter'},
                        {'label': 'Densité', 'value': 'density'}]
                if(selected_affichage not in ["scatter", "density"]):
                    selected_affichage = "scatter"
                
                # Type d'affichage
                if selected_affichage == "scatter":
                    figure_variables = px.scatter(file_df, x=selected_variable[0], y = selected_variable[1],
                            hover_data=["SK_ID_CURR", "score_note"],  
                            color=file_df["prediction_pret"], color_discrete_map=color_discrete_map)
                else :    
                    figure_variables = px.density_contour(file_df, x=selected_variable[0], y = selected_variable[1],
                        hover_data=["SK_ID_CURR", "score_note"],  
                        color=file_df["prediction_pret"], color_discrete_map=color_discrete_map)
                    
            figure_variables.update_layout(font=dict(size=font_size, color="black"))
            if selected_affichage != "density":
                figure_variables.update_traces(marker=dict(size=point_size))
            

        if (type(selected_client) == list) and (selected_client is not None) and (scoring_choisie != "score_note"):
            # On met à jour les graphiques avec les clients sélectionnés
            selected_data_p = file_df[(file_df["SK_ID_CURR"].isin(selected_client)) & (file_df["prediction_pret"] == "Pret")]
            selected_data_np = file_df[(file_df["SK_ID_CURR"].isin(selected_client)) & (file_df["prediction_pret"] == "Non pret")]

            # En créant une nouvelle trace
            if not selected_data_p.empty:
                # Mise à jour du graphique score avec les clients sélectionnés
                selected_trace_p = px.strip(selected_data_p, y=scoring_choisie,  
                                 color=selected_data_p["prediction_pret"], color_discrete_map=color_discrete_map,
                                 hover_data=["SK_ID_CURR", "score_note"],
                                 ).update_traces(marker_size=point_size+18, name = "Sélectionné(s)",
                                                marker_line_color="black", marker_line_width=2)
                figure_score.add_trace(selected_trace_p.data[0]).update_traces(offsetgroup=0.5)

                # Mise à jour du graphique variables avec les clients sélectionnés
                if type(selected_variable) == list :
                    if len(selected_variable) == 1 :
                        selected_trace_p = px.strip(selected_data_p, y=selected_variable[0],  
                                 color=selected_data_p["prediction_pret"], color_discrete_map=color_discrete_map,
                                 hover_data=["SK_ID_CURR", "score_note"],
                                 ).update_traces(marker_size=point_size+18, name = "Sélectionné(s)",
                                                marker_line_color="black", marker_line_width=2)
                        figure_variables.add_trace(selected_trace_p.data[0])
                        if selected_affichage == "strip":
                            figure_variables.update_traces(offsetgroup=0.5)

                    else :
                        selected_trace_p = px.scatter(selected_data_p, x=selected_variable[0], y = selected_variable[1],  
                                 color=selected_data_p["prediction_pret"], color_discrete_map=color_discrete_map,
                                 hover_data=["SK_ID_CURR", "score_note"],
                                 ).update_traces(marker_size=point_size+18, name = "Sélectionné(s)",
                                                marker_line_color="black", marker_line_width=2)
                        figure_variables.add_trace(selected_trace_p.data[0])


            if not selected_data_np.empty:
                # Mise à jour du graphique score avec les clients sélectionnés
                selected_trace_np = px.strip(selected_data_np, y=scoring_choisie, 
                                 color=selected_data_np["prediction_pret"], color_discrete_map=color_discrete_map,
                                 hover_data=["SK_ID_CURR", "score_note"],
                                 ).update_traces(marker_size=point_size+18, name = "Sélectionné(s)",
                                                marker_line_color="black", marker_line_width=2)
                figure_score.add_trace(selected_trace_np.data[0]).update_traces(offsetgroup=0.5)

                # Mise à jour du graphique variables avec les clients sélectionnés
                if type(selected_variable) == list :
                    if len(selected_variable) == 1 :
                        selected_trace_np = px.strip(selected_data_np, y=selected_variable[0],  
                                 color=selected_data_np["prediction_pret"], color_discrete_map=color_discrete_map,
                                 hover_data=["SK_ID_CURR", "score_note"],
                                 ).update_traces(marker_size=point_size+18, name = "Sélectionné(s)",
                                                marker_line_color="black", marker_line_width=2)
                        figure_variables.add_trace(selected_trace_np.data[0])
                        if selected_affichage == "strip":
                            figure_variables.update_traces(offsetgroup=0.5)

                    else :
                        selected_trace_np = px.scatter(selected_data_np, x=selected_variable[0], y = selected_variable[1],  
                                 color=selected_data_np["prediction_pret"], color_discrete_map=color_discrete_map,
                                 hover_data=["SK_ID_CURR", "score_note"],
                                 ).update_traces(marker_size=point_size+18, name = "Sélectionné(s)",
                                                marker_line_color="black", marker_line_width=2)
                        figure_variables.add_trace(selected_trace_np.data[0])
        
        # On prépare le graphique sur les features importances du modèle
        
        figure_feat_imp = figure_feature_importance_dash(feature_importance, "Features", "Importance", font_size, variance_importance, color_point, selected_variable)        
        
        list_var_dl = ["SK_ID_CURR", "score", "score_note"]
        
        # On télécharge l'ensemble des données clients
        if n_clicks_dl_file_all > 0:
            file_sent = dcc.send_data_frame(file_df[list_var_dl].to_csv, "Result_prediction_all.csv")
            n_clicks_dl_file_all = 0
        # On télécharge les données des clients sélectionnés
        elif n_clicks_dl_file_clients > 0 :
            print(selected_client)
            # Safegard pour éviter toute erreur de téléchargement
            if selected_client is not None:
                file_sent = dcc.send_data_frame(file_df.loc[file_df.SK_ID_CURR.isin(selected_client), list_var_dl].to_csv, "Result_prediction_client.csv")
            n_clicks_dl_file_clients = 0
        else :
            file_sent = None
            
        return table1, table2, figure_score, option_drop_var, option_drop_clients, option_drop_var_graph, selected_affichage, figure_variables, figure_feat_imp, n_clicks_file, file_sent, n_clicks_dl_file_all, n_clicks_dl_file_clients 

port = int(os.environ.get("PORT", 8050))

if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port=port)