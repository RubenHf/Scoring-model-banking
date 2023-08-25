# Import packages
from dash import Dash, html, dash_table, dcc, callback, Output, Input, State
import pandas as pd
import plotly.express as px
import copy

# Provide the correct absolute path to the CSV file
csv_path = "C:/Users/33664/Desktop/Data scientist formation/[Projets]/Projet 7/concatenate_files.csv"

# Read the CSV file using Pandas
data = pd.read_csv(csv_path)

# Incorporate data
df = copy.deepcopy(data[:100])

# Initialize the app - incorporate css
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)

# App layout
app.layout = html.Div([
    html.Div(className='row', children='My First App with Data, Graph, and Controls',
             style={'textAlign': 'center', 'color': 'black', 'fontSize': 30}),
    
    # Variables à afficher
    html.Div(className='row', children=[
        dcc.RadioItems(options=['DAYS_EMPLOYED_PERC', 'AMT_INCOME_TOTAL'],
                       value='AMT_INCOME_TOTAL',
                       inline=True,
                       style={'textAlign': 'left', 'color': 'black', 'fontSize': 15},
                       id='variables_affichage_buttons')
    ]),
    html.Div(className='row', children=[
        html.Label("Options d'affichages :"),
        dcc.RadioItems(options=['Non', 'Oui'],
                      value='Non',
                      inline=True,
                       style={'textAlign': 'left', 'color': 'black', 'fontSize': 15},
                       id='affichage_options_buttons')
    ]),
    
    # Options d'affichage
    html.Div(id='options_affichage_div', children=[
        dcc.Loading(id='options-loading', type='circle', children=[]),  # This will be populated conditionally
    ]),

    
    # Graph à afficher
    html.Div(className='row', children=[
        dcc.Graph(figure={}, id='histo-chart-final')
    ]),
    
    # Tableau à afficher
    html.Div(className='row', children=[
        dash_table.DataTable(data=df.to_dict('records'), page_size=10, style_table={'overflowX': 'auto'})
    ]),
    
    dcc.Store(id='options-affichage-store')  # Store the selected option value

])

# Callback to conditionally show/hide the "Options d'affichages" section
@app.callback(
    Output('options_affichage_div', 'children'),
    Input('affichage_options_buttons', 'value')
)
def update_options_affichage(selected_option):
    if selected_option == 'Oui':
        return [
            html.Label("Taille des axes et valeurs :"),
            dcc.Slider(
            id='font-size-slider',
            min=18,
            max=42,
            step=6,
            value=18,
            marks={i: str(i) for i in range(18, 43, 6)}),
        
            # Options for the color palette dropdown
            html.Label("Select Color Palette:"),
            dcc.RadioItems(
            id='color-palette-dropdown',
            options=[
                {'label': 'Default', 'value': 'plotly'},
                {'label': 'Colorblind-Friendly', 'value': 'colorblind'}
                # Add more palette options as needed
            ],
            value='plotly')
        ]
    else:
        return []

# Callback to store options
@app.callback(
    Output('options-affichage-store', 'data'),
    Input('font-size-slider', 'value'),
    Input('color-palette-dropdown', 'value')
)
def store_options(font_size, color_palette):
    return {'font_size': font_size, 'color_palette': color_palette}

# Add controls to build the interaction
@app.callback(
    Output('histo-chart-final', 'figure'),
    Input('variables_affichage_buttons', 'value'),
    Input('affichage_options_buttons', 'value'),
    Input('font-size-slider', 'value'),
    Input('color-palette-dropdown', 'value'),
    State('font-size-slider', 'value'),
    State('color-palette-dropdown', 'value'),
    prevent_initial_call=True
)
def update_graph(col_chosen, selected_option, font_size, color_palette, font_size_state, color_palette_state):
    if selected_option == 'Non':
        font_size = 18
    
    x_var = "CODE_GENDER"
    fig = px.histogram(df, x=x_var, y=col_chosen, histfunc='avg')
    
    if color_palette == 'colorblind':
        color_blind_hex = ['#d6f9cc', '#97989c', '#33926e', '#1c4226', '#1f3628']
        fig.update_traces(marker_color=color_blind_hex)
    else:
        fig.update_traces(marker_color=px.colors.qualitative.Plotly)
    
    fig.update_layout(
        xaxis_title=x_var,
        yaxis_title=col_chosen,
        font=dict(
            family='Arial',
            size=font_size,
            color='black')
    )
    
    return fig

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
