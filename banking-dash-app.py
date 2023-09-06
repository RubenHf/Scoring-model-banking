# app.py
import dash
from dash import Dash, html, dash_table, dcc, callback, Output, Input, State

app = dash.Dash(__name__)
server = app.server
app.layout = html.Div([
    html.H1("Hello Heroku!"),
    dcc.Input(id='my-input', value='Hello, Heroku!', type='text'),
    html.Div(id='my-output')
])

@app.callback(
    Output('my-output', 'children'),
    Input('my-input', 'value')
)
def update_output(input_value):
    return f'You entered: {input_value}'

if __name__ == '__main__':
    app.run_server(debug=True)
