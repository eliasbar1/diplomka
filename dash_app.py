#!/usr/bin/python
# -*- coding: utf-8 -*-

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

from dash.dependencies import Input, Output

import datetime
from dash import Dash, dcc, html, dash_table, ctx
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc

from data_preparator import get_pts_for_coords, get_num_of_assignments, get_pts_for_cats, generate_table, generate_dims_coords
from Trail import Trail
from HintModelPreparator import HintModelPreparator
from HintModel import HintModel

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = Dash(__name__, external_stylesheets=external_stylesheets)
my_model=HintModel('roberta')
weights='data/weights/roberta_resampled/cp.ckpt'
my_model.load_weights_from_path(weights)

trails=[Trail("16_fantom-brna"), Trail("3_bitva-o-brno"), Trail("44_moravsky-manchester"), Trail("59_sedm-klicu")]
hm_prep=HintModelPreparator()
df = trails[1].df_users

table = generate_table(df, 'my_table')


slider = html.Div(
    [html.Label("Vyberte šifru", htmlFor="sifra"),
        dcc.Slider(
            id='slider',
            min=1,
            max=10,
            step=1,
            value=5,
            tooltip={"placement": "bottom", "always_visible": True})])


app.layout = html.Div(
    [
        dbc.Row(
            dbc.Col(
                html.H1(id = 'H1', children = 'Výsledky týmů na trase šifrovací hry', style = {'textAlign':'center',\
                                            'marginTop':40,'marginBottom':40}),
            )
        ),
        dbc.Row([
            dbc.Col(
                dcc.Dropdown( id = 'dropdown',
                    options = [
                        {'label': 'Bitva o Brno', 'value':1},
                        {'label':'Fantom Brna', 'value':0 },
                        {'label': 'Moravský Manchester', 'value':2},
                        {'label': 'Sedm klíčů', 'value':3}
                        ],
                    value = 1,
                    clearable=False ), width={"size": 3, "order": 1, "offset": 1}
            ),
            dbc.Col(
                slider, width={"size": 3, "order": 2, "offset": 1}
            )
        ]),
        dbc.Row(
            dbc.Col(
                dcc.Graph(id = 'coords_plot'),
            )
        ),
        dbc.Row(
            dbc.Col( html.Button('Resetovat zobrazení', id='reset_btn', n_clicks=0),
            )
        ),
        dbc.Row(
            dbc.Container([
                dbc.Alert(id='selected_team_out'),
                html.Div(table, id='table'),
            ])
        ),
        dcc.Store(id='selected_trail'),
        dcc.Store(id='selected_user'),
        dcc.Store(id='reset_graph'),
        dcc.Store(id='slider_max'),
    
    ]
)
    
@app.callback(Output(component_id='coords_plot', component_property= 'figure'),
              Output(component_id='table', component_property= 'children'),
              Output(component_id='selected_team_out', component_property='children'),
              Output(component_id='selected_user', component_property='data'),           
              [Input(component_id='slider', component_property= 'value'),
               Input(component_id='selected_trail', component_property= 'data'),
               Input(component_id='my_table', component_property='active_cell'),
               State(component_id='selected_user', component_property='data'),
               Input(component_id='reset_btn', component_property= 'n_clicks'),
               Input(component_id='reset_graph', component_property='data'),
               State(component_id='slider_max', component_property='data')])

def graph_update(slider_value, t, active_cell, selected_user, clicked, dropdown, slider_max):  
    trigger=ctx.triggered_id
    my_df, my_dims=get_pts_for_cats(trails[t], slider_value)
    color = my_df['finished']
    colorscale = [[0, 'lightsteelblue'], [1, 'mediumseagreen']]
    user_id=None
    selected_team_to_display='Kliknutím zvolte tým v tabulce.'
    users=trails[t].finished_cipher(slider_value)
    predict_for_task_no=slider_value
    
    if trigger=='reset_graph' or trigger=='reset_btn':
        fig = go.Figure(data=
            go.Parcats(
                line={'color': color, 'colorscale': colorscale, 'shape': 'hspline'},
                dimensions = my_dims,))        
        selected_user=None
        
    elif trigger=='my_table':
            row_id=active_cell['row_id']
            user_id=users[users['id']==row_id]['user_id'].values[0]
            selected_user=user_id
            team=my_df[my_df['teams']==selected_user]
            team_dims=generate_dims_coords(team, trails[t].assignments_ids, slider_value)
            fig= go.Figure(data=go.Parcoords(line_color='red', dimensions=team_dims))  
            
            print(slider_max)
            predict_for_task_no=slider_value+1
            
            if (predict_for_task_no>slider_max):
                predict_for_task_no = slider_max
            
            my_x=hm_prep.get_team_x(trail=trails[t], task_no=predict_for_task_no, teams=[user_id])
            p=my_model.model.predict_on_batch(my_x)
            user_prediction=p[0][1]
            selected_team_to_display='Zobrazen průchod týmu s id: '+str(user_id)+', pravděpodobnost, že si na šifře č. '+str(predict_for_task_no)+ ' vezme nápovědu, je '+ str(int(user_prediction*100))+'%.'
            
    elif trigger=='slider':
        if selected_user:
            team=my_df[my_df['teams']==selected_user]
            team_dims=generate_dims_coords(team, trails[t].assignments_ids, slider_value)
            fig= go.Figure(data=go.Parcoords(line_color='red', dimensions=team_dims))  
            predict_for_task_no=slider_value+1
            
            if (predict_for_task_no>slider_max):
                predict_for_task_no = slider_max

            my_x=hm_prep.get_team_x(trail=trails[t], task_no=predict_for_task_no, teams=[selected_user])
            p=my_model.model.predict_on_batch(my_x)
            user_prediction=p[0][1]
            selected_team_to_display='Zobrazen průchod týmu s id: '+str(selected_user)+', pravděpodobnost, že si na šifře č. '+str(predict_for_task_no)+ ' vezme nápovědu, je '+ str(int(user_prediction*100))+'%.'
        else:
            fig = go.Figure(data=
                go.Parcats(
                    line={'color': color, 'colorscale': colorscale, 'shape': 'hspline'},
                    dimensions = my_dims,)) 
        
    df = trails[t].finished_cipher(predict_for_task_no)
    table = generate_table(df, 'my_table', user_id)
    return fig, [table], selected_team_to_display,  selected_user

    
@app.callback([Output(component_id='reset_graph', component_property= 'data'),
               Output(component_id='slider', component_property= 'max'),
               Output(component_id='slider', component_property= 'value'),
               Output(component_id='selected_trail', component_property= 'data'),
               Output(component_id='slider_max', component_property='data')],
              [Input(component_id='dropdown', component_property= 'value')]
             )

def set_slider_range(dropdown_value):    
    num_of_ass=trails[dropdown_value].num_of_tasks
    print('zmena')
    return True, num_of_ass, num_of_ass, dropdown_value, num_of_ass


if __name__ == '__main__':
    app.run_server(debug=True, port =9988)
