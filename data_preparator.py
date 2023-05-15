#!/usr/bin/python
# -*- coding: utf-8 -*-

import pandas as pd
from dash import dash_table
import plotly.graph_objects as go

from Trail import Trail
from peak_computations import finished_cipher, get_team_time


def select_finishers(x, finishers):
    if x in finishers.values:
        return 1
    else:
        return 0


'''
trail_slug = e.g. "16_fantom-brna"
'''
def get_pts(trail):
    struct=trail.struct    
    for my_set in struct['sets']:
        set_keys=my_set.keys()
        if 'is_bonus_set' in set_keys:
            if my_set['is_bonus_set']==False:
                assignments_ids=my_set['assignments']
        else:
            assignments_ids=my_set['assignments']
    df=trail.df    
    finishers=finished_cipher(len(assignments_ids), trail.df, trail.slug_to_assign)
    users=df['user_id'].unique()
    
    teams_pts=dict()
    for a in assignments_ids:
        teams_pts[a]=[]
        for team in list(users):
            team_df=df[(df['task_id']==a)&(df['user_id']==team)]
            team_pts=team_df[team_df['type']=='TASK_POINTS']['value'].values
            if len(team_pts)<int(1):
                teams_pts[a].append(-1)
            else:
                teams_pts[a].append(int(team_pts[0]))
    teams_pts['teams']=users            
    pts_df=pd.DataFrame.from_dict(teams_pts)            
    pts_df['sum'] = pts_df.drop('teams', axis=1).sum(axis=1)

    pts_df['finished']=pts_df['teams'].apply(lambda x: select_finishers(x, finishers))
    num_of_ass=len(assignments_ids)
    return pts_df, num_of_ass, assignments_ids


def generate_dims_coords2(df, assignments_ids,last_task=0):
    ret=[]
    num_of_ass=len(assignments_ids)
    my_minus=num_of_ass-last_task
    stop=num_of_ass-my_minus
    for a in range(stop):
        r=dict(range = [0,10],
                 label = str(a+1), values = df[df.columns[a]],
                 tickvals=[0,5,10])
        ret.append(r)
    return ret

def generate_dims_coords(df, assignments_ids,last_task=0):
    ret=[]
    num_of_ass=len(assignments_ids)
    my_minus=num_of_ass-last_task
    stop=num_of_ass-my_minus
    for a in range(stop):
        r=dict(range = [0,10],
                 label = 'Šifra č. '+str(a+1), values = df[df.columns[a]],
                 tickvals=[0,3,5,8,10])
        ret.append(r)
    return ret

def generate_dims_cats(df, assignments_ids, last_task=0):
    ret=[]
    num_of_ass=len(assignments_ids)
    my_minus=num_of_ass-last_task
    stop=num_of_ass-my_minus
    for ass in range(stop):
        a=assignments_ids[ass]
        r=go.parcats.Dimension(label = 'Šifra č. '+str(ass+1), values = df[a], categoryorder='category descending')
        ret.append(r)
    r=go.parcats.Dimension(label = 'Dokončili hru', 
           values = df['finished'], 
           categoryarray=[1, 0], 
           ticktext=['Ano','Ne'])
    ret.append(r)
    return ret

def get_pts_for_coords(trail, last_task):
    pts, num_of_assignments, assignments_ids=get_pts(trail)
    dims=generate_dims_coords(pts, assignments_ids,last_task)
    return pts, dims

def get_pts_for_cats(trail, last_task):
    pts, num_of_assignments, assignments_ids=get_pts(trail)
    dims=generate_dims_cats(pts, assignments_ids, last_task)
    return pts, dims
    
    
def get_num_of_assignments(trail):
    return trail.num_of_tasks

def generate_table(df, table_id, input_user_id=None):
    table = dash_table.DataTable(
        data=df.to_dict('records'),
        columns=[{"name": i, "id": i} for i in df.columns if i != 'id'], 
        id=table_id,
        style_cell={'textAlign': 'center'},
    )
    return table

