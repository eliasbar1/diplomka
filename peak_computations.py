#!/usr/bin/python
# -*- coding: utf-8 -*-

from scipy.stats import gaussian_kde
import numpy as np

def compute_peak(how_long):
    density = gaussian_kde(how_long)
    avg_max=max(how_long)
    xs = np.linspace(0, avg_max, 50000)
#    density.covariance_factor = lambda: .25
    density._compute_covariance()
    peak = xs[np.argmax(density(xs))]
    if peak == 0:
        return 1
    return peak


def times_on_cipher(cipher_num, df, slug_to_assign):
    teams=finished_cipher(cipher_num, df, slug_to_assign)
    how_long=df[(df['user_id'].isin(teams))
               &(df['task_id']==slug_to_assign[cipher_num])
               &(df['type']=='TASK_TIME')]['value'].values
    h=[]
    for i in how_long:
        h.append(i)
    return h
    
    
def get_peak(cipher_num, df, slug_to_assign):
    '''
    Parameters
    ----------
    cipher_num : int
        task is of the task
    df : pandas.dataframe
        data we want to use to get desired information
    slug_to_assign : dict
        pairing number of the task in order (first task, second task, ...) and id of the task
        key: value -> cipher number (1, 2, ...): task_id
        
    Returns
    -------
    float
        time peak for given task
    
    '''
    h=times_on_cipher(cipher_num, df, slug_to_assign)
    peak=compute_peak(h)
    return peak

def get_team_time(team, cipher, df):
    '''
    Parameters
    ----------
    team : int
        user id of the team
    cipher : int
        task id of the task
    df : pandas.dataframe
        data we want to use to get desired information
        
    Returns
    -------
    int
        time the team spent solving given task (in seconds)
    '''
    
    task=df[df['task_id']==cipher]
    t=task[task['user_id']==team]
    ret=t[t['type']=='TASK_TIME']['value'].values
    
    if len(ret)==0:
        return 0
    
    return ret[0]

def did_they_complete(team, cipher, df):
    they=df[df['user_id'] == team]
    finished=they[they['type']=='TASK_FINISHED']
    solved=finished['task_id'].unique()
    if cipher in solved:
        return True
    return False

def compute_peaks(df, slug_to_assign):
    '''
    Parameters
    ----------
    df : pandas.dataframe
        data we want to use to get desired information
    slug_to_assign : dict
        pairing number of the task in order (first task, second task, ...) and id of the task
        key: value -> cipher number (1, 2, ...): task_id
        
    Returns
    -------
    list[float]
        list of peaks for all tasks listed in slug_to_assign
    '''
    p=[]
    A=df[df['type']=='TASK_FINISHED']['task_id'].unique()
    for i in slug_to_assign:
        if i > len(A):
            break
        peak=get_peak(i, df, slug_to_assign)
        p.append(peak)
    return p

def finished_cipher(cipher_num, df, slug_to_assign):
    '''
    Parameters
    ----------
    cipher_num : int
        task id of the task
    df : pandas.dataframe
        data we want to use to get desired information
    slug_to_assign : dict
        pairing number of the task in order (first task, second task, ...) and id of the task
        key: value -> cipher number (1, 2, ...): task_id
        
    Returns
    -------
    pandas.dataframe
        user ids of all teams that finished given task
    
    '''
    return df[(df['type']=='TASK_FINISHED')&(df['task_id']==slug_to_assign[cipher_num])]['user_id']


def df_finished_cipher(cipher_num, df, slug_to_assign):
    '''
    Parameters
    ----------
    cipher_num : int
        task id of the task
    df : pandas.dataframe
        data we want to use to get desired information
    slug_to_assign : dict
        pairing number of the task in order (first task, second task, ...) and id of the task
        key: value -> cipher number (1, 2, ...): task_id
        
    Returns
    -------
    pandas.dataframe
        user ids of all teams that finished given task
    
    '''
    return df[(df['type']=='TASK_FINISHED')&(df['task_id']==slug_to_assign[cipher_num])][['user_id']]