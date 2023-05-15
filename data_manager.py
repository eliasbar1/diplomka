#!/usr/bin/python
# -*- coding: utf-8 -*-

import pandas as pd
import json
from SetOfAssignments import SetOfAssignments

def load_data(my_trail):
    '''
    Loads data from initial json files and constructs json of structure
    and dataframe of events.
    
    Parameters
    ----------
    my_trail : str
        name of the trail data file, such as bitva-o-brno
        
    Returns
    -------
    df : pandas.dataframe
        loaded events of the game - all timestamped events
    struct : json structure, dict
        structure of the game, name, sets, ...
    '''
    with open('data/original_trail_data/'+my_trail+'/events.json') as f:
        data=json.load(f)
        df=pd.json_normalize(data)    
    with open('data/original_trail_data/'+my_trail+ '/structure.json') as filatko:
        struct = json.load(filatko)
        
    return df, struct


def get_structure(struct):
    '''
    Splits structure of the game to game sets.
    
    Parameters
    ----------
    struct : json structure, dict
        game structure loaded from trail/structure.json
        
    Returns
    -------
    slug_to_assign : dict
        pairing number of the task in order (first task, second task, ...) and id of the task
        key: value -> cipher number (1, 2, ...): task_id
    my_sets : array<SetOfAssignments>
        array of game sets
    '''
    my_sets=[]
    if 'is_bonus_set' in struct['sets']:        
        for set_id, assignments, is_bonus, name, slug in ([(x['set_id'],
                                                            x['assignments'], 
                                                            x['is_bonus_set'], 
                                                            x['name'], 
                                                            x['slug']) for x in struct['sets']]):
            my_set=SetOfAssignments(set_id, name, slug, assignments, is_bonus)
            my_sets.append(my_set)
    else:
        for set_id, assignments, name, slug in ([(x['set_id'],
                                                 x['assignments'],  
                                                 x['name'], 
                                                 x['slug']) for x in struct['sets']]):
            my_set=SetOfAssignments(set_id, name, slug, assignments, False)
            my_sets.append(my_set)

    num=1
    slug_to_assign={}
    for i in range(len(my_sets)):
        for j in my_sets[i].assignments:
            slug_to_assign[num]=j
            num+=1
    return slug_to_assign, my_sets