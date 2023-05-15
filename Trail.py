#!/usr/bin/python
# -*- coding: utf-8 -*-

from SetOfAssignments import SetOfAssignments
from data_manager import load_data, get_structure
from peak_computations import compute_peaks, did_they_complete, get_team_time

class Trail:
    is_online=False
    def __init__(self, trail_slug):
        self.df, self.struct=load_data(trail_slug)
        self.trail_slug=trail_slug
        self.slug_to_assign, self.sets=get_structure(self.struct)
        self.name=self.struct['info']['name']
        self.peaks=None
        self.has_legends=False
        self.teams=self.df['user_id'].unique()
        
        for my_set in self.struct['sets']:
            set_keys=my_set.keys()
            if 'is_bonus_set' in set_keys:
                if my_set['is_bonus_set']==False:
                    self.assignments_ids=my_set['assignments']
            else:
                self.assignments_ids=my_set['assignments']
        self.num_of_tasks=len(self.assignments_ids)
        self.df_users=self.df[self.df['type']=='GAME_STARTED'][['user_id']]
        self.df_users=self.df_users.reset_index(drop=True)
        self.df_users=self.df_users.reset_index().rename(columns={"index": "id"})
        self.df_finishers=self.finished_cipher(self.num_of_tasks)
        self.finishers=self.df_finishers['user_id']
        
    def get_peaks(self):
        if self.peaks==None:
            self.peaks = compute_peaks(self.df, self.slug_to_assign)
        return self.peaks        
    def add_legends(self, legends):
        self.legends=legends
        self.has_legends=True
        
    def finished_cipher(self, cipher_num):
        finishers = self.df[
            (self.df['type']=='TASK_FINISHED')&
            (self.df['task_id']==self.slug_to_assign[cipher_num])][['user_id']]
        finishers=finishers.reset_index(drop=True)
        finishers=finishers.reset_index().rename(columns={"index": "id"})
        return finishers