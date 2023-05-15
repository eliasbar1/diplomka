#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import tensorflow_text as tf_text
tf.config.set_visible_devices([], 'GPU')
from ModelDataPreparator import ModelDataPreparator

class HintModelPreparator():
    def __init__(self):
        SAVED_DATA='data/saved/'
        self.trl=np.load(SAVED_DATA+"trail.npy")
        self.task=np.load(SAVED_DATA+"task.npy")
        self.lay=np.load(SAVED_DATA+"layer.npy")
        self.leg=np.load(SAVED_DATA+"legends.npy")
        self.names=np.load(SAVED_DATA+"names.npy")
        
        self.trl=np.append(self.trl, 0)
        self.task=np.append(self.task, 0)
        self.lay=np.concatenate((self.lay, np.zeros((1,10400))))
        self.leg=np.append(self.leg, '')

        self.tf_trl=tf.one_hot(self.trl,self.trl.max()+1)
        self.tf_task=tf.one_hot(self.task,self.task.max())
        self.tf_lay=tf.convert_to_tensor(self.lay)

        self.preparator=ModelDataPreparator()
        
        self.names_dict=dict()
        for n in range(len(self.names)):
            self.names_dict[self.names[n]]=n
        
    def get_stats_for_cipher(self, trail, task_no, teams):
        stats=self.preparator.get_hint_sum(trail.df, task_no, trail.slug_to_assign, teams)
        times=self.preparator.get_time_sum(trail.df, task_no, trail.slug_to_assign, teams)

        d_st=[]
        d_st.extend(zip(stats, times))
        tf_st=tf.convert_to_tensor(d_st)
        return tf_st
    def make_name(self, t, task_no):
        t_shortcut=self.preparator.inv_trail_dict[t.trail_slug]
        num="_{:02d}".format(task_no)
        name=t_shortcut+num+".png"
        return name

    def get_team_x(self, trail, task_no, teams):
        name=self.make_name(trail, task_no)
        if name in self.names_dict:
            indx=self.names_dict[name]
        else:
            indx=len(self.names_dict)
        tf_st=self.get_stats_for_cipher(trail, task_no, teams)
        my_x=[
            tf.reshape(tf_st, (1, -1)),
            tf.reshape(self.tf_task[indx], (1, -1)),
            tf.reshape(self.tf_trl[indx], (1, -1)),
            tf.reshape(self.tf_lay[indx], (1, -1)),
            tf.reshape(self.leg[indx], (1, -1))
        ]
        return my_x