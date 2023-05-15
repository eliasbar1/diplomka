#!/usr/bin/python
# -*- coding: utf-8 -*-

import json
import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf

from peak_computations import finished_cipher, get_team_time
from image_preprocess import get_image_name, text_from_image
from Trail import Trail


class ModelDataPreparator():
    def __init__(self, train_test_dir='data/output_resized', originals= 'data/output', labs=['easy', 'hard'], saved='data/saved/', data_dir='data/resized_images'):
        self.my_trails=dict()
        self.train_test_dir=train_test_dir
        self.originals= originals
        self.labs=labs
        self.saved=saved
        self.data_dir=data_dir
        
        # 'BOB': '3_bitva-o-brno'
        with open("data/trail_dict_shortcuts.json") as f:
            self.trail_dict=json.load(f)
        self.inv_trail_dict = {v: k for k, v in self.trail_dict.items()}
        
        # '0': '3_bitva-o-brno'
        with open("data/trail_dict.json") as f:
            self.trail_dict_nums=json.load(f)
        with open("data/extracted_texts.json") as f:
            self.texts=json.load(f)        
    
    def get_layer_output_from_img(self, image, text, original_img, embedding_model, intermediate_layer_model):    
        embedd=embedding_model.get_image_embedding([original_img], 1)
        t=torch.stack(embedd, dim=0)
        embedd=t.reshape((-1, 10240))
        e=embedd.detach().numpy()
        embedd=tf.convert_to_tensor(e)
        img=tf.convert_to_tensor(image)
        n_txt=np.array(text)
        x = [n_txt, img, embedd]
        intermediate_output = intermediate_layer_model.predict(x)
        return intermediate_output[0]

    
    # z názvu souboru získá jméno hry, aby bylo možné k němu dostat ostatní data
    def get_trail_from_img_name(self, img_name):
        my_split=img_name.split("_")
        trail_id=my_split[0]    
        if trail_id not in self.trail_dict.keys():
            return False, False, False

        # urci co je to za trail
        idx=list(self.trail_dict_nums.values()).index(self.trail_dict[trail_id])
        my_split1 = my_split[1].split(".")
        task_no = int(my_split1[0])
        if trail_id not in self.my_trails:
            trail_name=self.trail_dict[trail_id]
            trail = Trail(trail_name)
            print(trail.name)
            self.my_trails[trail_id]=trail

        return trail_id, task_no, idx

    # kolik nápověd si tým zatím vzal/kolik si mohl vzít
    def get_hint_sum(self, df, cipher_no, slug_to_assign, finished):    
        my_arr=[]
        for team in finished:
            team_data = df[df['user_id']==team]
            team_num=0
            cnt=0
            for t in slug_to_assign:
                my_t=team_data[(team_data['type']=='TASK_HINT')& (team_data['task_id']==slug_to_assign[t])]
                if t<cipher_no:
                    cnt+=2
                    team_num+=len(my_t)
            if(cnt>0):
                res = team_num/cnt
            else:
                res=team_num
            my_arr.append(res)
        return my_arr

    # vrací pole 1/0 pro každý z týmů, který vyřešil šifru, vzali/nevzali
    # toto je predikovaná veličina
    def get_hint_taking(self, df, cipher_no, slug_to_assign, finished):
        my_arr=[]
        for team in finished:
            team_task = df[df['user_id']==team]
            hint=team_task[team_task['type']=='TASK_HINT']
            if hint.shape[0]:
                my_arr.append(1)
            else:
                my_arr.append(0)
        return my_arr

    # průměrně strávený čas na předchozích šifrách
    def get_time_sum(self, df, cipher_no, slug_to_assign, finished):
        my_arr=[]
        for team in finished:
            team_num=0
            cnt=0
            for t in slug_to_assign:
                my_t=get_team_time(team, slug_to_assign[t], df)
                if t<cipher_no:
                    cnt+=1
                    team_num+=my_t
            if(cnt>0):
                res = team_num/cnt
            else:
                res=team_num
            my_arr.append(res)
        return my_arr

    def add_legend(self, trail_id):
        if self.my_trails[trail_id].has_legends:
            return
        trail_id_lower=trail_id.lower()
        with open("data/legends/" + trail_id_lower + "_legends.json") as f:
            my_leg=json.load(f)
            self.my_trails[trail_id].add_legends(my_leg)
    
    # returns concatenated legend for transition and for cipher

    def get_legend(self, trail_id, task_no):
        t=self.my_trails[trail_id]
        legends=t.legends
        legend=legends[str(task_no)]
        return legend[0] + " " + legend[1]
            
    def resample_data(self, st, trl, task, lay, h, leg):
        neg, pos = np.bincount(h)
        total = neg + pos

        print('Training examples:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n'.format(
            total, pos, 100 * pos / total))
        initial_bias = np.log([pos/neg])

        zeros_idx = np.argwhere(1 - h)
        rng = np.random.default_rng(seed=42)

        upsample_idx = rng.choice(zeros_idx, pos, replace=True)

        ones_idx = np.argwhere(h)
        my_idx = np.vstack([ones_idx, upsample_idx]).flatten()

        rng.shuffle(my_idx)

        st_up = st[my_idx]
        trl_up=trl[my_idx]
        task_up=task[my_idx]
        lay_up=lay[my_idx]
        h_up=h[my_idx]
        leg_up=leg[my_idx]

        return st_up, trl_up, task_up, lay_up, h_up, leg_up
    
    def create_datasets_from_folder(self, data_d, intermediate_layer_model, embedding_model, resample=False):
        
        TRAIN_TEST_DIR=self.train_test_dir
        TRAIN_TEST_ORIGINALS_DIR=self.originals
        LABELS=self.labs
        DATA_DIR=self.data_dir
        
        d_st=[]            # kolik nápověd si tým zatím vzal/kolik si mohl vzít a průměrně strávený čas na předchozích šifrách
        d_h=[]             # predikovaná veličina - vzal si tým nápovědu?
        d_task=[]          # číslo šifry - pořadí ve hře
        d_trl=[]           # co je to za hru
        d_lay_o=[]         # výstup předposlední vrstvy obrázkového modelu 
        d_leg=[]           # legenda k šifře
        img_names=[]       # názvy obrázků


        data_dir=TRAIN_TEST_DIR+data_d
        original_data_dir=TRAIN_TEST_ORIGINALS_DIR+data_d

        for label in range(len(LABELS)):
            for image in os.listdir(data_dir+'/'+LABELS[label]):
                trail_id, task_no, trail_idx = self.get_trail_from_img_name(image)
                if not trail_id:
                    continue
                self.add_legend(trail_id)
                t = self.my_trails[trail_id]
                df = t.df
                finished = finished_cipher(task_no, df, t.slug_to_assign)

                stats=self.get_hint_sum(df, task_no, t.slug_to_assign, finished)
                hints=self.get_hint_taking(df, task_no, t.slug_to_assign, finished)
                times=self.get_time_sum(df, task_no, t.slug_to_assign, finished)

                d_st.extend(zip(stats, times))
                d_h.extend(hints)
                d_task.extend([task_no]*len(stats))
                d_trl.extend([trail_idx]*len(stats))
                img_names.extend([image]*len(stats))

                rgb_img=[]
                rgb_img.append(np.array(Image.open(DATA_DIR+'/'+LABELS[label]+"/"+image).convert("RGB")))
                txt=[]

                name, _=get_image_name(image)
                txt.append(self.texts[name])

                original_img=plt.imread(original_data_dir+"/"+LABELS[label]+"/"+image)

                d_lay_o.extend([self.get_layer_output_from_img(rgb_img, txt, original_img, embedding_model, intermediate_layer_model)]*len(stats))

                leg = self.get_legend(trail_id, task_no)
                d_leg.extend([leg]*len(stats))

        n_st=np.array(d_st)
        n_trl=np.array(d_trl)
        n_task=np.array(d_task)
        n_lay_o=np.array(d_lay_o)
        n_h=np.array(d_h)
        n_leg=np.array(d_leg)
        n_img_names=np.array(img_names)

        if resample:
            return self.resample_data(n_st, n_trl, n_task, n_lay_o, n_h, n_leg)

        return n_st, n_trl, n_task, n_lay_o, n_h, n_leg, n_img_names    
    
    def load_datasets(self):
        SAVED_DATA=self.saved
        tr_st=np.load(SAVED_DATA+"tr_stats_updated.npy")
        val_st=np.load(SAVED_DATA+"val_stats_updated.npy")
        tr_trl=np.load(SAVED_DATA+"tr_trail.npy")
        val_trl=np.load(SAVED_DATA+"val_trail.npy")
        tr_task=np.load(SAVED_DATA+"tr_task.npy")
        val_task=np.load(SAVED_DATA+"val_task.npy")    
        tr_lay=np.load(SAVED_DATA+"tr_layer.npy")
        val_lay=np.load(SAVED_DATA+"val_layer.npy")    
        tr_h=np.load(SAVED_DATA+"tr_hints.npy") 
        val_h=np.load(SAVED_DATA+"val_hints.npy")     
        tr_leg=np.load(SAVED_DATA+"tr_legends.npy") 
        val_leg=np.load(SAVED_DATA+"val_legends.npy") 
        return tr_st, val_st, tr_trl, val_trl, tr_task, val_task, tr_lay, val_lay, tr_leg, val_leg, tr_h, val_h
  
    
    