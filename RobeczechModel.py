#!/usr/bin/python
# -*- coding: utf-8 -*-


import tensorflow as tf
from tensorflow.keras import layers, Input
from MyModel import MyModel
from transformers import AutoTokenizer, TFAutoModelForMaskedLM
import numpy as np
import os

class RobeczechModel(MyModel):
    def __init__(self, optimizer='adam'):
        self.tokenizer = AutoTokenizer.from_pretrained("ufal/robeczech-base")
        
        robeczech = TFAutoModelForMaskedLM.from_pretrained("ufal/robeczech-base", output_hidden_states=True)
        team_input=layers.Input(shape=(2,),name='team_stats')
        task_no_input=Input(shape=(12,),name='task_number')
        trail_input=Input(shape=(50,),name='trail_id')
        layer_input=Input(shape=(10400,),name='img_layer')       

        input_ids_in = layers.Input(shape=(128,), name='input_token', dtype='int32')
        input_masks_in = layers.Input(shape=(128,), name='masked_token', dtype='int32') 

        embedding_layer = robeczech(input_ids_in, attention_mask=input_masks_in)[0]
        embedding_layer=layers.AveragePooling1D()(embedding_layer)
        embedding_layer = layers.Flatten()(embedding_layer)
        
        y=layers.Concatenate()([team_input, task_no_input, trail_input, layer_input, embedding_layer])
        y=layers.Dense(64, activation='relu')(y)
        y=layers.Dense(16, activation='relu')(y)
        y=layers.Dense(32, activation='relu')(y)
        y=layers.Dense(2, activation='softmax')(y)

        self.model = tf.keras.Model(inputs=[team_input, task_no_input, trail_input, 
                                        layer_input, input_ids_in, input_masks_in], outputs = [y])

        self.model.compile(optimizer=optimizer,
                   loss='binary_crossentropy',
                   metrics=['accuracy'])

    '''
    vraci input ids, imput mask a segmenty    
    '''
    def tokenize(self, sentences):
        tokenizer=self.tokenizer
        input_ids, input_masks, input_segments = [],[],[]
        for sentence in sentences:
            inputs = tokenizer(sentence,
                               add_special_tokens=True,
                               max_length=128,
                               truncation=True,
                               padding="max_length",
                               return_attention_mask=True,
                               return_token_type_ids=True)
            input_ids.append(inputs['input_ids'])
            input_masks.append(inputs['attention_mask'])
            input_segments.append(inputs['token_type_ids'])        
        return np.asarray(input_ids, dtype='int32'), np.asarray(input_masks, dtype='int32'), np.asarray(input_segments, dtype='int32')
    
    def train_model(self, x_train, y_train, x_val, y_val, checkpoint_path, epochs=18, batch_size=64, class_weights=[1,1]):

        self.checkpoint_path = checkpoint_path
        checkpoint_dir = os.path.dirname(checkpoint_path)

        # Callback pro ukládání checkpointů
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                         save_best_only=True,
                                                         monitor='val_accuracy',
                                                         save_weights_only=True,
                                                         verbose=1)
        history = self.model.fit(x_train, y_train,
                         batch_size = batch_size, epochs=epochs, 
                         validation_data=(x_val, y_val), callbacks=[cp_callback], class_weight=class_weights)
        return history
