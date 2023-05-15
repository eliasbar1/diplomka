#!/usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
import os
from MyModel import MyModel

class HintModel(MyModel):
    def __init__(self, text_model_name, my_trainable=False, output_bias=None, optimizer='adam'):
        if output_bias is not None:
            output_bias = tf.keras.initializers.Constant(output_bias)
            
        tfhub_handle=self.select_text_model(text_model_name)
        tfhub_handle_encoder=tfhub_handle[0]
        tfhub_handle_preprocess=tfhub_handle[1]

        team_input=tf.keras.layers.Input(shape=(2,),name='team_stats')

        task_no_input=tf.keras.Input(shape=(12,),name='task_number')

        trail_input=tf.keras.Input(shape=(50,),name='trail_id')

        layer_input=tf.keras.Input(shape=(10400,),name='img_layer')
        
        
        legend_input = layers.Input(shape=(), dtype=tf.string, name='legend')
        preprocessing_layer = hub.KerasLayer(tfhub_handle_preprocess, name='preprocessing')
        encoder_inputs = preprocessing_layer(legend_input)
        encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=my_trainable, name='encoder')
        outputs = encoder(encoder_inputs)
        leg = outputs['pooled_output']
        leg = tf.keras.layers.Dense(128, activation='relu', name='c_classifier')(leg)                

        y=layers.Concatenate()([team_input, task_no_input, trail_input, layer_input, leg])
        y=layers.Dense(128, activation='relu')(y)
        y=layers.Dense(32, activation='relu')(y)
        y=layers.Dense(64, activation='relu')(y)
        y=layers.Dense(128, activation='relu')(y)
        y=layers.Dense(16, activation='relu')(y)

        y=layers.Dense(32, activation='relu')(y)
        y=layers.Dense(2, activation='softmax', bias_initializer=output_bias)(y)

        self.model = tf.keras.Model(inputs=[team_input, task_no_input, trail_input, 
                                            layer_input, legend_input], outputs = [y])
        self.model.compile(optimizer=optimizer,
                           loss='binary_crossentropy',
                           metrics=['accuracy'])
    
    def select_text_model(self, text_model_name):
        map_name_to_handle = {
            'bert':
                'https://tfhub.dev/tensorflow/bert_multi_cased_L-12_H-768_A-12/3',
            'roberta':
                'https://tfhub.dev/jeongukjae/xlm_roberta_multi_cased_L-12_H-768_A-12/1',
        }

        map_model_to_preprocess = {
            'bert':
                'https://tfhub.dev/tensorflow/bert_multi_cased_preprocess/3',
            'roberta':
                'https://tfhub.dev/jeongukjae/xlm_roberta_multi_cased_preprocess/1'
        }
        tfhub_handle_encoder = map_name_to_handle[text_model_name]
        tfhub_handle_preprocess = map_model_to_preprocess[text_model_name]

        print(f'Model selected           : {tfhub_handle_encoder}')
        print(f'Preprocess model auto-selected: {tfhub_handle_preprocess}')

        return [tfhub_handle_encoder, tfhub_handle_preprocess]
    
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
