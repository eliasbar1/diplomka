#!/usr/bin/python
# -*- coding: utf-8 -*-


import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers
from MyModel import MyModel

class ImageTextModelFusion(MyModel):
    def __init__(self, text_model_name, img_size, my_trainable=False, optimizer='adam', txt_dropout=True):
        
        self.text_model_name=text_model_name
        self.img_size=img_size


        tfhub_handle=self.select_text_model(text_model_name)
        tfhub_handle_encoder=tfhub_handle[0]
        tfhub_handle_preprocess=tfhub_handle[1]
        

        text_input = layers.Input(shape=(), dtype=tf.string, name='text')
        preprocessing_layer = hub.KerasLayer(tfhub_handle_preprocess, name='preprocessing')
        encoder_inputs = preprocessing_layer(text_input)
        encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=my_trainable, name='BERT_encoder')
        outputs = encoder(encoder_inputs)
        txt = outputs['sequence_output']        
        
        txt=layers.Conv1D(16, 3,padding='same', activation='relu')(txt)
        txt=layers.MaxPooling1D()(txt)
        txt=layers.Flatten()(txt)
        txt=layers.Dense(32, activation='tanh')(txt)
        if txt_dropout:            
            txt = tf.keras.layers.Dropout(0.1)(txt)

        img_input=layers.Input(shape=(img_size, img_size, 3), dtype=tf.int32, name='img_input')
        y=layers.Rescaling(1./255, input_shape=(img_size, img_size, 3))(img_input)
        y=layers.Conv2D(16, 3, padding='same', activation='relu')(y)
        y=layers.MaxPooling2D()(y)
        y=layers.Conv2D(32, 3, padding='same', activation='relu')(y)
        y=layers.MaxPooling2D()(y)
        y=layers.Conv2D(64, 3, padding='same', activation='relu')(y)
        y=layers.MaxPooling2D()(y)
        y=layers.Flatten()(y)
        img=layers.Dense(128, activation='relu')(y)

        y=layers.Concatenate()([img, txt])
        y=layers.Dense(2, activation='softmax')(y)

        self.model = tf.keras.Model(inputs=[text_input, img_input], outputs = [y])
        self.model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        
    
    def select_text_model(self, text_model_name):
        map_name_to_handle = {
            'bert_multi_cased_L-12_H-768_A-12':
                'https://tfhub.dev/tensorflow/bert_multi_cased_L-12_H-768_A-12/3',
            'roberta':
                'https://tfhub.dev/jeongukjae/xlm_roberta_multi_cased_L-12_H-768_A-12/1',
        }

        map_model_to_preprocess = {
            'bert_multi_cased_L-12_H-768_A-12':
                'https://tfhub.dev/tensorflow/bert_multi_cased_preprocess/3',
            'roberta':
                'https://tfhub.dev/jeongukjae/xlm_roberta_multi_cased_preprocess/1'
        }
        tfhub_handle_encoder = map_name_to_handle[text_model_name]
        tfhub_handle_preprocess = map_model_to_preprocess[text_model_name]

        print(f'Model selected           : {tfhub_handle_encoder}')
        print(f'Preprocess model auto-selected: {tfhub_handle_preprocess}')

        return [tfhub_handle_encoder, tfhub_handle_preprocess]