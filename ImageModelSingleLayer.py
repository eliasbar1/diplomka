#!/usr/bin/python
# -*- coding: utf-8 -*-


import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers
from MyModel import MyModel

class ImageModelSingleLayer(MyModel):
    def __init__(self, text_model_name, img_size, my_trainable=False, optimizer='adam', txt_dropout=True):
        
        self.text_model_name=text_model_name
        self.img_size=img_size

        img_input=layers.Input(shape=(img_size, img_size, 3), dtype=tf.int32, name='img_input')
        y=layers.Rescaling(1./255, input_shape=(img_size, img_size, 3))(img_input)
        y=layers.Conv2D(16, 3, padding='same', activation='relu')(y)
        y=layers.MaxPooling2D()(y)
        y=layers.Dense(2, activation='softmax')(y)

        self.model = tf.keras.Model(inputs=[img_input], outputs = [y])
        self.model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        