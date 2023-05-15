#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import tensorflow as tf


class MyModel():
    def train_model(self, x_train, y_train, x_val, y_val, checkpoint_path, epochs=18, batch_size=64):

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
                         validation_data=(x_val, y_val), callbacks=[cp_callback])
        return history

    def save_weights(self):
        self.model.save_weights(self.checkpoint_path)

    def load_weights(self):
        self.model.load_weights(self.checkpoint_path)
        
    def load_weights_from_path(self, checkpoint_path):
        self.model.load_weights(checkpoint_path)
        
    def set_checkpoint_path(self, chckp):
        self.checkpoint_path=chckp