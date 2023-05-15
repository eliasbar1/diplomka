#!/usr/bin/python
# -*- coding: utf-8 -*-

import pytesseract
from os import listdir
import tensorflow as tf


def get_image_name(img_name):
    my_split = img_name.split(".")
    my_name=my_split[0]
    my_appendix=my_split[1]
    return my_name, my_appendix

def text_from_image(img_path):
    return pytesseract.image_to_string(img_path)
        
        
def text_from_images_in_folder(source_dir):
    texts=dict()
    for image in os.listdir(source_dir):
        my_name,_=get_image_name(image)    
        img_path=source_dir+"/"+image
        texts[my_name]=text_from_image(img_path)
    return texts


'''
Preparing data for image classification model.
'''
def prepare_input_for_model(txt, img, embedd):
    img=tf.convert_to_tensor(img)
    embedd=embedd.reshape((-1, 10240))
    embedd=tf.convert_to_tensor(embedd)
    return [txt, img, embedd]

def prepare_output_for_model(lbl):
    return tf.one_hot(lbl,2)