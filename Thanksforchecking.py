#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 16:01:22 2020

@author: base


The model can be loaded with 

pip install git+https://github.com/rcmalli/keras-vggface.git


"""


import tensorflow as tf
import cv2
from keras_vggface.utils import preprocess_input
import numpy as np

print(tf.version.VERSION)
if tf.__version__.startswith('1.15'):
    # This prevents some errors that otherwise occur when converting the model with TF 1.15...
    tf.enable_eager_execution() # Only if TF is version 1.15

#--------------------------------
# If I fail to send you the model, it can be loaded and save with the following
from keras_vggface.vggface import VGGFace

pretrained_model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')  # pooling: None, avg or max
#pretrained_model.summary()
pretrained_model.save("my_model.h5") #using h5 extension

#-----------------------

fullint=True
saved_model_dir='PATH_where you saved the model/'
modelo='my_model.h5'

Datentyp='int8'   #'int8' or 'uint8'

print(tf.version.VERSION)

if tf.__version__.startswith('2.'):
    converter = tf.compat.v1.lite.TFLiteConverter.from_keras_model_file(saved_model_dir + modelo) #works now also with TF2.x
    #converter = tf.lite.TFLiteConverter.from_keras_model(saved_model_dir + modelo) # Works only with TF1.x       
if tf.__version__.startswith('1.'):
    converter = tf.lite.TFLiteConverter.from_keras_model_file(saved_model_dir + modelo)  
converter.optimizations = [tf.lite.Optimize.DEFAULT]
def representative_dataset_gen():
  for _ in range(10):
    pfad='Path to a sample image'
    Beleb=cv2.imread(pfad)
    pixels = Beleb.astype('float32')#!!!!! This is wrong as there is no normalization. use tf.convert_image_dtype
    samples = np.expand_dims(pixels, axis=0)
    samples = preprocess_input(samples, version=2) 
    samples = samples.astype(Datentyp)#!!!!! This is wrong as there is no normalization. use tf.convert_image_dtype
    # Get sample input data as a numpy array in a method of your choosing.
    yield [samples]
converter.representative_dataset = representative_dataset_gen
if fullint:
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    if Datentyp=='int8':
        print('I am int8**')
        converter.inference_input_type = tf.int8  # or tf.uint8
        converter.inference_output_type = tf.int8  # or tf.uint8
    elif Datentyp=='uint8': 
        print('I am uint8')
        converter.inference_input_type = tf.uint8  
        converter.inference_output_type = tf.uint8  
    else:
        print('check your datatype. Should be int8 or uint8')
        
tflite_quant_model = converter.convert()
print('model quantized ')
open("quantized_modelh5-15_we_int8", "wb").write(tflite_quant_model)

interpreter = tf.lite.Interpreter(model_content=tflite_model_quant)
input_type = interpreter.get_input_details()[0]['dtype']
print('input: ', input_type)
output_type = interpreter.get_output_details()[0]['dtype']
print('output: ', output_type)