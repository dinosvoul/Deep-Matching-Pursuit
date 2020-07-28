# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 15:11:26 2020

@author: Konstantinos
"""


import keras
import tensorflow as tf
from keras.layers import Input,Dense


from keras.models import Model


from keras.utils.generic_utils import get_custom_objects



def custom_activation(x, axis=-1):
    return tf.contrib.seq2seq.hardmax(x)*x


def deepmp(input_shape,SenMat,k):
    inputs = Input(shape=input_shape)

    r = inputs
    get_custom_objects().update({'custom_activation': custom_activation})

    for kk in range(k):
        if kk==0:
            
           
            denf1 = Dense(SenMat.shape[1],
                activation='custom_activation',
                trainable=True,
                use_bias=False,
                 weights=[SenMat]
                )
            denb1 = Dense(SenMat.shape[0],
                trainable=False,
                use_bias=False,
                weights=[SenMat.transpose()]
                )
            x=denf1(r)
            rx = denb1(x)
            r = keras.layers.subtract([r, rx])
            z=x
        else:
           
            denf1 = Dense(SenMat.shape[1],
                activation='custom_activation',
                 trainable=True,
                use_bias=False,
                  weights=[SenMat]
                )
            x=denf1(r)
            z=keras.layers.add([z,x])
            rx = denb1(x)
            r = keras.layers.subtract([r, rx])
    output = z    
    model = Model(inputs=inputs, outputs=output)
    return model