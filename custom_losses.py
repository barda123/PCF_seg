import numpy as np

from scipy.ndimage import morphology

from tensorflow as tf
from tensorflow.keras import backend as keras


def pixel_weight_balance(mask):
    '''image weighting - sets pixel weights to the inverse fraction they take up (i.e. if  of pixels are True, they will be weighted at 10'''
    

    npx = tf.size(mask)
    fracTrue = tf.reduce_sum(mask) / tf.cast(npx,dtype=tf.float32) 
    fracFalse = tf.subtract(1.,fracTrue)
    
    weights = fracFalse*mask + fracTrue*(1 - mask)
    
    #enforce a mean of 1
    weights = tf.divide(weights,tf.reduce_mean(weights))
    
    return weights

def binary_crossentropy_weight_balance(yTrue,yPred):
    
    '''a hack which weights the pixelwise binnry crossentropy according to the abundance of True/False in yTrue'''

    weights = pixel_weight_balance(yTrue)
    
    bce = keras.binary_crossentropy(yPred, yTrue)
    
    wcce = bce * weights
    
    return tf.reduce_mean(wcce)