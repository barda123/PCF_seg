import numpy as np

from scipy.ndimage import morphology

import tensorflow as tf
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



def binary_crossentropy_weight_dict(weightDict):

    '''returns a  function for calculating weights, but DOESN'T WORK YET '''    
    
    def weight_from_dict(yTrue,weightDict):
    
        '''returns the pixelwise weights based on the dict'''

        weights = tf.zeros_like(yTrue,dtype='float64')

        for yVal in weightDict.keys():

            weights += tf.cast(tf.math.equal(yTrue,yVal),'float64')*weightDict[yVal]

        return yTrue

    
    def loss(yTrue,yPred):
    
        weights = weight_from_dict(yTrue,weightDict)
    
        bce = keras.binary_crossentropy(yPred, yTrue)
    
        wce = bce * weights
        
        return tf.reduce_mean(wce)
        
    return loss




def closeness_to_foreground_balanced(mask,sigma=20):
    
    '''takes a boolean image (mask) and computes a weight map for it, such that the classes are balanced, and there is an exponential smoothing moving away from the forground pixels.'''
    
    distance = morphology.distance_transform_edt(1-mask)
    
    closeness = np.exp(-distance/sigma) #which will be between 0 and 1
    
    #get class imbalance, assuming that there are less +ve pixels
    imbalance = tf.cast(tf.size(mask),tf.float32)/tf.reduce_sum(mask)
    
    #rescale to be between 1 and imbalance: i.e. the fg pixels are imbalance, the far bg are ~1, and other pixels are between these 2 values.
    closeness = closeness*(imbalance-1)+1
    
    #rescale for mean of 1
    closeness /= np.mean(closeness)
    
    return closeness

def closeness_to_foreground_balanced_tensor(mask,sigma=20):

    '''applies closeness_to_foreground_balanced to a rank-3 or rank-4 mask array (i.e. examples of images stacked along axis 0)'''
    
    closeness = tf.map_fn(lambda x: closeness_to_foreground_balanced(x,sigma),mask)
    
    return closeness


def binary_crossentropy_closeness_to_foreground(sigma=20):
    
    '''returns a loss function which weights pixels according to an exponentially-smoothed class imbalance. Spatial smoothing parameter sigma must be input'''
    
    def lossFunction(yTrue,yPred):
    
        #apply the closeness function to each image within the True mask, and re-concatenate
        weights = tf.py_function(func=closeness_to_foreground_balanced_tensor,
                                 inp=[yTrue,sigma],
                                 Tout=tf.float32)
        
        bce = keras.binary_crossentropy(yPred, yTrue)
    
        wcce = tf.multiply(bce, weights)
        
        return tf.reduce_mean(wcce)
    
    return lossFunction

def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = keras.flatten(y_true)
    y_pred_f = keras.flatten(y_pred)
    intersection = keras.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (keras.sum(y_true_f) + keras.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)