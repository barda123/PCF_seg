#UTILITIES that can be used for any notebook involving networks for segmentation...

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import Sequence

import numpy as np

from mask_utils import dsc,iou

import itertools

def gpu_memory_limit(memory_limit):
    
    '''This function limits GPU memory usage and should be called after tensorflow import but before any session instantiation'''
    
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only allocate 16GB of memory on the first GPU
        try:
            tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit)])
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            print('GPU memory limit allocated.')
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)
            

class augmentImageSequence(Sequence):
    
    '''class for data augmentation on matched image/mask pairs'''
    
    def __init__(self,Images,Masks,dataGenArgs,batchSize=1,seed=42):
        
        #copy raw data in
        self.x,self.y = Images,Masks
        self.batch_size = batchSize
        
        #convert to imageDataGenerators/create flow objects...
        self.augmentIm = ImageDataGenerator(**dataGenArgs).flow(x=Images,batch_size=batchSize,seed=seed)
        self.augmentMa = ImageDataGenerator(**dataGenArgs).flow(x=Masks, batch_size=batchSize,seed=seed)
        
    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self,idx):
        #cheaty fake 1-stage loop, returns 1 batch from both flow objects (which will be matched)
        for _,ims,masks in zip(range(1),self.augmentIm,self.augmentMa):        
            masks = (masks>0.5).astype('float')            
            return ims,masks
        
        
        
#FUNCTIONS FOR DOING STOCHASTIC PREDICTIONS...

#FIXME! remove the unneeded metrics once network (and methodology) has been finalised.

#FIXMMEEEEEEEE make it so these can be called on arrays where M>1!!!!! BECAUSE THIS SUCKS

    
def mean_pairwise_dsc(predictions):
    
    #all combinations of samples, which will be axis 0
    dscs = np.array([dsc(a,b) for a,b in itertools.combinations(predictions,2)])
    
    #zero-predictions produce undefined dice scores
    dscs[np.isnan(dscs)] = 0
    
    assert not np.any(dscs<0),'wtf, there are dice scores less than 0'
    
    return np.mean(dscs)
    
def voxel_uncertainty(predictions):
    
    '''voxel-wise uncertainty as defined in Roy et al (2018)'''
    
    #strcture-and-voxel-wise uncertainty (compresses over the sample axis)
    feature_uncertainty = -np.sum(predictions*np.log(predictions),axis = 0)
    #global uncertainty is the sum over the feature axis
    global_uncertainty = np.sum(feature_uncertainty,axis=-1)
    
    return global_uncertainty
    
def mean_std_area(predictions):
    
    '''the area occupied by each segmented channel. outputs two array: mean and standard deviation
    RETURNS ANSWERS IN PIXELS WHICH MUST BE RESCALED LATER!!!!!!
    '''
    #get the dims
    N = predictions.shape[0]
    nPixels = np.product(predictions.shape[1:-1])
    nFeatures = predictions.shape[-1]
    
    #reshape array so that it is (N,pixels,features) and ensure it is boolean via threshold.
    predictions = predictions.reshape((N,nPixels,nFeatures)) > 0.5
    
    #sum of voxels for each 
    areas = np.sum(predictions,axis = 1)
    
    #mean, returning a value for each segmentation channel
    mu = np.mean(areas,axis=0)[0]
    sigma = np.std(areas,axis=0)[0]
    
    return mu,sigma

def predict_stochastic(segmentationModel,N,accuracyModel, X):
    
    '''draw and summarise multiple predictions from a model
    Arguments:
        model {a model, for example a Keras model, with a predict method} -- is assumed to have some stochastic component, i.e. multiple
        N {int} -- the number of sample predictions to be drawn from the stochastic model
        X {numpy array, probably float} -- assumed to be already consistent with inputs to the model. MUST ONLY BE A SINGLE IMAGE AND NOT MULTIPLE STACKED!!!!!
        
    Returns:
        consensus {numpy array, boolean} -- pixelwise segmentation of x
        also various floats, representing different metrics for uncertainty and the outputs.
    '''
    
    #draw N predictions from the model over x
    predictions = np.stack([segmentationModel.predict(X) for n in range(N)],axis=0)
        
    consensus = np.mean(predictions>0.5,axis=0)>0.5 
    
    #metrics described in Roy et al...
    uncertainty = voxel_uncertainty(predictions)
    
    mpDsc = mean_pairwise_dsc(predictions)
    predictedDsc = accuracyModel.predict(mpDsc.reshape(-1,1))[0][0]
    #no Dice < 0
    predictedDsc = max(predictedDsc,0)
    # no Dice > 1
    predictedDsc = min(predictedDsc,1)
    
#     gDsc = global_dsc(predictions)
    
#     mpIou = mean_pairwise_iou(predictions)
#     gIou = global_iou(predictions)
    meanArea,stdArea = mean_std_area(predictions)
    
    return consensus,uncertainty,meanArea,stdArea,predictedDsc