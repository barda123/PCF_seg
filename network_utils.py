#UTILITIES that can be used for any notebook involving networks for segmentation...

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import Sequence


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