#This script contains utility functions concerned with loading and messing with paired dicom files and masks from pickles created by a Wenjia Bai script. 

import re
import numpy as np
import pydicom as dcm
from matplotlib.path import Path as mPath
import pickle
import warnings
import matplotlib.pyplot as plt
from scipy.spatial import distance

def centered_slice(X, L):
    L = np.asarray(L)
    shape = np.array(X.shape)

    # verify assumptions
    assert L.shape == (X.ndim,)
    assert ((0 <= L) & (L <= shape)).all()

    # calculate start and end indices for each axis
    starts = (shape - L) // 2
    stops = starts + L

    # convert to a single index
    idx = tuple(np.s_[a:b] for a, b in zip(starts, stops))
    return X[idx]

def pad_voxels(voxels,pad_size):
    
    nx,ny = voxels.shape    

    #calculate edges and create tuple to ensure correct dimension
    xedge = np.maximum((pad_size[0] - nx) //2,0)
    yedge = np.maximum((pad_size[1] - ny) //2,0)
    pad_width = ( (int(np.floor(xedge)),int(np.ceil(xedge))) , (int(np.floor(yedge)),int(np.ceil(yedge))) )

    voxels= np.pad(voxels,pad_width,'constant')
    
    if np.any([nx,ny] > pad_size): 
        warnings.warn('Image is larger than padding dimension you specified, so you are losing pixels at the edges')
        
        voxels = centered_slice(voxels, pad_size)
    
    return voxels



def load_image(dicomPath,pad_size=None):
    
    '''load an image from a single .dcm file. returns the normalised pixel array (range 0-1) and the pixel size in mm^2, and '''

    #load dicom image.
    image = dcm.dcmread(dicomPath,stop_before_pixels=False)
    
    #extract the raw pixel values from the dicom file, and normalise to 0-1
    minVal = np.min(image.pixel_array)
    maxVal = np.max(image.pixel_array)
    im = (image.pixel_array - minVal) / (maxVal - minVal)
    
    #get size of pixels(required for downstream analysis)
    pxArea = np.product(image.PixelSpacing)
    
    return im,pxArea
    


def load_image_and_mask(picklePath,dicomPath,pad_size = None, collapse=True,labelFilter=''):

    '''takes paths to matched files - a pickle output from parsing a cvi42wsx, and the corresponding dicom
    the pickle must refer directly to a single dicom file (i.e. not a higher-order one referring to a whole sequence)
    padSize is the size of the output images - it currently allows cropping or padding.
    labelFilter allows passing in of a regex string for the NAMES of the different contours. 
    collapse specifies whether the different contours are or-ed (i.e. forcing a single-channel boolean mask)
    WARNING - will have nonintuituve behaviour with collapse=False and heterogeneous labels, their ordering might not be deterministic 
    '''
    
    #load image, but do not pad or trim as this needs to be done in parallel with the mask, otherwise the pixel coordinates in the contour will not match
    im,pxArea = load_image(dicomPath,pad_size=None)
    
    #load the pickled contour
    with open(picklePath,'rb') as f:
        contour = pickle.load(f)
    
    #consider case where there are >=1 contours per image
    nContours = len(contour)
    
    #get dimensions of image
    nx,ny = im.shape
    
    #create indexers for filling in mask
    x,y = np.meshgrid(range(nx),range(ny))
    x = x.reshape(-1,1)
    y = y.reshape(-1,1)
    xy = np.concatenate((x,y),axis=1) #xy matrix
        
    #create mask which can contain all contours.
    mask = np.zeros((*im.shape,nContours),dtype = 'bool')
    
    #if no filter specified, an empty string/default one will always match
    labelFilter = re.compile(labelFilter)
    
    for ind,c in enumerate(sorted(contour.keys())):
        #if regex for the name of the contour is correct, use it... default argument for labelFilter will always match
        if labelFilter.match(c):
            #get grid points inside contour
            path = mPath(contour[c])
            inContour = path.contains_points(xy)
            #index into mask...
            mask[y[inContour],x[inContour],ind] = True
    
    #if specified, collapse down to 1D representation
    if collapse:
        mask = np.max(mask,axis=2)
    
    #now pad the matched image and mask to the desired dimensions
    if pad_size != None:
        im = pad_voxels(im,pad_size)
        mask = pad_voxels(mask,pad_size)

    return im,mask,pxArea

#A FUNCTION FOR SHOWING AN IMAGE AND MASK IN A NICE FORMAT

def mask2line(mask,addNa = True):
        
    '''takes a mask (i.e. a boolean image) and turns it into a collection of horizontal and vertical lines, which can be plotted on top of an image. stolen from https://stackoverflow.com/questions/24539296/outline-a-region-in-a-graph'''
    
    # a vertical line segment is needed, when the pixels next to each other horizontally
    #   belong to diffferent groups (one is part of the mask, the other isn't)
    # after this ver_seg has two arrays, one for row coordinates, the other for column coordinates 
    ver_seg = np.where(mask[:,1:] != mask[:,:-1])

    # the same is repeated for horizontal segments
    hor_seg = np.where(mask[1:,:] != mask[:-1,:])

    # if we have a horizontal segment at 7,2, it means that it must be drawn between pixels
    #   (2,7) and (2,8), i.e. from (2,8)..(3,8)
    # in order to draw a discountinuous line, we add nans in between segments
    l = []
    for p in zip(*hor_seg):
#         print(p)
        l.append((p[1], p[0]+1))
        l.append((p[1]+1, p[0]+1))
        if addNa:
            l.append((np.nan,np.nan))

    # and the same for vertical segments
    for p in zip(*ver_seg):
        l.append((p[1]+1, p[0]))
        l.append((p[1]+1, p[0]+1))
        if addNa:
            l.append((np.nan, np.nan))

    # now we transform the list into a numpy array of Nx2 shape - all coordinates will be in pixel space
    segments = np.array(l)
    
    return segments

def show_image_with_masks(image,masks,maskOptions=[]):
    
    '''uses matplotlib to show an image, with masks as line overlays over the top. 
    masks should be either a single boolean image, or a list of boolean images. Each should be of equal dimension to the image
    maskOptions should be either a dict, or a lust of dicts, with options for mask display
    '''
    
    #show the image
    plt.imshow(image,cmap='gray') #gray is standard for medical images.
    
    #ensure that if a single mask is passed it will 
    if type(masks) == np.ndarray or type(masks) == np.array:
        masks = [masks]
    
    for mInd,mask in enumerate(masks):
        assert mask.shape == image.shape,'Image and mask ' + str(mInd) + 'are not of equal dimension'
    
        if np.sum(mask)>0:
            segments = mask2line(mask)

            if type(maskOptions) == dict:
                # plot all with the same options
                plt.plot(segments[:,0], segments[:,1], **maskOptions)
            elif type(maskOptions)==list and len(maskOptions) == len(masks): #i.e. if there is a dict for each mask
                plt.plot(segments[:,0], segments[:,1], **maskOptions[mInd])
            else:
                plt.plot(segments[:,0], segments[:,1])


    #no x or y ticks needed
    plt.xticks([])
    plt.yticks([])
    
    return segments



#METRICS for segmentation accuracy
def iou(yTrue,yPred):
    '''intersection-over-union score for numpy arrays'''
    
    #ensure booleans, assuming scaling is 0-1
    yTrue = yTrue>=0.5
    yPred = yPred>=0.5
    
    intersection = np.sum(np.logical_and(yTrue,yPred))
    
    union = np.sum(np.logical_or(yTrue,yPred))
    
    return intersection/union

def dsc(yTrue,yPred):
    '''dice-sorenson coefficient for numpy arrays'''

    #ensure booleans, assuming scaling is 0-1
    yTrue = yTrue>=0.5
    yPred = yPred>=0.5
    
    numerator = 2 * np.sum(np.logical_and(yTrue,yPred))
    
    denominator = np.sum(yTrue) + np.sum(yPred)
    
    return numerator/denominator

def symmetric_hausdorff_distance(yTrue,yPred,pxSpacing):
    
    #ensure booleans and get point representation
    yTrue = mask2line(yTrue >=0.5,addNa = False)
    yPred = mask2line(yPred >=0.5,addNa = False)
    
    #symmetric hausdorff is maximum of directed hausdorffs. Multiply by a pixel size for real-world
    hd = pxSpacing * np.maximum(distance.directed_hausdorff(yTrue,yPred)[0],
                                distance.directed_hausdorff(yPred,yTrue)[0])
    
    return hd

def mean_contour_distance(yTrue,yPred,pxSpacing):
    
    #ensure booleans and get point representation
    yTrue = mask2line(yTrue >=0.5,addNa = False)
    yPred = mask2line(yPred >=0.5,addNa = False)
    
    #get pairwise distances between the two arrays - from each point, the minimum distance to the other array
    pwdistTrue = np.min(distance.cdist(yTrue,yPred),axis=1)
    pwdistPred = np.min(distance.cdist(yPred,yTrue),axis=1)
    
    #mean-of-means for these two sets of distances
    meanDistTrue = np.mean(pwdistTrue)
    meanDistPred = np.mean(pwdistPred)
    
    return pxSpacing * np.mean([meanDistTrue,meanDistPred])
    
    