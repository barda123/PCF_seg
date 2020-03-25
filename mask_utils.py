#This script contains utility functions concerned with loading and messing with paired dicom files and masks from pickles created by a Wenjia Bai script. 

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



def load_image_and_mask(picklePath,dicomPath,pad_size = None, collapse=True,labelFilter=''):

    '''takes paths to matched files - a pickle output from parsing a cvi42wsx, and the corresponding dicom
    the pickle must refer directly to a single dicom file (i.e. not a higher-order one referring to a whole sequence)
    padSize is the size of the output images - it currently allows cropping or padding.
    labelFilter allows passing in of a regex string for the NAMES of the different contours. 
    collapse specifies whether the different contours are or-ed (i.e. forcing a single-channel boolean mask)
    WARNING - will have unexpected behaviour with collapse=False and heterogeneous labels 
    '''
    
    #load dicom image.
    image = dcm.dcmread(dicomPath,stop_before_pixels=False)
    
    #load the pickled contour
    with open(picklePath,'rb') as f:
        contour = pickle.load(f)
    
    #consider case where there are >=1 contours per image
    nContours = len(contour)
    
    #get dimensions of image
    nx,ny = image.pixel_array.shape
    
    #create indexers for filling in mask
    x,y = np.meshgrid(range(nx),range(ny))
    x = x.reshape(-1,1)
    y = y.reshape(-1,1)
    xy = np.concatenate((x,y),axis=1) #xy matrix
    
#     print(xy.shape)
    
    #create mask which can contain all contours.
    mask = np.zeros((*image.pixel_array.shape,nContours),dtype = 'bool')
    
    #if no filter specified, the default one will always match
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
        
    #extract the raw pixel values from the dicom file, and normalise to 0-1
    minVal = np.min(image.pixel_array)
    maxVal = np.max(image.pixel_array)
    im = (image.pixel_array - minVal) / (maxVal - minVal)
    
    #get size of pixels(required for downstream analysis)
    pxSize = np.product(image.PixelSpacing)
    
    if pad_size != None:
        
        im = pad_voxels(im,pad_size)
        mask = pad_voxels(mask,pad_size)

    return im,mask,pxSize