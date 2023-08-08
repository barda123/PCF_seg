#!/usr/bin/env python
# coding: utf-8

from Converters import parse_cvi42_xml
import pickle
from zipfile import ZipFile
import os
import glob
from IPython.display import clear_output
import re
import numpy as np
import pandas as pd
import seaborn as sns
from itertools import compress
import matplotlib.pyplot as plt
from mask_utils import load_image, load_image_and_mask, show_image_with_masks


wsxDir = (
    "./data/pericardial/wsx_round2/"  # directory where finalised wsx files are kept.
)

pairedDir = os.path.join(wsxDir, "paired")  # subdirectory for outputs.

if not os.path.isdir(pairedDir):
    os.mkdir(pairedDir)

wsxFiles = glob.glob(os.path.join(wsxDir, "*.cvi42wsx"))

# parse all the wsx files into pickles.
# [parse_cvi42_xml.parseFile(w,output_dir=pairedDir) for w in wsxFiles]

# get only the pickle files referring to individual slice names - i.e. named using uids.
pickles = glob.glob(os.path.join(pairedDir, "*.pickle"))

pickles = [p for p in pickles if "_contours_dct.pickle" not in p]


def find_and_extract_relevant_dcm(
    fileNames,
    outputDir=".",
    zippedDataPath="data/imaging_by_participant",
    zipFilter="[\S\s]*",
):
    """takes a pickle file, or list/array thereof (presumablty exported from a cvi42wsx file) and finds the correctn corresponding dicom file
    fileNames: list or array of paths to pickle files (or corresponding dicom files) created by parseFile()
    outputDir: where to put the dicom file
    zippedDataPath: the top-level directory within which all zipped dicom files reside.
    zipFilter: a regex that can be used to filter for only the zipfiles we care about.
    """

    # if 1 file, make it a list
    if type(fileNames) == str:
        fileNames = [fileNames]

    # use names of pickles to get names of their (expected) dicom file
    dicomNames = [os.path.basename(p.replace(".pickle", ".dcm")) for p in fileNames]

    # uniqueify
    dicomNames = list(set(dicomNames))

    # create list of the outputs!
    dicomPaths = [os.path.join(outputDir, d) for d in dicomNames]
    # check for dicom files in the output directory, so we can subset and avoid duplicated work
    alreadyThere = [
        os.path.basename(f) for f in glob.glob(os.path.join(outputDir, "*.dcm"))
    ]
    dicomNames = list(set(dicomNames) - set(alreadyThere))

    if len(dicomNames) == 0:
        print("no work to do!!")
    else:
        print("getting list of all zipfiles in path...")
        # get list of ALL dicoms within top-level directory
        allZips = glob.glob(os.path.join(zippedDataPath, "**", "*.zip"), recursive=True)

        # filter names of zips using regex, and give some idea of how much this has achieved.
        nAllZips = len(allZips)
        zipFilter = re.compile(zipFilter)
        allZips = [z for z in allZips if zipFilter.match(os.path.basename(z))]
        nFilteredZips = len(allZips)
        print(
            "regex filtering reduced "
            + str(nAllZips)
            + " zipfiles to "
            + str(nFilteredZips)
        )

        i = 0
        while len(dicomNames) > 0 and i < len(allZips):
            zf = ZipFile(allZips[i])

            contents = zf.namelist()
            for d in dicomNames:
                if d in contents:
                    zf.extract(d, path=outputDir)
                    dicomNames.remove(d)
                    # give some indication of how much is done
            #                     print(str(100*((len(dicomPaths) - len(dicomNames))/len(dicomNames))) + '% found and extracted')
            zf.close()
            i += 1

        if len(dicomNames) != 0:
            print(
                "warning: not all dicoms found. consider broadening your regex. files not found:\n"
                + "\n".join(dicomNames)
            )

    return dicomPaths


# In[ ]:


dicomPaths = find_and_extract_relevant_dcm(
    fileNames=pickles, outputDir=pairedDir, zipFilter="[\S\s]*_longaxis"
)  # as we are only looking for long axis images.

# now, it is possible that dicom and pickle paths are not in the same order... check that they are matched.
pickles = sorted(pickles)
dicomPaths = sorted(dicomPaths)

# subset for those with image...
dcmFound = [os.path.isfile(d) for d in dicomPaths]

pickles = list(compress(pickles, dcmFound))
dicomPaths = list(compress(dicomPaths, dcmFound))


# preload images and spacings so we can select parameters
info = pd.DataFrame({"dicom": dicomPaths})

images, spacings = zip(*info["dicom"].apply(load_image))

info.loc[:, "xDim"], info.loc[:, "yDim"] = zip(*[i.shape for i in images])

info.loc[:, "xSpacing"], info.loc[:, "ySpacing"] = zip(*spacings)


sns.jointplot(x=info["xSpacing"], y=info["ySpacing"], kind="hex")


sns.jointplot(x=info["xDim"], y=info["yDim"], kind="hex")


# pick resolution and spacing based on the above plots....

PADSIZE = np.concatenate([info["xDim"].mode().values, info["yDim"].mode().values])

print("modal image size = " + str(PADSIZE))

PXSPACING = np.concatenate(
    [info["xSpacing"].mode().values, info["ySpacing"].mode().values]
)

print("modal image resolution = " + str(PXSPACING))


# load all files, and put into arrays of dimension (m,x,y)
PADSIZE = [208, 208]
PXSPACING = [1.82692313, 1.82692313]

# write out the details for image sizes and pixel spacing so they can be reused in other notebooks
pickle.dump(PADSIZE, open(os.path.join("data", "PADSIZE.pickle"), "wb"))
pickle.dump(PXSPACING, open(os.path.join("data", "PXSPACING.pickle"), "wb"))

# preallocate arrays..
m = len(pickles)
X = np.zeros((m, *PADSIZE))
Y = np.zeros((m, *PADSIZE), dtype="bool")
pxSize = np.zeros((m, 2))


for ind, (p, d) in enumerate(zip(pickles, dicomPaths)):
    X[ind, :, :], Y[ind, :, :], pxSize[ind, :] = load_image_and_mask(
        p, d, PXSPACING, PADSIZE, labelFilter="freeDraw"
    )

pxSize = np.product(pxSize, axis=-1)
# remove images without any contours.
use = np.max(np.max(Y, axis=2), axis=1) > 0
if np.any(~use):
    print(
        "check your input data regarding, there are "
        + str((~use).sum())
        + " images with no mask"
    )
X = X[use, :, :]
Y = Y[use, :, :]

# also filter pickles and dicom paths for later, just in case
pickles = list(compress(pickles, use))
dicomPaths = list(compress(dicomPaths, use))

# save X and Y for use in the ML dev notebook
np.save(os.path.join(wsxDir, "X.npy"), X)
np.save(os.path.join(wsxDir, "Y.npy"), Y)
np.save(os.path.join(wsxDir, "pxSize.npy"), pxSize)


# summary statistics for the area of pcf....

fatArea = np.sum(Y, axis=(1, 2)) * pxSize / 100  # in mm^2

plt.hist(fatArea)

negs = 54

egs = np.random.randint(m, size=negs)

ncols = 4
nrows = np.ceil(negs / ncols)

plt.figure(figsize=(5 * ncols, 5 * nrows))
# lets ave a look
for i in range(negs):
    plt.subplot(nrows, ncols, i + 1)

    show_image_with_masks(
        X[egs[i], :, :], Y[egs[i], :, :], {"linewidth": 1, "color": "y"}
    )
