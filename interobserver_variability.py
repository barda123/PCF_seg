#!/usr/bin/env python
# coding: utf-8

# This script is for quantifying the contours drawn by mutiple different observers.


import sklearn

import glob

import os

import numpy as np

from Converters import parse_cvi42_xml

from mask_utils import (
    contour2mask,
    iou,
    dsc,
    symmetric_hausdorff_distance,
    mean_contour_distance,
)

import pickle

import matplotlib.pyplot as plt

import pandas as pd

from itertools import combinations

import pydicom as dcm

# main directory to store pickle files etc
crossValidation_temp_folder = "./data/pericardial/interobserver_variability/"

# list of folders containing contour files
crossValidationFolders = [
    "./data/pericardial/wsx_round2",
    "./data/pericardial/interobserver_variability/zre_contours/",
    "./data/pericardial/interobserver_variability/sep_contours/",
]

DICOMDIR = "./data/pericardial/wsx_round2/paired/"  # the folder where all the original dicoms can be found...

# list of all wsx files in each folder
allWsxFiles = [glob.glob(os.path.join(f, "*.cvi42wsx")) for f in crossValidationFolders]


# just the patient names...
def get_patient_name(file):
    return os.path.basename(file)[:8]


patientNames = [[get_patient_name(file) for file in folder] for folder in allWsxFiles]


# all the patient names which are in ALL of the folders
intersectionPatientNames = set(patientNames[0]).intersection(*patientNames)

observerFolders = []

# loop over each folder, creating another folder for the outputs, and processing the wsx files into them
for ind, folder in enumerate(allWsxFiles):
    outputFolder = os.path.join(crossValidation_temp_folder, "observer" + str(ind + 1))

    if not os.path.isdir(outputFolder):
        os.mkdir(outputFolder)

    # loop over the files
    for file in folder:
        # if wsx file is shared, process it into pickle files...
        if get_patient_name(file) in intersectionPatientNames:
            parse_cvi42_xml.parseFile(file, output_dir=outputFolder)

    observerFolders.append(outputFolder)


# Due to slight naming inconsistencies, we should use only the pickle files named after dicom images from here onwards.

allPickles = []
for folder in observerFolders:
    pickles = glob.glob(os.path.join(folder, "*.pickle"))
    pickles = [os.path.basename(p) for p in pickles if "_contours_dct.pickle" not in p]

    allPickles.append(pickles)

# the unique pickle files for each contour
intersectPickles = set(allPickles[0]).intersection(*allPickles)

# also get the pixel spacing from the original DICOM files..
dicomFiles = [
    os.path.join(DICOMDIR, p.replace(".pickle", ".dcm")) for p in intersectPickles
]
pxSpacing = [dcm.read_file(d).PixelSpacing[0] for d in dicomFiles]


assert len(intersectPickles) == len(
    intersectionPatientNames
), "You have messed something up"

# an arbitrary numpy array which is bigger than any of the images and can be used for mask creation
im = np.zeros((400, 400))


def get_mask(file):
    with open(file, "rb") as f:
        contour = pickle.load(f)
    mask = contour2mask(contour, im)
    return mask


pairwiseComparisons = combinations(observerFolders, 2)

ious = pd.DataFrame()
dscs = pd.DataFrame()
shds = pd.DataFrame()
mcds = pd.DataFrame()

# loop over the pickle files
for ind, file in enumerate(intersectPickles):
    pairwiseComparisons = combinations(observerFolders, 2)
    # loop over the pairwise observer comparisons
    for a, b in pairwiseComparisons:
        columnName = " vs ".join([os.path.basename(a), os.path.basename(b)])

        # get masks
        aMask = get_mask(os.path.join(a, file))
        bMask = get_mask(os.path.join(b, file))

        # compare them!
        ious.loc[ind, columnName] = iou(aMask, bMask)
        dscs.loc[ind, columnName] = dsc(aMask, bMask)
        shds.loc[ind, columnName] = symmetric_hausdorff_distance(
            aMask, bMask, pxSpacing[ind]
        )
        mcds.loc[ind, columnName] = mean_contour_distance(aMask, bMask, pxSpacing[ind])

allMetrics = pd.DataFrame(columns=ious.columns)

for name, metric in [
    ("intersection-over-union", ious),
    ("dice coefficient", dscs),
    ("mean contour distance (mm)", mcds),
    ("symmetric hausdorff distance (mm)", shds),
]:
    desc = metric.describe()
    allMetrics.loc[name + " mean", :] = desc.loc["mean", :]
    allMetrics.loc[name + " std", :] = desc.loc["std", :]

allMetrics.to_csv("./graphs/interobserver_variability_statistics.csv")


allMetrics


# Now, lets do a bland-altman plot (because that's fun)


plt.figure(figsize=(15, 5))

pairwiseComparisons = combinations(observerFolders, 2)

for ind, (a, b) in enumerate(pairwiseComparisons):
    # bland-altman plot for the the test set...
    aAreas = np.array(
        [
            np.sum(get_mask(os.path.join(a, file))) * (spacing**2) / 100
            for spacing, file in zip(pxSpacing, intersectPickles)
        ]
    )
    bAreas = np.array(
        [
            np.sum(get_mask(os.path.join(b, file))) * (spacing**2) / 100
            for spacing, file in zip(pxSpacing, intersectPickles)
        ]
    )

    meanArea = (aAreas + bAreas) / 2
    diffArea = aAreas - bAreas

    meanDiff = np.mean(diffArea)
    stdDiff = np.std(diffArea)

    plt.subplot(1, 3, ind + 1)
    plt.scatter(meanArea, diffArea)

    plt.axhline(meanDiff, c="k", alpha=0.5)
    plt.axhline(meanDiff + 1.96 * stdDiff, c="k", alpha=0.5, linestyle="--")
    plt.axhline(meanDiff - 1.96 * stdDiff, c="k", alpha=0.5, linestyle="--")

    plt.title(
        " vs ".join([os.path.basename(a), os.path.basename(b)])
        + " (n = "
        + str(len(intersectPickles))
        + ")"
    )

    plt.ylim([-40, 40])
    plt.xlim([0, 80])

plt.savefig("./graphs/interobserver_variability_bland-altman.png")
plt.savefig("./graphs/interobserver_variability_bland-altman.svg")


# In[ ]:


plt.figure(figsize=(12, 10))

plt.subplot(2, 2, 1)
for col in ious.columns:
    plt.hist(
        ious[col].values,
        bins=np.arange(0, 1.05, 0.05),
        density=True,
        label=col,
        alpha=0.5,
    )
#     plt.hist(ious[col].values,density = True,label = col, alpha = 0.5)
plt.xlim([0, 1])
plt.xlabel("intersection-over-union")
plt.ylabel("probability density")

plt.subplot(2, 2, 2)
for col in dscs.columns:
    plt.hist(
        dscs[col].values,
        bins=np.arange(0, 1.05, 0.05),
        density=True,
        label=col,
        alpha=0.5,
    )
#     plt.hist(dscs[col].values,density = True,label = col, alpha = 0.5)
plt.xlim([0, 1])
plt.xlabel("dice coefficient")
plt.ylabel("probability density")

plt.subplot(2, 2, 3)

for col in mcds.columns:
    #     plt.hist(mcds[col].values,bins = np.arange(0,20,1),density = True,label = col, alpha = 0.5)
    plt.hist(mcds[col].values, density=True, label=col, alpha=0.5)
plt.xlabel("mean contour distance (mm)")

plt.ylabel("probability density")
plt.subplot(2, 2, 4)
for col in shds.columns:
    #     plt.hist(shds[col].values,bins = np.arange(0,200,10),density = True,label = col, alpha = 0.5)
    plt.hist(shds[col].values, density=True, label=col, alpha=0.5)
plt.xlabel("hausdorff distance (mm)")

plt.ylabel("probability density")
plt.legend()

plt.savefig("./graphs/interobserver_variability_histograms.png")
plt.savefig("./graphs/interobserver_variability_histograms.svg")
