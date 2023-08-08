#!/usr/bin/env python

import pandas as pd

import numpy as np

from mask_utils import load_image, resample_image, pad_voxels

from tensorflow.keras.models import models_from_json
import os
from network_utils import gpu_memory_limit, predict_stochastic
from MultiResUNet.MultiResUNet import MultiResUnet
import pickle
import tempfile
import zipfile
import re
import glob
import pydicom as dcm
from optparse import OptionParser
import dicom_to_nifti as dn

# limit how much GPU RAM can be allocated by this notebook... 8GB is 1/3 of available
gpu_memory_limit(6000)

parser = OptionParser()

parser.add_option(
    "-i",
    "--input-directory",
    default="",
    dest="input_dir",
    help="Input directory - directory containing DICOM files to be analysed",
)
parser.add_option(
    "-o",
    "--output-dir",
    dest="output_dir",
    default=os.path.join("results", "new_predictions"),
    help="",
)

parser.add_option(
    "-f",
    "--force",
    dest="force_output",
    action="store_true",
    default=False,
    help="Force overwrite output - if the specified output file already exists, overwrite it.",
)

parser.add_option(
    "-m",
    "--model-base-name",
    dest="modelBaseName",
    default="mrunet_bayesian_2020-07-13_13-40",
    help="base name (within data/models) of the model you want to use. Default value is the model used in The Paper",
)
parser.add_option(
    "-s",
    "--output-segmentations",
    dest="output_segmentations",
    action="store_true",
    default=False,
    help="Store segmentation results in .nii format",
)

(options, args) = parser.parse_args()

assert options.force_output or not os.path.isdir(
    options.output_dir
), "Specified output dir already exists, choose a different location or force overwrite"

output_dir = options.output_dir

if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

input_dir = options.input_dir

try:
    PADSIZE = np.array(pickle.load(open(os.path.join("data", "PADSIZE.pickle"), "rb")))
    PXSPACING = np.array(
        pickle.load(open(os.path.join("data", "PXSPACING.pickle"), "rb"))
    )
except:
    print(
        "PICKLES NOT FOUND OR UNREADABLE. try running extract_dcm_for_wsx.py which should create them"
    )
PXAREA = np.product(PXSPACING)

# First, load the model
# specify which model to use
modelBaseName = options.modelBaseName

# location of the actual saved model
modelBaseName = os.path.join("data", "models", modelBaseName)

modelParamFile = modelBaseName + ".h5"
modelArchitecture = modelBaseName + ".json"

# with open( modelArchitecture , 'r') as json_file: # this has rotted due to tf upgrades
#     MODEL = model_from_json( json_file.read() )

MODEL = MultiResUnet(
    height=PADSIZE[0],
    width=PADSIZE[1],
    n_channels=1,
    layer_dropout_rate=None,
    block_dropout_rate=0.25,  # extarcted from json file which is now unreadable...
)

MODEL.load_weights(modelParamFile)

# hyperparameter N, defined according to quantify_model_performance.ipynb
N = 15

accuracyModelPath = modelBaseName + "_prediction_conversion.pickle"
ACCURACYMODEL = pickle.load(open(accuracyModelPath, "rb"))
# IF THIS BREAKS IN FUTURE:
# coefficient is 1.63920111
# intercept is -0.66730187

# file for writing results
OUTPUT_DIRECTORY = options.output_dir
RESULTSFILE = os.path.join(OUTPUT_DIRECTORY, "fat_predictions.csv")


def get_all_dicoms_in_directory(directory):
    all_files = glob.glob(os.path.join(directory, "**", "*"), recursive=True)

    all_files = [f for f in all_files if os.path.isfile(f)]

    all_dicoms = []
    for f in all_files:
        try:
            d = dcm.read_file(f)
            all_dicoms.append(f)
        except:
            pass
    return all_dicoms


def extract_if_first_4Ch_image(dicom_file):
    # filter for those described as 4Ch or cine

    d = dcm.read_file(dicom_file)
    if (
        "cine" in d.SeriesDescription.lower()
        and "4ch" in d.SeriesDescription.lower()
        and d.TriggerTime == 0.0
    ):
        print(f"found {dicom_file}")
        try:
            image, spacing, dicom_object = load_image(
                dicomPath=dicom_file,
                desiredPxSpacing=PXSPACING,
                padSize=PADSIZE,
                return_dicom_object=True,
            )
            return image, dicom_object
        except:
            return None, None
    else:
        return None, None


# update this list to match the output arguments in network_utils/predict_stochastic
RESNAMES = ["meanArea (cm2)", "stdArea (cm2)", "predicted DSC"]


def quantify_fat(dicom_file):
    # extract the pixels for each image.
    im, dicom_object = extract_if_first_4Ch_image(dicom_file)

    if im is not None:
        # create dictionary for returning results.
        resultDict = {
            "PatientName": dicom_object.PatientName,
            "PatientID": dicom_object.PatientID,
            "StudyDate": dicom_object.StudyDate,
            "StudyTime": dicom_object.StudyTime,
            "InstanceCreationDate": dicom_object.InstanceCreationDate,
        }

        im = im.reshape((1, *im.shape, 1))
        print(f"Predicting on {dicom_file}")
        res = predict_stochastic(MODEL, N, ACCURACYMODEL, im)

        if options.output_segmentations:
            # get the actual predicted segmentation (boolean version)
            segmentation = res[0].squeeze()

            # resize to original resolution and dimensions
            segmentation = resample_image(
                segmentation, PXSPACING, np.array(dicom_object.PixelSpacing)
            )
            segmentation = pad_voxels(
                segmentation, dicom_object.pixel_array.squeeze().shape
            )

            assert all(
                [
                    s == o
                    for s, o in zip(segmentation.shape, dicom_object.pixel_array.shape)
                ]
            ), "you've broken something in the image dimensions"

            # output directory for segmentation
            output_directory = os.path.join(OUTPUT_DIRECTORY, str(series_identifier))
            if not os.path.isdir(output_directory):
                os.makedirs(output_directory)

            # locations of the output nifti files...
            img_loc = os.path.join(output_directory, "la_4ch_ED.nii.gz")
            seg_loc = os.path.join(output_directory, "pcf_seg_la_4ch_ED.nii.gz")

            # make niftis from dicom object...
            vol, pixdim, affine = dn.dicom_to_volume([dicom_object])
            segmentation = segmentation.reshape(*vol.shape).astype("int")
            affine = dn.get_affine(dicom_object)

            # convert coordinares and write nifti for image
            dn.write_nifti(img_loc, vol, affine)
            vol, pixdim, _ = dn.dicom_to_volume([dicom_object])

            assert all(
                [s == o for s, o in zip(segmentation.shape, vol.shape)]
            ), "you've broken something in the image dimensions"
            dn.write_nifti(seg_loc, segmentation, affine)

        # wrap quantitative results up into a dict for easy DataFram-ing
        resultDict.update(
            dict(zip(RESNAMES, res[2:]))
        )  # remove first 2 elements of output as they are image-like (can't be in csv).

        # ensure that units of area are correct...
        resultDict["meanArea (cm2)"] *= PXAREA / 100
        resultDict["stdArea (cm2)"] *= PXAREA / 100
        return resultDict
    else:
        return None


all_input_dicoms = get_all_dicoms_in_directory(input_dir)

print(f"Found {len(all_input_dicoms)} dicom files in {input_dir}")
# create a dataframe to store results
results = []

for d in all_input_dicoms:
    result = quantify_fat(d)
    if result:
        results.append(result)

results = pd.DataFrame.from_records(results)


print(
    f"Found and analysed n={results.shape[0]} images which appear to be 4Ch cine.\nWriting results to {RESULTSFILE}"
)

print()
results.to_csv(RESULTSFILE, index=False)
