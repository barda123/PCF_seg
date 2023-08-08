# PCF_seg

This repo centres on quantification of pericardial fat deposits from cardiac MRI, using a CNN, and data from the UK Biobank.

I start by preprocessing manually-segmented images, created using cvi42 software. 

Files containing manual contours are used to pull out the relevant dicom files from the (vast) UK Biobank dataset. The contours are used to make "masks" of the same dimension as the images.

Image-mask pairs are then split into train and test sets, and used to train/evaluate a CNN - which is based around a multi-residual U-net (Ozcan 2020) with added quality control mechanism.

You can (and should) read all about it in The Paper before continuing:
https://www.frontiersin.org/articles/10.3389/fcvm.2021.677574/full

## How to use this repo

This repo contains a single model, and some code which should be useful for onward development. In order for *most* of this to function, you will need to follow the notes in `data/notes.txt`.

### Assumed knowledge
 * linux command line and directory structures, symlinks
 * python virtual environments
 * numpy, pydicom, tensorflow, pandas
 * The Paper

### The dustbin of history

When this was developed, it was a bit messy, and I haven't cleaned up perfectly. Most development was performed in IPython (jupyter) notebooks (`.ipynb`). There is a folder marked `dev_notebooks` containing them. I cannot guarantee they will still work - and in order to use them they will need to be moved into the main directory of this repo (but hopefully you won't need them).

### Setting up your virtual environment

Uses python version 3.8.5

```pyenv install 3.8.5
pyenv local 3.8.5
pip install virtualenv
```

Create a new virtual environment:

```
python -m venv .venv
```

activate it:
`source .venv/bin/activate`

and install all requirements:
`pip install -r requirements.txt`

### Preprocessing wsx files and extracting dicoms for train/test data

The script `extract_dcm_for_wsx.py` is used to: 
 * look up manual segmentations
 * extract dicoms corresponding to them from the UK Biobank database
 * preprocess images and segmentations to ensure uniform resolution, pixel scaling and image size 
 * save preprocessed images and segmentations as numpy arrays
 
In order to run it, follow the notes in `data/notes.txt`, ensuring in particular that you can access `data/pericardial/wsx_round2/` **on the GPU machine**. Then you can simply:

`python extract_dcm_for_wsx.py`

### Training a new model

Follow the notes in `data/notes.txt`, ensuring in particular that you can access `data/pericardial/wsx_round2/` **on the GPU machine**. These are the data which were used to train and test the original published model, and were created by `extract_dcm_for_wsx.py`.

This script performs the following:
 * Loads numpy arrays representing preprocessed images
 * (Performs a train/test split, if it cannot find numpy arrays already representing split data)
 * Instantiates, trains and saves a model, as described in The Paper
 * Creates some descriptive statistics and graphs

In order to run as-is, you can simply:

`python train.py`

Running this script should create and train some figures which describe the new model's performance. The script will print out the location where the model is being saved, you should make a note of this.

In order to train a model using a combination of existing + novel data, I recommend:
 * Understanding how `train.py` currently functions
 * Modifying it so that after it loads pre-existing data (`splitData.npy`, lines 76-80), additional data are appended to the varibles `X` and `Y` using image preprocessing techniques you can get from `extract_dcm_for_wsx.py`
 * You **should** consider making the management of datasets used to create particular models more reproducible. At minimum, this might consist of saving the `X` and `Y` numpy arrays into the same directory as model weights, after any additional data are appended to them.
 * I also recommend modifying `train.py` so that a copy of it is automatically saved along with the created model (for reproducibility)

*NB if interested, this script was based on `dev_notebooks/mrunet_Bayesian_dev.ipynb`, which contains a number of extra options commented out*

### Evaluating model performance

The notebook `dev_notebooks/quantify_model_performance.ipynb` is pretty similar to `train.py`, but does not create or save a model (if you need to recreate graphs etc on a pre-existing model).

See also - `interobserver_variability.py`, which (with appropriate data access on GPU machine) allows you to quantify the agreement between 3 different readers using various different metrics. This can be used to contextualise model performance

### Running the inference pipeline on novel data

There is an example of how this functions in `predict_fat_areas_UKB.ipynb` - although this has hardcoded locations of all files (including output and input locations). A more useful tool is provided in `predict_fat_areas.py`.

This script assumes that all the DICOMs you want to quantify are in a beneath a single directory. It does not currently handle zipfiles, so you will need to extract DICOMs into the directory.

End-diastole images from 4Ch cine sequences are identified using the following combination of metadata attributes:
 * "cine" in the Series Description
 * "4ch" in the Series Description
 * Trigger Time = 0
This combination is likely to be a problem at some point.

To see options, run:
`python predict_fat_areas --help` or `python predict_fat_areas -h`

 