#!/usr/bin/env python
# coding: utf-8


import numpy as np

import os

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import tensorflow as tf

from tensorflow.keras.models import Model

from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as keras
from tensorflow.keras import callbacks
from tensorflow.keras import metrics

from scipy.stats import pearsonr

from custom_losses import (
    binary_crossentropy_weight_balance,
    binary_crossentropy_weight_dict,
    binary_crossentropy_closeness_to_foreground,
    dice_coef_loss,
)

from mask_utils import (
    show_image_with_masks,
    iou,
    symmetric_hausdorff_distance,
    mean_contour_distance,
    dsc,
)

from network_utils import (
    gpu_memory_limit
    augmentImageSequence,
    mean_pairwise_dsc,
    mean_std_area,
    voxel_uncertainty,
)

from MultiResUNet.MultiResUNet import MultiResUnet

from datetime import datetime

import itertools

import pickle

# limit how much GPU RAM can be allocated by this notebook... 8GB is 1/3 of available
# gpu_memory_limit(8000)

gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# directory for keeping models, and journaling their performance/graphs
modelDir = os.path.join("data", "models")
if not os.path.isdir(modelDir):
    os.mkdir(modelDir)

dateStr = datetime.now().strftime("%Y-%m-%d_%H:%M")
outputName = os.path.join(modelDir, "mrunet_bayesian_" + dateStr)

print(f"Model name = {outputName}")

DataDir = "./data/pericardial/wsx_round2/"

splitDataFile = os.path.join(DataDir, "splitData.npy")

if os.path.isfile(splitDataFile):
    splitData = pickle.load(open(splitDataFile, "rb"))
    X, X_test, Y, Y_test, pxArea, pxArea_test, pxSpacing, pxSpacing_test = splitData

else:
    from sklearn.model_selection import train_test_split

    # load data - these files created by extract_dcm_for_wsx.ipynb
    X = np.load(os.path.join(DataDir, "X.npy"))
    Y = np.load(os.path.join(DataDir, "Y.npy")).astype("float")
    pxArea = np.load(os.path.join(DataDir, "pxSize.npy"))
    pxSpacing = np.sqrt(pxArea)

    # ensure the shape is correct arrays saved were rank 3, so this changes to rank 4 (last dimension represents channels)
    X = X.reshape([*X.shape, 1])
    Y = Y.reshape([*Y.shape, 1])

    # do train/test split!
    splitData = train_test_split(
        X, Y, pxArea, pxSpacing, test_size=0.2, random_state=101
    )
    pickle.dump(splitData, open(splitDataFile, "wb"))
    # extract individual bits
    X, X_test, Y, Y_test, pxArea, pxArea_test, pxSpacing, pxSpacing_test = splitData

del splitData  # as this variable is o longer required


M = X.shape[0]
MTest = X_test.shape[0]
imShape = (1, *X.shape[1:])

# properties for data augmentation
dataGenArgs = dict(
    rotation_range=20,
    width_shift_range=0.25,
    height_shift_range=0.25,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=False,  # DO NOT FLIP THE IMAGES FFS
    vertical_flip=False,
    fill_mode="nearest",
    data_format="channels_last",
    featurewise_center=False,
    featurewise_std_normalization=False,
    zca_whitening=False,
)

earlyStop = callbacks.EarlyStopping(
    patience=20,  # be a bit patient...
    min_delta=0,
    monitor="val_loss",
    restore_best_weights=True,
    mode="min",
)

reduceLR = callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    patience=10,
    factor=0.3,
    verbose=1,
    cooldown=5,
)

CALLBACKS = [earlyStop, reduceLR]

OPT = Adam(learning_rate=1e-2, beta_1=0.9, beta_2=0.999, amsgrad=False)

# other hyperparameters
BATCHSIZE = 8  # THIS MATTERS A LOT

keras.clear_session()

tf.random.set_seed(
    101
)  # FIXME!!! this is not sufficient to guarantee deterministic behaviour during fitting.

model = MultiResUnet(
    height=X.shape[1],
    width=X.shape[2],
    n_channels=1,
    layer_dropout_rate=None,
    block_dropout_rate=0.4,
)

model.compile(
    optimizer=OPT,
    loss="binary_crossentropy",
    metrics=["accuracy", metrics.MeanIoU(num_classes=2)],
)

fitHistory = model.fit(
    augmentImageSequence(X, Y, dataGenArgs, batchSize=BATCHSIZE),
    epochs=300,  # think about me...
    steps_per_epoch=M // BATCHSIZE,  # obvs
    workers=8,
    use_multiprocessing=True,
    validation_data=(X_test, Y_test.astype("float")),
    callbacks=CALLBACKS,
    verbose=1,
)


# Lets have a look at how fitting has proceeded
plt.figure(figsize=(15, 10))

plt.subplot(2, 1, 1)
plt.plot(fitHistory.history["loss"], label="train")
plt.plot(fitHistory.history["val_loss"], label="dev")
plt.ylabel("loss")
plt.ylim([0, 1])
plt.legend()
plt.xticks([])

plt.subplot(2, 1, 2)
plt.plot(fitHistory.history["mean_io_u"], label="train")
plt.plot(fitHistory.history["val_mean_io_u"], label="dev")
plt.ylim([0, 1])
plt.ylabel("mean iou")

plt.xlabel("epoch #")

plt.savefig(outputName + "_loss_history.svg")
plt.savefig(outputName + "_loss_history.png")

# FUNCTIONS FOR DOING STOCHASTIC PREDICTIONS...

def global_iou(predictions):
    """takes the iou of multiple different segmentations"""

    intersection = np.min(predictions, axis=0).sum()
    union = np.max(predictions, axis=0).sum()

    return intersection / union


def global_dsc(predictions):
    N = predictions.shape[0]
    numerator = N * np.min(predictions, axis=0).sum()
    denominator = predictions.sum()

    return numerator / denominator


def mean_pairwise_iou(predictions):
    # all combinations of inputs
    ious = [iou(a, b) for a, b in itertools.combinations(predictions, 2)]

    return np.mean(ious)


def predict_stochastic(model, N, X):
    """draw and summarise multiple predictions from a model
    Arguments:
        model {a model, for example a Keras model, with a predict method} -- is assumed to have some stochastic component, i.e. multiple
        N {int} -- the number of sample predictions to be drawn from the stochastic model
        X {numpy array, probably float} -- assumed to be already consistent with inputs to the model. MUST ONLY BE A SINGLE IMAGE AND NOT MULTIPLE STACKED!!!!!

    Returns:
        consensus {numpy array, boolean} -- pixelwise segmentation of x
        also various floats, representing different metrics for uncertainty and the outputs.
    """

    # draw N predictions from the model over x
    predictions = np.stack([model.predict(X) for n in range(N)], axis=0)

    # binarise
    consensus = np.mean(predictions, axis=0) > 0.5

    # metrics described in Roy et al...
    uncertainty = voxel_uncertainty(predictions)

    mpDsc = mean_pairwise_dsc(predictions)
    gDsc = global_dsc(predictions)

    mpIou = mean_pairwise_iou(predictions)
    gIou = global_iou(predictions)
    meanArea, stdArea = mean_std_area(predictions)

    return consensus, uncertainty, meanArea, stdArea, mpDsc, gDsc, mpIou, gIou


# Lets have a look at the  distribution of IoU, hausdorff distance and mean contour distance, for each example image in train and test set.

N = 15  # Roy et al use N=15

(
    predTest,
    uncertaintyTest,
    meanAreaTest,
    stdAreaTest,
    mpDscTest,
    gDscTest,
    mpIouTest,
    gIouTest,
) = map(
    np.array,
    zip(*[predict_stochastic(model, N, x.reshape(1, 208, 208, 1)) for x in X_test]),
)
(
    predTrain,
    uncertaintyTrain,
    meanAreaTrain,
    stdAreaTrain,
    mpDscTrain,
    gDscTrain,
    mpIouTrain,
    gIouTrain,
) = map(
    np.array, zip(*[predict_stochastic(model, N, x.reshape(1, 208, 208, 1)) for x in X])
)


predTrain = predTrain.reshape(*Y.shape)
predTest = predTest.reshape(*Y_test.shape)


# In[ ]:


# loop over th eexample axis, calculating metrics for each image separately
TrainIOU = [iou(Y[m, :, :, :], predTrain[m, :, :]) for m in range(M)]
TestIOU = [iou(Y_test[m, :, :, :], predTest[m, :, :]) for m in range(MTest)]

TrainDSC = [dsc(Y[m, :, :, :], predTrain[m, :, :]) for m in range(M)]
TestDSC = [dsc(Y_test[m, :, :, :], predTest[m, :, :]) for m in range(MTest)]

TrainHD = [
    symmetric_hausdorff_distance(Y[m, :, :, :], predTrain[m, :, :], pxSpacing[m])
    for m in range(M)
]
TestHD = [
    symmetric_hausdorff_distance(
        Y_test[m, :, :, :], predTest[m, :, :], pxSpacing_test[m]
    )
    for m in range(MTest)
]

TrainMCD = [
    mean_contour_distance(Y[m, :, :, :], predTrain[m, :, :], pxSpacing[m])
    for m in range(M)
]
TestMCD = [
    mean_contour_distance(Y_test[m, :, :, :], predTest[m, :, :], pxSpacing_test[m])
    for m in range(MTest)
]


# Lets have a look at network performance..

# Histograms for each of the metrics...

plt.figure(figsize=(20, 5))

plt.subplot(1, 4, 1)
plt.hist(
    TrainIOU, bins=np.arange(0, 1.05, 0.05), density=True, alpha=0.5, label="Train"
)
plt.hist(TestIOU, bins=np.arange(0, 1.05, 0.05), density=True, alpha=0.5, label="Test")
plt.xlabel("Intersection-over-Union")
plt.title(f"test mean = {np.mean(TestIOU):.3f}")

plt.ylabel("probability density")

plt.subplot(1, 4, 2)
plt.hist(
    TrainDSC, bins=np.arange(0, 1.05, 0.05), density=True, alpha=0.5, label="Train"
)
plt.hist(TestDSC, bins=np.arange(0, 1.05, 0.05), density=True, alpha=0.5, label="Test")
plt.xlabel("Dice-Sorenson coefficient")
plt.title(f"test mean = {np.mean(TestDSC):.3f}")


plt.subplot(1, 4, 3)
plt.hist(TrainHD, bins=np.arange(0, 125, 5), density=True, alpha=0.5, label="Train")
plt.hist(TestHD, bins=np.arange(0, 125, 5), density=True, alpha=0.5, label="Test")
plt.xlabel("Hausdorff Distance (mm)")
plt.title(f"test mean = {np.mean(TestHD):.3f}")


plt.subplot(1, 4, 4)
plt.hist(TrainMCD, bins=np.arange(0, 25, 2), density=True, alpha=0.5, label="Train")
plt.hist(TestMCD, bins=np.arange(0, 25, 2), density=True, alpha=0.5, label="Test")
plt.xlabel("Mean Contour Distance (mm)")
plt.title(f"test mean = {np.mean(TestMCD):.3f}")


plt.legend()


plt.savefig(outputName + "_metrics_histogram.svg")
plt.savefig(outputName + "_metrics_histogram.png")


# Now, are there correlations between predicted and real metric values?

# two ground-truths to be predicted

# 4 potential predictions to be made...
plt.figure(figsize=(20, 10))

metrics = [mpDscTest, gDscTest, mpIouTest, gIouTest]
metricNames = [
    "mean pairwise Dice coefficient",
    "global Dice coefficient",
    "mean pairwise IOU",
    "global IOU",
]


def scatter_with_title(x, y):
    plt.plot([0, 1], [0, 1], c="k")
    plt.scatter(x, y)
    r, p = pearsonr(x, y)
    mae = np.mean(np.abs(x - y))
    plt.title(f"r = {r:.2f}, MAE = {mae:.2f}")
    plt.axis("equal")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    return r


rBest = 0

for metricInd in range(4):
    plt.subplot(2, 4, metricInd + 1)
    rIOU = scatter_with_title(metrics[metricInd], TestDSC)
    if metricInd == 0:
        plt.ylabel("True DSC")

    plt.subplot(2, 4, 5 + metricInd)
    rDSC = scatter_with_title(metrics[metricInd], TestIOU)
    if metricInd == 0:
        plt.ylabel("True IOU")
    plt.xlabel(metricNames[metricInd])
    # extract the best correlation for model assessment
    rBest = np.max([rBest, rIOU, rDSC])

plt.savefig(outputName + "_QC_predictions.svg")
plt.savefig(outputName + "_QC_predictions.png")


# a few examples of the training set segmentations

negs = 25

egs = np.random.choice(range(M), negs, replace=False)

ncols = 5
nrows = np.ceil(negs / ncols)

plt.figure(figsize=(5 * ncols, 5 * nrows))

imShape = X.shape[1:-1]

for i in range(negs):
    plt.subplot(nrows, ncols, i + 1)

    manual, automated = (
        Y[egs[i], :, :].reshape(imShape),
        predTrain[egs[i], :, :].reshape(imShape) > 0.5,
    )

    pxS = pxSpacing[egs[i]]

    show_image_with_masks(
        image=X[egs[i], :, :].reshape(imShape),
        masks=[manual, automated],
        maskOptions=[{"linewidth": 1, "color": "y"}, {"linewidth": 1, "color": "r"}],
    )

    plt.title(
        "iou = "
        + f"{iou(manual,automated):.03}"
        + "\n"
        + "hd = "
        + f"{symmetric_hausdorff_distance(manual,automated,pxS):.03}"
        + "\n"
        + "mcd = "
        + f"{mean_contour_distance(manual,automated,pxS):.03}"
    )

plt.savefig(outputName + "_train_examples.svg")
plt.savefig(outputName + "_train_examples.png")


# Examples from the test set:

# In[ ]:


negs = 25

egs = np.random.choice(range(MTest), negs, replace=False)

ncols = 5
nrows = np.ceil(negs / ncols)

plt.figure(figsize=(5 * ncols, 5 * nrows))

imShape = X_test.shape[1:-1]

for i in range(negs):
    plt.subplot(nrows, ncols, i + 1)

    manual, automated = (
        Y_test[egs[i], :, :].reshape(imShape),
        predTest[egs[i], :, :].reshape(imShape) > 0.5,
    )

    pxS = pxSpacing_test[egs[i]]

    show_image_with_masks(
        image=X_test[egs[i], :, :].reshape(imShape),
        masks=[manual, automated],
        maskOptions=[{"linewidth": 1, "color": "y"}, {"linewidth": 1, "color": "r"}],
    )

    plt.title(
        "iou = "
        + f"{iou(manual,automated):.03}"
        + "\n"
        + "hd = "
        + f"{symmetric_hausdorff_distance(manual,automated,pxS):.03}"
        + "\n"
        + "mcd = "
        + f"{mean_contour_distance(manual,automated,pxS):.03}"
    )


plt.savefig(outputName + "_test_examples.svg")
plt.savefig(outputName + "_test_examples.png")


# Now, save the model for use elsewhere, along with some performance statistics

# need to save architecture and weight separately as custom loss functions cause issues with loading from a single .h5
# serialize model to JSON
model_json = model.to_json()
with open(outputName + ".json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights(outputName + ".h5")


# Write some metrics of model performance for the future....

# format a line to add to the csv

modelDetails = {
    "Filename": outputName,
    "TrainIOUMean": str(np.mean(TrainIOU)),
    "TrainIOUStd": str(np.std(TrainIOU)),
    "TestIOUMean": str(np.mean(TestIOU)),
    "TestIOUStd": str(np.std(TestIOU)),
    "TrainHDMean": str(np.mean(TrainHD)),
    "TrainHDStd": str(np.std(TrainHD)),
    "TestHDMean": str(np.mean(TestHD)),
    "TestHDStd": str(np.std(TestHD)),
    "TrainMCDMean": str(np.mean(TrainMCD)),
    "TrainMCDStd": str(np.std(TrainMCD)),
    "TestMCDMean": str(np.mean(TestMCD)),
    "TestMCDStd": str(np.std(TestMCD)),
    "BestR": str(rBest),
}

# if the file containing details of past models does not exist, then create it (with a header row)
historyFile = os.path.join(modelDir, "model_history.csv")

if not os.path.isfile(historyFile):
    fields = modelDetails.keys()
    with open(historyFile, "w+") as f:
        f.write(",".join(fields) + "\n")

# now write out the line of performance statistics.
with open(historyFile, "a") as f:
    f.write(",".join(modelDetails.values()) + "\n")
