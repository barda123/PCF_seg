# PCF_seg

This repo centres on quantification of pericardial fat deposits, using a CNN, and data from the UK Biobank.

I start by preprocessing manually-segmented images, created using cvi42 software. 

Files containing manual contours are used to pull out the relevant dicom files from the (vast) UK Biobank dataset. The contours are used to make "masks" of the same dimension as the images.

Image-mask pairs are then split into train and test sets, and used to train/evaluate a CNN - which is currently based around a U-net (Ronneberger 2015).

