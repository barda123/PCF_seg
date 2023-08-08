"""
    Module:  dicom_to_nifti.py
    Nov 28, 2019
    David Gobbi
    https://gist.github.com/dgobbi/ab71f5128aa43f0d33a41775cb2bcca6
"""

import pydicom
import nibabel as nib
import numpy as np

import argparse
import glob
import sys
import os

usage = """
Convert DICOM (MRI) files to NIFTI:
  dicom_to_nifti -i /path/to/input/ -o /path/to/output.nii.gz
  The input is a directory that contains a series of DICOM files.
  It must contain only one series, corresponding to one 3D volume.
  You can use either ".nii" or ".nii.gz" as the suffix for the
  nifti files (use .nii.gz to write compressed files).
  The output nifti file will be float32 (single precision).
"""


def write_nifti(filename, vol, affine):
    """Write a nifti file with an affine matrix."""
    output = nib.Nifti1Image(vol.T, affine)
    nib.save(output, filename)


def create_affine(ipp, iop, ps):
    """Generate a NIFTI affine matrix from DICOM IPP and IOP attributes.
    The ipp (ImagePositionPatient) parameter should an Nx3 array, and
    the iop (ImageOrientationPatient) parameter should be Nx6, where
    N is the number of DICOM slices in the series.
    The return values are the NIFTI affine matrix and the NIFTI pixdim.
    Note the the output will use DICOM anatomical coordinates:
    x increases towards the left, y increases towards the back.
    """
    # solve Ax = b where x is slope, intecept
    n = ipp.shape[0]
    A = np.column_stack([np.arange(n), np.ones(n)])
    x, r, rank, s = np.linalg.lstsq(A, ipp, rcond=None)
    # round small values to zero
    x[(np.abs(x) < 1e-6)] = 0.0
    vec = x[0, :]  # slope
    pos = x[1, :]  # intercept

    # pixel spacing should be the same for all image
    spacing = np.ones(3)
    spacing[0:2] = ps[0, :]
    if np.sum(np.abs(ps - spacing[0:2])) > spacing[0] * 1e-6:
        sys.stderr.write("Pixel spacing is inconsistent!\n")

    # compute slice spacing
    spacing[2] = np.round(np.sqrt(np.sum(np.square(vec))), 7)

    # get the orientation
    iop_average = np.mean(iop, axis=0)
    u = iop_average[0:3]
    u /= np.sqrt(np.sum(np.square(u)))
    v = iop_average[3:6]
    v /= np.sqrt(np.sum(np.square(v)))

    # round small values to zero
    u[(np.abs(u) < 1e-6)] = 0.0
    v[(np.abs(v) < 1e-6)] = 0.0

    # create the matrix
    mat = np.eye(4)
    mat[0:3, 0] = u * spacing[0]
    mat[0:3, 1] = v * spacing[1]
    mat[0:3, 2] = vec
    mat[0:3, 3] = pos

    # check whether slice vec is orthogonal to iop vectors
    dv = np.dot(vec, np.cross(u, v))
    qfac = np.sign(dv)
    if np.abs(qfac * dv - spacing[2]) > 1e-6:
        sys.stderr.write("Non-orthogonal volume!\n")

    # compute the nifti pixdim array
    pixdim = np.hstack([np.array(qfac), spacing])

    return mat, pixdim


def convert_coords(vol, mat):
    """Convert a volume from DICOM coords to NIFTI coords or vice-versa.
    For DICOM, x increases to the left and y increases to the back.
    For NIFTI, x increases to the right and y increases to the front.
    The conversion is done in-place (volume and matrix are modified).
    """
    # the x direction and y direction are flipped
    convmat = np.eye(4)
    convmat[0, 0] = -1.0
    convmat[1, 1] = -1.0

    # apply the coordinate change to the matrix
    mat[:] = np.dot(convmat, mat)

    # look for x and y elements with greatest magnitude
    xabs = np.abs(mat[:, 0])
    yabs = np.abs(mat[:, 1])
    xmaxi = np.argmax(xabs)
    yabs[xmaxi] = 0.0
    ymaxi = np.argmax(yabs)

    # re-order the data to ensure these elements aren't negative
    # (this may impact the way that the image is displayed, if the
    # software that displays the image ignores the matrix).
    if mat[xmaxi, 0] < 0.0:
        # flip x
        vol[:] = np.flip(vol, 2)
        mat[:, 3] += mat[:, 0] * (vol.shape[2] - 1)
        mat[:, 0] = -mat[:, 0]
    if mat[ymaxi, 1] < 0.0:
        # flip y
        vol[:] = np.flip(vol, 1)
        mat[:, 3] += mat[:, 1] * (vol.shape[1] - 1)
        mat[:, 1] = -mat[:, 1]

    # eliminate "-0.0" (negative zero) in the matrix
    mat[mat == 0.0] = 0.0
    return vol, mat


def find_dicom_files(path):
    """Search for DICOM files at the provided location."""
    if os.path.isdir(path):
        # check for common DICOM suffixes
        for ext in ("*.dcm", "*.DCM", "*.dc", "*.DC", "*.IMG"):
            pattern = os.path.join(path, ext)
            files = glob.glob(pattern)
            if files:
                break
        # if no files with DICOM suffix are found, get all files
        if not files:
            pattern = os.path.join(path, "*")
            contents = glob.glob(pattern)
            files = [f for f in contents if os.path.isfile(f)]
    elif os.path.isfile(path):
        # if 'path' is a file (not a folder), return the file
        files = [args.input]
    else:
        sys.stderr.write("Cannot open %s\n" % (path,))
        return []

    return files


def load_dicom_series(files):
    """Load a series of dicom files and return a list of datasets.
    The resulting list will be sorted by InstanceNumber.
    """
    # start by sorting filenames lexically
    sorted_files = sorted(files)

    # create list of tuples (InstanceNumber, DataSet)
    dataset_list = []
    for f in files:
        ds = pydicom.dcmread(f)
        try:
            i = int(ds.InstanceNumber)
        except (AttributeError, ValueError):
            i = -1
        dataset_list.append((i, ds))

    # sort by InstanceNumber (the first element of each tuple)
    dataset_list.sort(key=lambda t: t[0])

    # get the dataset from each tuple
    series = [t[1] for t in dataset_list]

    return series


def dicom_to_volume(dicom_series):
    """Convert a DICOM series into a float32 volume with orientation.
    The input should be a list of 'dataset' objects from pydicom.
    The output is a tuple (voxel_array, voxel_spacing, affine_matrix)
    """
    # Create numpy arrays for volume, pixel spacing (ps),
    # slice position (ipp or ImagePositinPatient), and
    # slice orientation (iop or ImageOrientationPatient)
    n = len(dicom_series)
    shape = (n,) + dicom_series[0].pixel_array.shape
    vol = np.empty(shape, dtype=np.float32)
    ps = np.empty((n, 2), dtype=np.float64)
    ipp = np.empty((n, 3), dtype=np.float64)
    iop = np.empty((n, 6), dtype=np.float64)

    for i, ds in enumerate(dicom_series):
        # create a single complex-valued image from real,imag
        image = ds.pixel_array
        try:
            slope = float(ds.RescaleSlope)
        except (AttributeError, ValueError):
            slope = 1.0
        try:
            intercept = float(ds.RescaleIntercept)
        except (AttributeError, ValueError):
            intercept = 0.0
        vol[i, :, :] = image * slope + intercept
        ps[i, :] = dicom_series[i].PixelSpacing
        ipp[i, :] = dicom_series[i].ImagePositionPatient
        iop[i, :] = dicom_series[i].ImageOrientationPatient

    # create nibabel-style affine matrix and pixdim
    # (these give DICOM LPS coords, not NIFTI RAS coords)
    affine, pixdim = create_affine(ipp, iop, ps)
    return vol, pixdim, affine


def main(argv):
    """Callable entry point."""
    parser = argparse.ArgumentParser(prog=argv[0], usage=usage)

    parser.add_argument(
        "-i", "--input", required=True, help="Input folder for complex images."
    )
    parser.add_argument(
        "-o", "--output", required=True, help="Output folder for nifti images."
    )
    parser.add_argument("-v", "--verbose", action="count", help="Turn on verbosity.")

    args = parser.parse_args(argv[1:])

    if args.verbose:
        sys.stdout.write("Converting DICOM to NIFTI.\n")
        sys.stdout.write("DICOM: %s\n" % args.input)
        sys.stdout.write("NIFTI: %s\n" % args.output)
        sys.stdout.flush()

    # create a list of the dicom files
    files = find_dicom_files(args.input)
    if not files:
        sys.stderr.write("No DICOM files found.\n")
        return 1

    # load the files to create a list of slices
    series = load_dicom_series(files)
    if not series:
        sys.stderr.write("Unable to read DICOM files.\n")
        return 1

    # reconstruct the images into a volume
    vol, pixdim, mat = dicom_to_volume(series)

    # convert DICOM coords to NIFTI coords (in-place)
    convert_coords(vol, mat)

    # write the file (note that the volume will be transposed so that the
    # indices will be ordered the way that nibabel prefers: pixel,row,slice
    # instead of slice,row,pixel)
    write_nifti(args.output, vol, mat)


if __name__ == "__main__":
    r = main(sys.argv)
    if r is not None:
        sys.exit(r)


def get_affine(d):
    # Copyright 2017.
    # Author: Wenjia Bai, Biomedical Image Analysis Group, Imperial College London.
    #
    # Licensed under the Apache License, Version 2.0 (the "License");
    # you may not use this file except in compliance with the License.
    # You may obtain a copy of the License at
    #
    #     http://www.apache.org/licenses/LICENSE-2.0
    #
    # Unless required by applicable law or agreed to in writing, software
    # distributed under the License is distributed on an "AS IS" BASIS,
    # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    # See the License for the specific language governing permissions and
    # limitations under the License.
    # ==============================================================================
    X = d.Columns
    Y = d.Rows
    Z = 1  # because in this context all images are 2D

    dx = float(d.PixelSpacing[1])
    dy = float(d.PixelSpacing[0])

    # DICOM coordinate (LPS)
    #  x: left
    #  y: posterior
    #  z: superior
    # Nifti coordinate (RAS)
    #  x: right
    #  y: anterior
    #  z: superior
    # Therefore, to transform between DICOM and Nifti, the x and y coordinates need to be negated.
    # Refer to
    # http://nifti.nimh.nih.gov/pub/dist/src/niftilib/nifti1.h
    # http://nifti.nimh.nih.gov/nifti-1/documentation/nifti1fields/nifti1fields_pages/figqformusage

    # The coordinate of the upper-left voxel of the first and second slices
    pos_ul = np.array([float(x) for x in d.ImagePositionPatient])
    pos_ul[:2] = -pos_ul[:2]

    # Image orientation
    axis_x = np.array([float(x) for x in d.ImageOrientationPatient[:3]])
    axis_y = np.array([float(x) for x in d.ImageOrientationPatient[3:]])
    axis_x[:2] = -axis_x[:2]
    axis_y[:2] = -axis_y[:2]

    if Z >= 2:
        # Read a dicom file at the second slice
        d2 = dicom.read_file(os.path.join(dir[1], sorted(os.listdir(dir[1]))[0]))
        pos_ul2 = np.array([float(x) for x in d2.ImagePositionPatient])
        pos_ul2[:2] = -pos_ul2[:2]
        axis_z = pos_ul2 - pos_ul
        axis_z = axis_z / np.linalg.norm(axis_z)
    else:
        axis_z = np.cross(axis_x, axis_y)

    # Determine the z spacing
    if hasattr(d, "SpacingBetweenSlices"):
        dz = float(d.SpacingBetweenSlices)
    elif Z >= 2:
        print(
            "Warning: can not find attribute SpacingBetweenSlices. Calculate from two successive slices."
        )
        dz = float(np.linalg.norm(pos_ul2 - pos_ul))
    else:
        print(
            "Warning: can not find attribute SpacingBetweenSlices. Use attribute SliceThickness instead."
        )
        dz = float(d.SliceThickness)

    # Affine matrix which converts the voxel coordinate to world coordinate
    affine = np.eye(4)
    affine[:3, 0] = axis_x * dx
    affine[:3, 1] = axis_y * dy
    affine[:3, 2] = axis_z * dz
    affine[:3, 3] = pos_ul

    return affine
