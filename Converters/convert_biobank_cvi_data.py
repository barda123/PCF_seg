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

"""The main converting script.

This script uncompresses the UK Biobank zip files, sort the DICOM files according to
 information provided in the manifest.csv spreadsheet, parse manual annotated contours
 from cvi42 xml files, and finally read the matching DICOM and cvi42 contours and save
 them into a nifti image. 
"""


import os, csv, glob, re, time
import pandas as pd
import dateutil.parser
from biobank_utils import *
import parse_cvi42_xml


def repl(m):
    return '{}{}-{}-20{}'.format(m.group(1), m.group(2), m.group(3), m.group(4))


# Read the lines in the manifest.csv file and check whether the date format consists a comma, which needs to be removed
# since it affects parsing the file.
def process_manifest(name, name2):
    with open(name2, 'w') as f2:
        with open(name, 'r') as f:
            for line in f:
                line2 = re.sub('([A-Z])(\w{2}) (\d{1,2}), 20(\d{2})', repl, line)
                f2.write(line2)


# # The authentication file (application id + password)
# ukbkey = '/homes/wbai/ukbkey'

# Paths
data_path = '/vol/vipdata/data/biobank/cardiac/Application_2964/data'
cvi42_list = []

for sub_path in sorted(os.listdir(data_path)):
    sub_path = os.path.join(data_path, sub_path)
    for eid in sorted(os.listdir(sub_path)):
        print(eid)
        data_dir = os.path.join(sub_path, eid)
        if os.path.exists(os.path.join(data_dir, '{0}_cvi42.zip'.format(eid))):
            cvi42_list += [eid]

            # # Remove old files which may be wrong conversion
            # os.system('rm -f {0}/sa.nii.gz'.format(data_dir))
            # os.system('rm -f {0}/la_*ch.nii.gz'.format(data_dir))
            # os.system('rm -f {0}/label_*.nii.gz'.format(data_dir))

            # Decompress zip files
            files = glob.glob('{0}/{1}_*.zip'.format(data_dir, eid))
            dicom_dir = os.path.join(data_dir, 'dicom')
            if not os.path.exists(dicom_dir):
                os.mkdir(dicom_dir)

            for f in files:
                if os.path.basename(f) == '{0}_cvi42.zip'.format(eid):
                    os.system('unzip -o {0} -d {1}'.format(f, data_dir))
                else:
                    os.system('unzip -o {0} -d {1}'.format(f, dicom_dir))

                    # Organise the dicom files
                    # Process the manifest file
                    process_manifest(os.path.join(dicom_dir, 'manifest.csv'), \
                                     os.path.join(dicom_dir, 'manifest2.csv'))
                    df2 = pd.read_csv(os.path.join(dicom_dir, 'manifest2.csv'), error_bad_lines=False)

                    # Group the files into subdirectories
                    for series_name, series_df in df2.groupby('series discription'):
                        series_dir = os.path.join(dicom_dir, series_name)
                        if not os.path.exists(series_dir):
                            os.mkdir(series_dir)
                        series_files = [os.path.join(dicom_dir, x) for x in series_df['filename']]
                        os.system('mv {0} {1}'.format(' '.join(series_files), series_dir))

            # Parse cvi42 xml file
            cvi42_contours_dir = os.path.join(data_dir, 'cvi42_contours')
            if not os.path.exists(cvi42_contours_dir):
                os.mkdir(cvi42_contours_dir)
            xml_name = os.path.join(data_dir, '{0}_cvi42.cvi42wsx'.format(eid))
            parse_cvi42_xml.parseFile(xml_name, cvi42_contours_dir)

            # Rare cases when no dicom files exist
            # e.g. 12xxxxx/1270299
            if not os.listdir(dicom_dir):
                print('Warning: empty dicom directory! Skip this one.')
                continue

            # Convert dicom to nifti
            dset = Biobank_Dataset(dicom_dir, cvi42_contours_dir)
            dset.read_dicom_images()
            dset.convert_dicom_to_nifti(data_dir)

            # Remove intermediate files
            os.system('rm -rf {0} {1}'.format(dicom_dir, cvi42_contours_dir))
            os.system('rm -f {0}'.format(xml_name))
