#===============================================================================
# Copyright 2020 Intel Corporation
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
#===============================================================================

import os
import sys
import argparse

from a9a.a9a_loader import a9a
from ijcnn.ijcnn_loader import ijcnn
from connect.connect_loader import connect
from sensit.sensit_loader import sensit
from gisette.gisette_loader import gisette
from news20.news20_loader import news20
from mnist.mnist_loader import mnist
from rcv1.rcv1_loader import rcv1
from klaverjas.klaverjas_loader import klaverjas
from skin_segmentation.skin_segmentation_loader import skin_segmentation
from covertype.covertype_loader import covertype
from creditcard.creditcard_loader import creditcard
from codrnanorm.codrnanorm_loader import codrnanorm
from year_prediction.year_prediction_loader import year_prediction
from california_housing.california_housing_loader import california_housing
from fried.fried_loader import fried
from twodplanes.twodplanes_loader import twodplanes
from aloi.aloi_loader import aloi
from letter.letter_loader import letter
from medical_charges_nominal.medical_charges_nominal_loader import medical_charges_nominal
from yolanda.yolanda_loader import yolanda

dataset_loaders = {
    "a9a": a9a,
    "ijcnn": ijcnn,
    "connect": connect,
    "sensit": sensit,
    "gisette": gisette,
    "klaverjas": klaverjas,
    "mnist": mnist,
    "skin_segmentation": skin_segmentation,
    "covertype": covertype,
    "creditcard": creditcard,
    "codrnanorm": codrnanorm,
    "year_prediction": year_prediction,
    "california_housing": california_housing,
    "fried": fried,
    "2dplanes": twodplanes,
    "aloi": aloi,
    "letter": letter,
    "medical_charges_nominal": medical_charges_nominal,
    "yolanda": yolanda,
    # "news20": news20,
    # "rcv1": rcv1,
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Use \'-d\' or \'--datasets\' option to enumerate dataset(s) which should be downloaded')
    parser.add_argument('-l', '--list', action='store_const', const=True, help='list of available datasets')
    parser.add_argument('-d', '--datasets', type=str, nargs='*', help='datasets which should be downloaded')
    args = parser.parse_args()

    if args.list:
        for key in dataset_loaders.keys():
            print(key)
        sys.exit(0)

    root_dir = os.environ['DATASETSROOT']

    if args.datasets == None:
        for val in dataset_loaders.values():
            val(root_dir)
    elif len(args.datasets) == 0:
        print('Warning: Enumerate dataset(s) which should be downloaded')
    else:
        for key,val in dataset_loaders.items():
            if key in args.datasets:
                val(root_dir)
