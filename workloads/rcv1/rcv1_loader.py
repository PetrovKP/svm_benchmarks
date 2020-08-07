
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
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml, load_svmlight_file
from urllib.request import urlretrieve

def save_csr(filepath, x):
    np.save(filepath  + "_data.npy", x.data)
    np.save(filepath  + "_indices.npy", x.indices)
    np.save(filepath  + "_indptr.npy", x.indptr)

def rcv1(root_dir=None):
    """
    GISETTE is a handwritten digit recognition problem.
    The problem is to separate the highly confusable digits '4' and '9'.
    This dataset is one of five datasets of the NIPS 2003 feature selection challenge.
    The digits have been size-normalized and centered in a fixed-size image of dimension 28x28.
    The original data were modified for the purpose of the feature selection challenge.
    In particular, pixels were samples at random in the middle top part of
    the feature containing the information necessary to disambiguate 4 from 9
    and higher order features were created as products of these pixels
    to plunge the problem in a higher dimensional feature space.
    We also added a number of distractor features called 'probes' having no predictive power.
    The order of the features and patterns were randomized.
    Preprocessing: The data set is also available at UCI.
    Because the labels of testing set are not available,
    here we use the validation set (gisette_valid.data and gisette_valid.labels)
    as the testing set. The training data (gisette_train)
    are feature-wisely scaled to [-1,1].
    Then the testing data (gisette_valid) are scaled based on the same scaling factors for the training data.
    rcv1 X train dataset (20242 , 47236)
    rcv1 y train dataset (20242 , 1)
    rcv1 X test dataset  (677399 , 47236)
    rcv1 y train dataset (677399 , 1)
    """

    dataset_dir = os.path.join(root_dir, 'workloads', 'rcv1', 'dataset')

    try:
        os.makedirs(dataset_dir)
    except FileExistsError:
        pass

    filename_rcv1_X_train = os.path.join(dataset_dir, 'rcv1_x_train')
    filename_rcv1_y_train = os.path.join(dataset_dir, 'rcv1_y_train.npy')
    filename_rcv1_X_test = os.path.join(dataset_dir, 'rcv1_x_test')
    filename_rcv1_y_test = os.path.join(dataset_dir, 'rcv1_y_test.npy')

    rcv1_train_data_url = 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/rcv1_train.binary.bz2'

    filename_train_data = os.path.join(dataset_dir, 'rcv1_train.bz2')
    if not os.path.exists(filename_train_data):
        urlretrieve(rcv1_train_data_url, filename_train_data)

    rcv1_test_data_url = 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/rcv1_test.binary.bz2'
    filename_test_data = os.path.join(dataset_dir, 'rcv1_test.bz2')
    if not os.path.exists(filename_test_data):
        urlretrieve(rcv1_test_data_url, filename_test_data)

    print('rcv1 dataset is downloaded')
    print('reading CSV file...')

    X_train, y_train = load_svmlight_file(filename_train_data)
    X_test, y_test = load_svmlight_file(filename_test_data)

    save_csr(os.path.join(dataset_dir, filename_rcv1_X_train), X_train)
    print(f'rcv1 X train dataset {X_train.shape} is ready to be used')

    np.save(os.path.join(dataset_dir, filename_rcv1_y_train), y_train)
    print(f'rcv1 y train dataset {y_train.shape} is ready to be used')

    save_csr(os.path.join(dataset_dir, filename_rcv1_X_test), X_test)
    print(f'rcv1 X test dataset {X_test.shape} is ready to be used')

    np.save(os.path.join(dataset_dir, filename_rcv1_y_test), y_test)
    print(f'rcv1 y train dataset {y_test.shape} is ready to be used')


if __name__ == '__main__':
    root_dir = os.environ['DATASETSROOT']
    rcv1(root_dir)