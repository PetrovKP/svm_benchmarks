
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

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

def save_csr(filepath, x):
    np.save(filepath  + "_data.npy", x.data)
    np.save(filepath  + "_indices.npy", x.indices)
    np.save(filepath  + "_indptr.npy", x.indptr)

def rcv1(root_dir=None):
    """
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

    filename_rcv1_X_train = os.path.join(dataset_dir, 'rcv1_small_x_train')
    filename_rcv1_y_train = os.path.join(dataset_dir, 'rcv1_small_y_train.npy')
    filename_rcv1_X_test = os.path.join(dataset_dir, 'rcv1_small_x_test')
    filename_rcv1_y_test = os.path.join(dataset_dir, 'rcv1_small_y_test.npy')

    rcv1_train_data_url = 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/rcv1_train.binary.bz2'

    filename_train_data = os.path.join(dataset_dir, 'rcv1_train.bz2')
    if not os.path.exists(filename_train_data):
        urlretrieve(rcv1_train_data_url, filename_train_data)

    # rcv1_test_data_url = 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/rcv1_test.binary.bz2'
    # filename_test_data = os.path.join(dataset_dir, 'rcv1_test.bz2')
    # if not os.path.exists(filename_test_data):
    #     urlretrieve(rcv1_test_data_url, filename_test_data)

    print('rcv1_small dataset is downloaded')
    print('reading CSV file...')

    X_train, y_train = load_svmlight_file(filename_train_data)
    X_test, y_test = load_svmlight_file(filename_test_data)

    save_csr(os.path.join(dataset_dir, filename_rcv1_X_train), X_train)
    print(f'rcv1_small X train dataset {X_train.shape} is ready to be used')

    np.save(os.path.join(dataset_dir, filename_rcv1_y_train), y_train)
    print(f'rcv1_small y train dataset {y_train.shape} is ready to be used')

    save_csr(os.path.join(dataset_dir, filename_rcv1_X_train), X_test)
    print(f'rcv1_small X test dataset {X_test.shape} is ready to be used')

    np.save(os.path.join(dataset_dir, filename_rcv1_y_train), y_test)
    print(f'rcv1_small y train dataset {y_test.shape} is ready to be used')


if __name__ == '__main__':
    root_dir = os.environ['DATASETSROOT']
    rcv1(root_dir)