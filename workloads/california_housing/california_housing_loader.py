# ===============================================================================
# Copyright 2021 Intel Corporation
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
# ===============================================================================

import os
import sys
import pandas as pd
import numpy as np
import scipy as sc
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split


def save_csr(filepath, x):
    np.save(filepath + "_data.npy", x.data)
    np.save(filepath + "_indices.npy", x.indices)
    np.save(filepath + "_indptr.npy", x.indptr)


def california_housing(root_dir=None):
    """
    Abstract:
    TDB

    Source:
    TDB

    Data Set Information:
    TDB

    Attribute Information:

    california_housing x train dataset (17935, 62061)
    california_housing y train dataset (17935, 1)
    california_housing x test dataset  (1993,  62061)
    california_housing y train dataset (1993,  1)
    """

    dataset_dir = os.path.join(root_dir, 'workloads', 'california_housing', 'dataset')

    try:
        os.makedirs(dataset_dir)
    except FileExistsError:
        pass

    filename_california_housing_x_train = os.path.join(
        dataset_dir, 'california_housing_x_train.csv')
    filename_california_housing_y_train = os.path.join(
        dataset_dir, 'california_housing_y_train.csv')
    filename_california_housing_x_test = os.path.join(
        dataset_dir, 'california_housing_x_test.csv')
    filename_california_housing_y_test = os.path.join(
        dataset_dir, 'california_housing_y_test.csv')
    x, y = fetch_california_housing(return_X_y=True, as_frame=True, data_home=dataset_dir)
    print('california_housing dataset is downloaded')

    print('reading CSV file...')
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.1, random_state=42)

    x_train.to_csv(os.path.join(
        dataset_dir, filename_california_housing_x_train), header=False, index=False)
    print(f'california_housing X train dataset {x_train.shape} is ready to be used')

    y_train.to_csv(os.path.join(
        dataset_dir, filename_california_housing_y_train), header=False, index=False)
    print(f'california_housing y train dataset {y_train.shape} is ready to be used')

    x_test.to_csv(os.path.join(
        dataset_dir, filename_california_housing_x_test), header=False, index=False)
    print(f'california_housing X test dataset {x_test.shape} is ready to be used')

    y_test.to_csv(os.path.join(
        dataset_dir, filename_california_housing_y_test), header=False, index=False)
    print(f'california_housing y test dataset {y_test.shape} is ready to be used')



if __name__ == '__main__':
    root_dir = os.environ['DATASETSROOT']
    california_housing(root_dir)
