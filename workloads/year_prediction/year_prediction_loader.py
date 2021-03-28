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
from urllib.request import urlretrieve

def year_prediction(root_dir=None):
    """
    """

    dataset_dir = os.path.join(root_dir, 'workloads', 'year_prediction', 'dataset')

    try:
        os.makedirs(dataset_dir)
    except FileExistsError:
        pass

    filename_year_prediction_X_train = os.path.join(dataset_dir, 'year_prediction_x_train.csv')
    filename_year_prediction_y_train = os.path.join(dataset_dir, 'year_prediction_y_train.csv')
    filename_year_prediction_X_test = os.path.join(dataset_dir, 'year_prediction_x_test.csv')
    filename_year_prediction_y_test = os.path.join(dataset_dir, 'year_prediction_y_test.csv')

    year_prediction_train_data_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00203/YearPredictionMSD.txt.zip'

    filename_train_data = os.path.join(dataset_dir, 'YearPredictionMSD.txt.zip')
    if not os.path.exists(filename_train_data):
        urlretrieve(year_prediction_train_data_url, filename_train_data)

    # year_prediction_test_data_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/year_prediction/GISETTE/year_prediction_valid.data'
    # filename_test_data = os.path.join(dataset_dir, 'year_prediction_valid.data')
    # if not os.path.exists(filename_test_data):
    #     urlretrieve(year_prediction_test_data_url, filename_test_data)

    # year_prediction_test_labels_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/year_prediction/year_prediction_valid.labels'
    # filename_test_labels = os.path.join(dataset_dir, 'year_prediction_valid.labels')
    # if not os.path.exists(filename_test_labels):
    #     urlretrieve(year_prediction_test_labels_url, filename_test_labels)

    print('year_prediction dataset is downloaded')
    print('reading CSV file...')

    df = pd.read_csv(filename_train_data)

    num_train = 463715

    X_train = df.iloc[:num_train, 1:]

    X_train.to_csv(os.path.join(dataset_dir, filename_year_prediction_X_train), header=False, index=False)
    print(f'year_prediction X train dataset {X_train.shape} is ready to be used')

    y_train = df.iloc[:num_train, 0]

    y_train.to_csv(os.path.join(dataset_dir, filename_year_prediction_y_train), header=False, index=False)
    print(f'year_prediction y train dataset {y_train.shape} is ready to be used')

    num_test = num_train

    X_test = df.iloc[num_test:, 1:]

    X_test.to_csv(os.path.join(dataset_dir, filename_year_prediction_X_test), header=False, index=False)
    print(f'year_prediction X test dataset {X_test.shape} is ready to be used')

    y_test = df.iloc[num_test:, 0]

    y_test.to_csv(os.path.join(dataset_dir, filename_year_prediction_y_test), header=False, index=False)
    print(f'year_prediction y train dataset {y_test.shape} is ready to be used')

if __name__ == '__main__':
    root_dir = os.environ['DATASETSROOT']
    year_prediction(root_dir)
