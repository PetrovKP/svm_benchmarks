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

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

def letter(root_dir=None):
    """
    TDB
    """

    dataset_dir = os.path.join(root_dir, 'workloads', 'letter', 'dataset')

    try:
        os.makedirs(dataset_dir)
    except FileExistsError:
        pass

    filename_letter_x_train = os.path.join(dataset_dir, 'letter_x_train.csv')
    filename_letter_y_train = os.path.join(dataset_dir, 'letter_y_train.csv')
    filename_letter_x_test = os.path.join(dataset_dir, 'letter_x_test.csv')
    filename_letter_y_test = os.path.join(dataset_dir, 'letter_y_test.csv')

    x, y = fetch_openml(name='BNG(letter,5000,1)', return_X_y=True, as_frame=True, data_home=dataset_dir)

    y = y.cat.codes

    print(x.shape, y.shape)
    print('letter dataset is downloaded')
    print('reading CSV file...')

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=42)
    x_train.to_csv(os.path.join(dataset_dir, filename_letter_x_train), header=False, index=False)
    print(f'letter x train dataset {x_train.shape} is ready to be used')

    y_train.to_csv(os.path.join(dataset_dir, filename_letter_y_train), header=False, index=False)
    print(f'letter y train dataset {y_train.shape} is ready to be used')

    x_test.to_csv(os.path.join(dataset_dir, filename_letter_x_test), header=False, index=False)
    print(f'letter x test dataset {x_test.shape} is ready to be used')

    y_test.to_csv(os.path.join(dataset_dir, filename_letter_y_test), header=False, index=False)
    print(f'letter y test dataset {y_test.shape} is ready to be used')


if __name__ == '__main__':
    root_dir = os.environ['DATASETSROOT']
    letter(root_dir)
