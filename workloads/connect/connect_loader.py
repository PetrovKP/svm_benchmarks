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

def connect(root_dir=None):
    """
    Abstract:
    Predict next moves in game of connect-4. Represent basic model for AI usage in games, 
    a part of UCI repo >6.3K references in articles

    Source:
    https://archive.ics.uci.edu/ml/datasets/Connect-4

    Data Set Information:
    This database contains all legal 8-ply positions in the game of connect-4 
    in which neither player has won yet, and in which the next move is not forced. 
    x is the first player; o the second.
    The outcome class is the game theoretical value for the first player.

    Attribute Information:
    Used binary encoding for each feature (o, b, x), so the number of features is 42*3 = 126

    connect x train dataset (196045, 3)
    connect y train dataset (196045, 1)
    connect x test dataset  (49012,  3)
    connect y train dataset (49012,  1)
    """

    dataset_dir = os.path.join(root_dir, 'workloads', 'connect', 'dataset')

    try:
        os.makedirs(dataset_dir)
    except FileExistsError:
        pass

    filename_connect_x_train = os.path.join(dataset_dir, 'connect_x_train.csv')
    filename_connect_y_train = os.path.join(dataset_dir, 'connect_y_train.csv')
    filename_connect_x_test = os.path.join(dataset_dir, 'connect_x_test.csv')
    filename_connect_y_test = os.path.join(dataset_dir, 'connect_y_test.csv')

    x, y = fetch_openml(name='connect-4', version=1, return_X_y=True, as_frame=False, data_home=dataset_dir)
    x = pd.DataFrame(x.todense())
    y = pd.DataFrame(y)
    y = y.astype(int)

    print(x.shape, y.shape)

    print('connect dataset is downloaded')
    print('reading CSV file...')

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)
    x_train.to_csv(os.path.join(dataset_dir, filename_connect_x_train), header=False, index=False)
    print(f'connect x train dataset {x_train.shape} is ready to be used')

    y_train.to_csv(os.path.join(dataset_dir, filename_connect_y_train), header=False, index=False)
    print(f'connect y train dataset {y_train.shape} is ready to be used')

    x_test.to_csv(os.path.join(dataset_dir, filename_connect_x_test), header=False, index=False)
    print(f'connect x test dataset {x_test.shape} is ready to be used')

    y_test.to_csv(os.path.join(dataset_dir, filename_connect_y_test), header=False, index=False)
    print(f'connect y test dataset {y_test.shape} is ready to be used')


if __name__ == '__main__':
    root_dir = os.environ['DATASETSROOT']
    connect(root_dir)
