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

def a9a(root_dir=None):
    """
    Abstract:
    Predict whether income exceeds $50K/yr based on census data in the USA. 
    a part of UCI repo >6.3K references in articles
    Top #2 popular dataset from UCI (repo with ML workloads) >34K references in articles

    Source:
    http://archive.ics.uci.edu/ml/datasets/Adult

    Data Set Information:
    Extraction was done by Barry Becker from the 1994 Census database. 
    A set of reasonably clean records was extracted using 
    the following conditions: ((AAGE>16) && (AGI>100) && (AFNLWGT>1)&& (HRSWK>0))
    Prediction task is to determine whether a person makes over 50K a year.

    Attribute Information:
    The original Adult data set has 14 features, 
    among which six are continuous and eight are categorical. 
    In this data set, continuous features are discretized into quantiles, 
    and each quantile is represented by a binary feature. 
    Also, a categorical feature with m categories is converted to m binary features. 
    Details on how each feature is converted can be found in the beginning of each file from this page.

    a9a x train dataset (39073, 123)
    a9a y train dataset (39073, 1)
    a9a x test dataset  (9769,  123)
    a9a y train dataset (9769,  1)
    """

    dataset_dir = os.path.join(root_dir, 'workloads', 'a9a', 'dataset')

    try:
        os.makedirs(dataset_dir)
    except FileExistsError:
        pass

    filename_a9a_x_train = os.path.join(dataset_dir, 'a9a_x_train.csv')
    filename_a9a_y_train = os.path.join(dataset_dir, 'a9a_y_train.csv')
    filename_a9a_x_test = os.path.join(dataset_dir, 'a9a_x_test.csv')
    filename_a9a_y_test = os.path.join(dataset_dir, 'a9a_y_test.csv')

    x, y = fetch_openml(name='a9a', return_X_y=True, as_frame=False, data_home=dataset_dir)
    x = pd.DataFrame(x.todense())
    y = pd.DataFrame(y)

    y[y == -1] = 0

    print('a9a dataset is downloaded')
    print('reading CSV file...')

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=11)
    x_train.to_csv(os.path.join(dataset_dir, filename_a9a_x_train), header=False, index=False)
    print(f'a9a x train dataset {x_train.shape} is ready to be used')

    y_train.to_csv(os.path.join(dataset_dir, filename_a9a_y_train), header=False, index=False)
    print(f'a9a y train dataset {y_train.shape} is ready to be used')

    x_test.to_csv(os.path.join(dataset_dir, filename_a9a_x_test), header=False, index=False)
    print(f'a9a x test dataset {x_test.shape} is ready to be used')

    y_test.to_csv(os.path.join(dataset_dir, filename_a9a_y_test), header=False, index=False)
    print(f'a9a y test dataset {y_test.shape} is ready to be used')


if __name__ == '__main__':
    root_dir = os.environ['DATASETSROOT']
    a9a(root_dir)
