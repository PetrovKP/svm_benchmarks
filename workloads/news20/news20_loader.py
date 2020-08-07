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
import scipy as sc
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

def save_csr(filepath, x):
    np.save(filepath  + "_data.npy", x.data)
    np.save(filepath  + "_indices.npy", x.indices)
    np.save(filepath  + "_indptr.npy", x.indptr)


def news20(root_dir=None):
    """
    Abstract:
    Predict connections in an online social network.
    Kaggle ML challenge proposed by International Joint Conference on Neural Networks (IJCNN), 
    winners presented their solution on the conference

    Source:
    https://www.kaggle.com/c/socialNetwork

    Data Set Information:
    TDB

    Attribute Information:

    news20 x train dataset (17935, 62061)
    news20 y train dataset (17935, 1)
    news20 x test dataset  (1993,  62061)
    news20 y train dataset (1993,  1)
    """

    dataset_dir = os.path.join(root_dir, 'workloads', 'news20', 'dataset')

    try:
        os.makedirs(dataset_dir)
    except FileExistsError:
        pass

    filename_news20_x_train = os.path.join(dataset_dir, 'news20_x_train')
    filename_news20_y_train = os.path.join(dataset_dir, 'news20_y_train.npy')
    filename_news20_x_test = os.path.join(dataset_dir, 'news20_x_test')
    filename_news20_y_test = os.path.join(dataset_dir, 'news20_y_test.npy')

    x, y = fetch_openml(name='news20', return_X_y=True, as_frame=False, data_home=dataset_dir)
    x = sc.sparse.csr_matrix(x)
    print('news20 dataset is downloaded')
    print('reading CSV file...')

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)

    save_csr(os.path.join(dataset_dir, filename_news20_x_train), x_train)
    print(f'news20 x train dataset {x_train.shape} is ready to be used')

    
    np.save(os.path.join(dataset_dir, filename_news20_y_train), y_train)
    print(f'news20 y train dataset {y_train.shape} is ready to be used')

    save_csr(os.path.join(dataset_dir, filename_news20_x_test), x_test)
    print(f'news20 x test dataset {x_test.shape} is ready to be used')

    np.save(os.path.join(dataset_dir, filename_news20_y_test), y_test)
    print(f'news20 y train dataset {y_test.shape} is ready to be used')


if __name__ == '__main__':
    root_dir = os.environ['DATASETSROOT']
    news20(root_dir)
