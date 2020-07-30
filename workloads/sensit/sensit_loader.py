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

def sensit(root_dir=None):
    """
    Abstract:
    Vehicle type classification for intelligent transportation systems based on info from sensors.
    The real-world task proposed by Department of Electrical and 
    Computer Engineering, University of Wisconsin-Madison

    Source:
    https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass.html

    Data Set Information:
    The task of classifying the types of moving vehicles in a distributed, 
    wireless sensor network is investigated. Specifically, 
    based onan extensive real world experiment, 
    we have compiled a data set that consists of 820 MByte raw time series data, 
    70 MByte of pre-processed, extracted spectral feature vectors, 
    and baseline classification results using the maximum likelihood classifier.
    See here: http://www.ecs.umass.edu/~mduarte/images/JPDC.pdf

    Attribute Information:
    Regenerate features by the authors' matlab scripts (see Sec. C of Appendix A), 
    then randomly select 10% instances from the noise class so that the 
    class proportion is 1:1:2 (AAV:DW:noise). 
    The training/testing sets are from a random 80% and 20% split of the data.

    sensit x train dataset (78822, 100)
    sensit y train dataset (78822, 1)
    sensit x test dataset  (19706, 100)
    sensit y train dataset (19706, 1)
    """

    dataset_dir = os.path.join(root_dir, 'workloads', 'sensit', 'dataset')

    try:
        os.makedirs(dataset_dir)
    except FileExistsError:
        pass

    filename_sensit_x_train = os.path.join(dataset_dir, 'sensit_x_train.csv')
    filename_sensit_y_train = os.path.join(dataset_dir, 'sensit_y_train.csv')
    filename_sensit_x_test = os.path.join(dataset_dir, 'sensit_x_test.csv')
    filename_sensit_y_test = os.path.join(dataset_dir, 'sensit_y_test.csv')

    x, y = fetch_openml(name='SensIT-Vehicle-Combined', return_X_y=True, as_frame=False, data_home=dataset_dir)
    x = pd.DataFrame(x.todense())
    y = pd.DataFrame(y)
    y = y.astype(int)

    print(x.shape, y.shape)

    print('sensit dataset is downloaded')
    print('reading CSV file...')

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    x_train.to_csv(os.path.join(dataset_dir, filename_sensit_x_train), header=False, index=False)
    print(f'sensit x train dataset {x_train.shape} is ready to be used')

    y_train.to_csv(os.path.join(dataset_dir, filename_sensit_y_train), header=False, index=False)
    print(f'sensit y train dataset {y_train.shape} is ready to be used')

    x_test.to_csv(os.path.join(dataset_dir, filename_sensit_x_test), header=False, index=False)
    print(f'sensit x test dataset {x_test.shape} is ready to be used')

    y_test.to_csv(os.path.join(dataset_dir, filename_sensit_y_test), header=False, index=False)
    print(f'sensit y test dataset {y_test.shape} is ready to be used')


if __name__ == '__main__':
    root_dir = os.environ['DATASETSROOT']
    sensit(root_dir)
