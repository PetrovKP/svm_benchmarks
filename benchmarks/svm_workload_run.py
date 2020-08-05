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

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import timeit
import os
import argparse

# from sklearn.svm import SVC
from thundersvm import SVC

from sklearn.metrics import accuracy_score

cache_size = 8*1024 # 8 GB
tol = 1e-3

def load_data(name_workload):
    root_dir    = os.environ['DATASETSROOT']
    dataset_dir = os.path.join(root_dir, 'workloads', name_workload, 'dataset')
    x_train_path = os.path.join(dataset_dir, '{}_x_train.csv'.format(name_workload))
    x_train = pd.read_csv(x_train_path, header=None)
    x_test_path = os.path.join(dataset_dir, '{}_x_test.csv'.format(name_workload))
    x_test  = pd.read_csv(x_test_path, header=None)
    y_train_path = os.path.join(dataset_dir, '{}_y_train.csv'.format(name_workload))
    y_train = pd.read_csv(y_train_path, header=None)
    y_test_path = os.path.join(dataset_dir, '{}_y_test.csv'.format(name_workload))
    y_test  = pd.read_csv(y_test_path, header=None)
    return x_train, x_test, y_train, y_test

def run_svm_workload(workload_name, x_train, x_test, y_train, y_test, C=1.0, kernel='linear'):
    gamma = 1.0 / x_train.shape[1]
    # Create C-SVM classifier
    clf = SVC(C=C, kernel=kernel, max_iter=-1, cache_size=cache_size, 
        tol=tol, gamma=gamma)    

    t0 = timeit.default_timer()
    clf.fit(x_train, y_train)
    t1 = timeit.default_timer()
    time_run = t1 - t0

    y_pred = clf.predict(x_test)
    acc = accuracy_score(y_pred, y_test)
    print('Workload {} {}x{} running {:.2f} (sec) have accuracy_score: {:.3f}'.format(workload_name, 
        x_train.shape[0], x_train.shape[1], time_run, acc))
    return time_run

parser = argparse.ArgumentParser()
parser.add_argument('--workload', type=str, default='all', 
    help='Choose worload for SVM. Default all worloads')
args = parser.parse_args()
arg_name_workload = args.workload
times_worloads = []

name_workload = 'a9a'
if arg_name_workload in [name_workload, 'all']:
    # Predict whether income exceeds $50K/yr based on census data in the USA
    x_train, x_test, y_train, y_test = load_data(name_workload)
    times_worloads.append(run_svm_workload(name_workload, x_train, x_test, y_train, y_test, 
                                           C=500, kernel='rbf'))

name_workload = 'ijcnn'
if arg_name_workload in [name_workload, 'all']:
    # Predict connections in an online social network
    x_train, x_test, y_train, y_test = load_data(name_workload)
    times_worloads.append(run_svm_workload(name_workload, x_train, x_test, y_train, y_test, 
                                           C=1000, kernel='linear'))

name_workload = 'sensit'
if arg_name_workload in [name_workload, 'all']:
    # Vehicle type classification for intelligent transportation systems based on info from sensors.
    name_workload = 'sensit'
    x_train, x_test, y_train, y_test = load_data(name_workload)
    times_worloads.append(run_svm_workload(name_workload, x_train, x_test, y_train, y_test, 
                                           C=500, kernel='linear'))

name_workload = 'connect'
if arg_name_workload in [name_workload, 'all']:
    # Predict next moves in game of connect-4.
    x_train, x_test, y_train, y_test = load_data(name_workload)
    times_worloads.append(run_svm_workload(name_workload, x_train, x_test, y_train, y_test, 
                                           C=100, kernel='linear'))

if len(times_worloads) == 0:
    raise 'Not workload {} for this benchmarks'.format(name_workload)
