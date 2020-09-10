# ===============================================================================
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
# ===============================================================================

from sklearn.metrics import accuracy_score
import argparse
import os
import timeit
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser()
parser.add_argument('--workload', type=str, default='all',
                    help='Choose worload for SVM. Default all worloads')
parser.add_argument('--library', type=str, default='idp_sklearn',
                    choices=['sklearn', 'thunder', 'cuml', 'idp_sklearn'],
                    help='Choose library for SVM. Default idp_sklearn')

args = parser.parse_args()
arg_name_workload = args.workload
arg_name_library = args.library
times_worloads = []

if arg_name_library == 'idp_sklearn':
    from daal4py.sklearn import patch_sklearn
    patch_sklearn()
    from sklearn.svm import SVC
elif arg_name_library == 'sklearn':
    from sklearn.svm import SVC
elif arg_name_library == 'thunder':
    from thundersvm import SVC
elif arg_name_library == 'cuml':
    from cuml import SVC


cache_size = 8*1024  # 8 GB
tol = 1e-3

workloads = {
    'a9a':               {'C': 500.0,  'kernel': 'rbf'},
    'ijcnn':             {'C': 1000.0, 'kernel': 'linear'},
    'sensit':            {'C': 500.0,  'kernel': 'linear'},
    'connect':           {'C': 100.0,  'kernel': 'linear'},
    'gisette':           {'C': 0.0015, 'kernel': 'linear'},
    'mnist':             {'C': 100.0,  'kernel': 'linear'},
    'klaverjas':         {'C': 1.0,    'kernel': 'rbf'},
    'skin_segmentation': {'C': 1.0,    'kernel': 'rbf'},
    'covertype':         {'C': 100.0,  'kernel': 'rbf'},
    'creditcard':        {'C': 100.0,  'kernel': 'linear'},
    'codrnanorm':        {'C': 1000.0, 'kernel': 'linear'},
}


def load_data(name_workload):
    root_dir = os.environ['DATASETSROOT']
    dataset_dir = os.path.join(root_dir, 'workloads', name_workload, 'dataset')
    x_train_path = os.path.join(
        dataset_dir, '{}_x_train.csv'.format(name_workload))
    x_train = pd.read_csv(x_train_path, header=None)
    x_test_path = os.path.join(
        dataset_dir, '{}_x_test.csv'.format(name_workload))
    x_test = pd.read_csv(x_test_path, header=None)
    y_train_path = os.path.join(
        dataset_dir, '{}_y_train.csv'.format(name_workload))
    y_train = pd.read_csv(y_train_path, header=None)
    y_test_path = os.path.join(
        dataset_dir, '{}_y_test.csv'.format(name_workload))
    y_test = pd.read_csv(y_test_path, header=None)
    return x_train, x_test, y_train, y_test


def run_svm_workload(workload_name, x_train, x_test, y_train, y_test, C=1.0, kernel='linear'):
    gamma = 1.0 / x_train.shape[1]
    # Create C-SVM classifier
    clf = SVC(C=C, kernel=kernel, max_iter=-1, cache_size=cache_size,
              tol=tol, gamma=gamma)

    t0 = timeit.default_timer()
    clf.fit(x_train, y_train)
    t1 = timeit.default_timer()
    time_fit_train_run = t1 - t0

    t0 = timeit.default_timer()
    y_pred_train = clf.predict(x_train)
    t1 = timeit.default_timer()
    time_predict_train_run = t1 - t0
    acc_train = accuracy_score(y_train, y_pred_train)

    t0 = timeit.default_timer()
    y_pred = clf.predict(x_test)
    t1 = timeit.default_timer()
    time_predict_test_run = t1 - t0
    acc_test = accuracy_score(y_test, y_pred)

    print('{}: n_samples:{}; n_features:{}; n_classes:{}; C:{}; kernel:{}'.format(
        workload_name, x_train.shape[0], x_train.shape[1], len(np.unique(y_train)), C, kernel))
    print('Fit   [Train n_samples:{:6d}]: {:6.2f} sec'.format(
        x_train.shape[0], time_fit_train_run))
    print('Infer [Train n_samples:{:6d}]: {:6.2f} sec. accuracy_score: {:.3f}'.format(
        x_train.shape[0], time_predict_train_run, acc_train))
    print('Infer [Test  n_samples:{:6d}]: {:6.2f} sec. accuracy_score: {:.3f}'.format(
        x_test.shape[0], time_predict_test_run, acc_test))


for name_workload, params in workloads.items():
    if arg_name_workload in [name_workload, 'all']:
        x_train, x_test, y_train, y_test = load_data(name_workload)
        run_svm_workload(name_workload, x_train, x_test, y_train, y_test,
                         C=params['C'], kernel=params['kernel'])
