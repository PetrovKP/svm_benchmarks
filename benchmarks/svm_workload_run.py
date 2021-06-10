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

import argparse
import os
import timeit
import numpy as np
import pandas as pd
import warnings

from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser()
parser.add_argument('--workload', type=str, default='all',
                    help='Choose worload for SVM. Default all worloads')
parser.add_argument('--task', type=str, default='svc',
                    choices=['svc', 'svc_proba', 'svr', 'nusvc', 'nusvr'],
                    help='Choose task for SVM. Default svc')
parser.add_argument('--library', type=str, default='sklearn-intelex',
                    choices=['sklearn', 'onedal',
                             'thunder', 'cuml', 'sklearn-intelex'],
                    help='Choose library for SVM. Default sklearn-intelex')

args = parser.parse_args()
arg_name_workload = args.workload
arg_name_library = args.library
arg_name_task = args.task
times_worloads = []

if arg_name_library == 'sklearn-intelex':
    from sklearnex import patch_sklearn
    patch_sklearn()
    from sklearn.svm import SVR, SVC, NuSVC, NuSVR
    from sklearn.metrics import mean_squared_error, accuracy_score, log_loss
elif arg_name_library == 'onedal':
    from onedal.svm import SVR, SVC, NuSVC, NuSVR
    from sklearn.metrics import mean_squared_error, accuracy_score, log_loss
elif arg_name_library == 'sklearn':
    from sklearn.svm import SVR, SVC, NuSVC, NuSVR
    from sklearn.metrics import mean_squared_error, accuracy_score, log_loss
elif arg_name_library == 'thunder':
    from thundersvm import SVR, SVC, NuSVC, NuSVR
    from sklearn.metrics import mean_squared_error, accuracy_score, log_loss
elif arg_name_library == 'cuml':
    from cuml import SVR, SVC
    NuSVC = None
    NuSVR = None
    from cuml.metrics import mean_squared_error, accuracy_score, log_loss


cache_size = 2*1024  # 2 GB
tol = 1e-3


svc_workloads = {
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
    # 'aloi':              {'C': 10.0,   'kernel': 'rbf'},
    # 'letter':            {'C': 10.0,   'kernel': 'rbf'},
}

poly_str = 'polynomial' if arg_name_library == 'thunder' else 'poly'

svr_workloads = {
    'california_housing':  {'C': 0.1,  'kernel': poly_str},
    'fried': {'C': 2.0,  'kernel': 'rbf'},
    'twodplanes': {'C': 10.0,  'kernel': 'rbf', 'epsilon': 0.5},
    'medical_charges_nominal': {'C': 10.0, 'kernel': poly_str, 'epsilon': 0.1, 'degree': 2},
    'yolanda':  {'C': 10,  'kernel': 'rbf'},
    # 'yolanda':  {'C': 10,  'kernel': 'linear'},
    'year_prediction':  {'C': 1.0,  'kernel': 'linear'},

}

nusvc_workloads = {
    'a9a':               {'nu': 0.25,  'kernel': 'rbf'}, # (39073, 123) (9769, 123)
    'ijcnn':             {'nu': 0.17,  'kernel': 'linear'}, # (153344, 22) (38337,  22)
    'sensit':            {'nu': 0.5,   'kernel': 'linear'}, # (78822, 100) (19706, 100)
    'connect':           {'nu': 0.25,  'kernel': 'linear'}, # (60801, 123) (6756, 123)
    'gisette':           {'nu': 0.07,  'kernel': 'linear'}, # (6000, 5000) (1000, 5000)
    'mnist':             {'nu': 0.5,   'kernel': 'linear'}, # (60000, 784) (10000, 784)
    'klaverjas':         {'nu': 0.7,   'kernel': 'rbf'}, # (196308, 32) (785233, 32)
    'skin_segmentation': {'nu': 0.01,  'kernel': 'rbf'}, # (196045, 3) (49012, 3)
    'covertype':         {'nu': 0.01,  'kernel': 'rbf'}, # (348607, 54) (232405, 54)
    'creditcard':        {'nu': 0.002, 'kernel': 'linear'}, # (256326, 29) (28481, 29)
    'codrnanorm':        {'nu': 0.15,  'kernel': 'linear'}, # (390852, 8) (97713, 8)
}

nusvr_workloads = {
    'california_housing':      {'nu': 0.17, 'C': 0.1,  'kernel': poly_str}, # (18576, 8), (2064, 8)
    'fried':                   {'nu': 0.8,  'C': 2.0,  'kernel': 'rbf'}, # (32614, 10), (8154, 10)
    'twodplanes':              {'nu': 0.5,  'C': 10.0, 'kernel': 'rbf'}, # (106288, 10), (70859, 10)
    'medical_charges_nominal': {'nu': 0.5,  'C': 10.0, 'kernel': poly_str, 'degree': 2}, # (130452, 11), (32613, 11)
    'yolanda':                 {'nu': 0.5,  'C': 10.0, 'kernel': 'rbf'}, # (240000, 100), (160000, 100)
    'year_prediction':         {'nu': 0.5,  'C': 1.0,  'kernel': 'linear'}, # (463715, 90), (51629, 90)
}


def load_data(name_workload):
    root_dir = os.environ['DATASETSROOT']
    dataset_dir = os.path.join(root_dir, 'workloads', name_workload, 'dataset')
    x_train_path = os.path.join(
        dataset_dir, '{}_x_train.csv'.format(name_workload))
    x_train = pd.read_csv(x_train_path, header=None, dtype=np.float64)
    x_test_path = os.path.join(
        dataset_dir, '{}_x_test.csv'.format(name_workload))
    x_test = pd.read_csv(x_test_path, header=None, dtype=np.float64)
    y_train_path = os.path.join(
        dataset_dir, '{}_y_train.csv'.format(name_workload))
    y_train = pd.read_csv(y_train_path, header=None, dtype=np.float64)
    y_test_path = os.path.join(
        dataset_dir, '{}_y_test.csv'.format(name_workload))
    y_test = pd.read_csv(y_test_path, header=None, dtype=np.float64)
    return x_train, x_test, y_train, y_test


def run_svm_workload(workload_name, x_train, x_test, y_train, y_test, task, **params):
    if task == 'svr':
        scater = StandardScaler().fit(x_train, y_train)
        x_train = scater.transform(x_train)
        x_test = scater.transform(x_test)

        if workload_name in ['medical_charges_nominal']:
            scater = StandardScaler().fit(y_train)
            y_train = scater.transform(y_train)
            y_test = scater.transform(y_test)

        clf = SVR(**params, cache_size=cache_size, tol=tol)
        def metric_call(x, y): return mean_squared_error(x, y, squared=True)
        def predict_call(clf, x): return clf.predict(x)
        metric_name = 'rmse'
    elif task == 'svc':
        clf = SVC(**params, cache_size=cache_size, tol=tol)
        metric_call = accuracy_score
        metric_name = 'accuracy'
        def predict_call(clf, x): return clf.predict(x)
    elif task == 'svc_proba':
        clf = SVC(**params, cache_size=cache_size, tol=tol, probability=True)
        def predict_call(clf, x): return clf.predict_proba(x)
        metric_call = log_loss
        metric_name = 'log_loss'
    elif task == 'nusvr':
        if arg_name_library == 'cuml':
            raise ValueError("cuml don't have NuSVR")

        scater = StandardScaler().fit(x_train, y_train)
        x_train = scater.transform(x_train)
        x_test = scater.transform(x_test)

        if workload_name in ['medical_charges_nominal']:
            scater = StandardScaler().fit(y_train)
            y_train = scater.transform(y_train)
            y_test = scater.transform(y_test)

        clf = NuSVR(**params, cache_size=cache_size, tol=tol)
        def metric_call(x, y): return mean_squared_error(x, y, squared=True)
        def predict_call(clf, x): return clf.predict(x)
        metric_name = 'rmse'
    elif task == 'nusvc':
        if arg_name_library == 'cuml':
            raise ValueError("cuml don't have NuSVC")

        clf = NuSVC(**params, cache_size=cache_size, tol=tol)
        metric_call = accuracy_score
        metric_name = 'accuracy'
        def predict_call(clf, x): return clf.predict(x)
    elif task == 'nusvc_proba':
        if arg_name_library == 'cuml':
            raise ValueError("cuml don't have NuSVC")

        clf = NuSVC(**params, cache_size=cache_size, tol=tol, probability=True)
        def predict_call(clf, x): return clf.predict_proba(x)
        metric_call = log_loss
        metric_name = 'log_loss'
    else:
        raise ValueError('Incorrect name a task {}'.format(task))

    print('{}:{{ n_samples : {}; n_features : {}; n_classes : {} }};'.format(
        workload_name, x_train.shape[0], x_train.shape[1], len(np.unique(y_train))))
    print("params:{", end='')
    for i, v in params.items():
        print(" ", i, ":", v, end=';')
    print(end=' };\n')
    t0 = timeit.default_timer()
    clf.fit(x_train, y_train)
    t1 = timeit.default_timer()
    time_fit_train_run = t1 - t0

    n_sv = clf.n_support_ if arg_name_library == 'thunder' else clf.support_.shape[0]
    print('Fit   [Train n_samples:{:6d}]: {:6.2f} sec. n_sv: {}'.format(
        x_train.shape[0], time_fit_train_run, n_sv))

    t0 = timeit.default_timer()
    pred_train = predict_call(clf, x_train)
    t1 = timeit.default_timer()
    time_predict_train_run = t1 - t0
    acc_train = metric_call(y_train, pred_train)

    print('Infer [Train n_samples:{:6d}]: {:6.2f} sec. {} score: {:.5f}'.format(
        x_train.shape[0], time_predict_train_run, metric_name, acc_train))

    t0 = timeit.default_timer()
    pred_test = predict_call(clf, x_test)
    t1 = timeit.default_timer()
    time_predict_test_run = t1 - t0
    acc_test = metric_call(y_test, pred_test)

    print('Infer [Test  n_samples:{:6d}]: {:6.2f} sec. {} score: {:.5f}'.format(
        x_test.shape[0], time_predict_test_run, metric_name, acc_test))


if arg_name_task in ['svc', 'svc_proba']:
    workloads = svc_workloads
elif arg_name_task in ['svr']:
    workloads = svr_workloads
elif arg_name_task in ['nusvc', 'nusvc_proba']:
    workloads = nusvc_workloads
elif arg_name_task in ['nusvr']:
    workloads = nusvr_workloads
else:
    workloads = {}

for name_workload, params in workloads.items():
    if arg_name_workload in [name_workload, 'all']:
        try:
            x_train, x_test, y_train, y_test = load_data(name_workload)
            run_svm_workload(name_workload, x_train, x_test,
                             y_train, y_test, arg_name_task, **params)

        except FileNotFoundError:
            print('WARNING! Workload: {} not found'.format(name_workload))
