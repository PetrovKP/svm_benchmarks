# SVM Benchmarks

### Dependencies

```bash
pip install -r requirements.txt
```

### Setting up the environment

```
Directory for saving and unloading datasets while the benchmark is running
```bash
export DATASETSROOT=<enter absolute path>
```

### Download datasets

```bash
python workloads/load_datasets.py
```

### Running benchmarks

For runs all workloads:

```bash
python benchmarks/svm_workload_run.py
```

For runs of the selected workload:

```bash
python benchmarks/svm_workload_run.py --workload a9a
```

You can choose library: `sklearn`, `cuml`, `thundersvm`, `sklearn-intelex`
For runs of the selected library.
By default using sklearn with oneDAL optimizations (`sklearn-intelex`).
Example for thundersvm library:

```bash
python benchmarks/svm_workload_run.py --library thundersvm
```

*NOTE: for thundersvm/cuml runs need thundersvm/cuml library.
Can you download with help pip or conda*

You can also choose a task for the svm: `svc`, `svc_proba`, `svr`. By default `svc`
