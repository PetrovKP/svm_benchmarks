# SVM Benchmarks

### Dependencies

```bash
conda install -c intel scikit-learn pandas daal4py==0.2020.2
```

### Setting up the environment

For use Intel DAAL solvers which makes it possible to get a performance gain without any code changes.
```bash
export USE_DAAL4PY_SKLEARN=yes
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
