# Sharpened Dimensionality Reduction with Neural Networks(SDR-NNP)
## Table of contents
* [About](#about)
* [Setup](#setup)
* [Experimental results](#experimental-results)


## About
Sharpened dimensionality reduction with neural networks (SDR-NNP) consists of a neural network that imitates sharpened dimensionality reduction, which is used to sharpen the original data before dimensionality reduction to create visually segregrated sample clusters. 


## Setup
Code is written in Python with a C++ wrapper for the SDR implementation and is available [here](https://github.com/youngjookim/sdr-nnp/Code/python-nnp).
```
https://github.com/youngjookim/sdr-nnp
```

# SDR-NNP

Python scripts for the SDR-NNP experiment can be found on the `python-nnp` folder

## Brief description of each script

- [01_convert_datasets.py](01_convert_datasets.py): convert datasets from npy or csv to the SDR format. 
    - Usage: `python 01_convert_datasets.py <data folder>`, where
    - `<data folder>` is a folder that contains one folder per dataset

- [02_run_sdr.py](02_run_sdr.py): run SDR for a set of datasets.
    - Usage: `python 02_run_sdr.py <input folder> <num consumers> <sdr command>`, where
    - `<input folder>` is the folder that contains one folder per experiment, default is `<data folder>/_sdr`
    - `<num consumers>` is the number of SDR processes to be run in parallel
    - `<sdr command>` is the full path to the SDR binary

- [03_run_nnp.py](03_run_nnp.py): run the NNP experiment over data generated by SDR.
    - Usage: `python 03_run_nnp.py <input dir> <datasets>`, where
    - `<input folder>` is the folder that contains one folder per experiment, default is `<data folder>/_sdr`
    - `<datasets>` is a comma-separated list of datasets to run NNP over. Can be a single dataset, without comma.

- [04_compute_metrics.py](04_compute_metrics.py): compute metrics for all experiments.
    - Usage: `python 04_compute_metrics.py <data folder> <datasets>`, where
    - `<data folder>` is a folder that contains one folder per dataset plus the SDR results inside `<data folder>/_sdr`
    - `<datasets>` is a comma-separated list of datasets to run NNP over. Can be a single dataset, without comma.

- [05_consolid_results.py](05_consolid_results.py): consolidate results for all experiments into a single `all_metrics.csv` file.
    - Usage: `python 05_consolid_results.py <data folder>`, where
    - `<data folder>` is a folder that contains one folder per dataset plus the SDR results inside `<data folder>/_sdr`

- [nnp.py](nnp.py): implementation of NNP.

- [sdr.py](sdr.py): python wrapper to the SDR binary.

- [metrics.py](metrics.py): implementation of the metrics.

- [params.py](params.py): definition of the parameter grid used in the experiment


## Experimental results
All the experimental results and the appendix for this project are in the paper in the root directory.
