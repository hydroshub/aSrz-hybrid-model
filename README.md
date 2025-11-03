# Spatiotemporal dynamics of active root zone storage revealed from hybrid machine learning 

## Overview
The code demonstrates a straightforward, torch implementation of the model proposed in the preprint **Spatiotemporal dynamics of active root zone storage revealed from hybrid machine learning**.

***!!!** The **libs** modules will be publicly available upon acceptance of the manuscript. **!!!***

If you have any questions or suggestions with the code or find a bug, please let us know. You are welcome to raise an issue or contact us at gblougouras(at)bgc-jena.mpg.de.

## Repository structure
```text
├── libs/        # Directory containing modules to be called in the main training script
│   ├── model.py # Contains all necessary classes and functions to build the model used in the manuscript
│   ├── utils.py # Functions to guide the training process
├── train.py     # Executable code to train the model (more information on the below sections)
├── README.md    # This file
└── LICENSE      # Project license 
```

## Quick Start

### Dependencies
The code in this repository was generated and executed in `python=3.10.13` by using `torch=2.4.1` (with `cuda-version=12.9`). Additional packages needed to execute the code are `numpy=2.1.3` and `pandas=2.2.3`.

### Download and prepare the datasets
```
- Caravan: https://doi.org/10.5281/ZENODO.15529786
- MODIS LAI: https://doi.org/10.7289/V5TT4P69
- FLUXCOM X-BASE ET: https://doi.org/10.18160/5NZG-JMJE
- NSIDC SWE: https://doi.org/10.5067/0GGPB220EX6A
- GRACE TWSA: https://doi.org/10.5067/TEMSC-3MJ634
```

All data are aggregated on the basin-scale. For more information about alternative data employed, gap-filling LAI, catchment filtering and other methodological steps, please refer to the manuscript. 

After acquiring all the basin-scale data, in order to train the model, the data must already be stored in pickle format (see L72-L78 of `train.py`). In our study, we use 5fold temporal cross validation, but in ```train.py```, for convenience, we assume that the data is already split in the three sets (training, validation, testing). To this end, the pickled python dictionary contains three keys, with  training, validation and test **tensor datasets** (```torch.utils.data.TensorDataset```). 

Each of these three tensor datasets is a tuple: **TensorDataset(x, y, z)**, where:
```
1. x -> input, with shape [B, T, V] (number of basins, daily timesteps, inputs inlc. daily forcing and static attributes)
2. y -> daily targets, with shape [B, T, V] (number of basins, daily timesteps, daily targets incl. Q, ET, SWE)
3. z -> monthly targets, with shape [B, M, K] (number of basins, monthly timesteps, monthly targets incl. TWSA)
```

### Run the code

In order to train the model, all you need to do is run ```train.py```.

The provided code will execute **a single run** (1 fold, 1 seed): 

```python train.py```

The arguments already have default values, but you can always modify in-line, e.g.: 

```python train.py --save_dir results_dir --data_path pickle_dir```.

For training across many folds/seeds, a batch script is best.