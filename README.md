# Spatiotemporal Dynamics of Active Root Zone Storage Revealed from Hybrid Machine Learning

This repository provides a straightforward PyTorch implementation of the model proposed in the manuscript [Spatiotemporal dynamics of active root zone storage revealed from hybrid machine learning](https://doi.org/10.1029/2025WR042803).

The globally operationalized version of the model can be found in our [MOREDO repository](https://github.com/hydroshub/moredo).

The model is a differentiable, process-based hybrid model that simulates water balance at basin scales. It jointly predicts streamflow (Q), evapotranspiration (ET), snow water equivalent (SWE), and terrestrial water storage anomaly (TWSA). The model focuses on diagnosing the active root zone water storage dynamics.

If you have any questions or suggestions for the code, or find you a bug, please let us know. You are welcome to raise an issue or contact us at gblougouras(at)bgc-jena.mpg.de.

---

## File Overview

| File | Description |
|------|-------------|
| `config.py` | **Configuration file.** Sets data paths, input/output variable names, time range, model hyperparameter bounds, and training settings. |
| `data.py` | **Data module.** Handles CSV loading, attribute preprocessing (standardization, land cover aggregation), dataset caching, time-block cross-validation (`TimeBlockCV`), and PyTorch dataset wrapping (`UnifiedDataset`). |
| `main.py` | **Entry point.** Parses command-line arguments, loads data, sets up cross-validation splits, builds the model, and starts training. |
| `model.py` | **Model module.** Implements the hybrid hydrological model `HybridModel`, including a neural-network parameter generator (`StaticParamGenerator`) and process simulation modules (`HydroProcess`: snow, soil moisture, fast/slow flow, river routing). |
| `utils.py` | **Utilities.** Contains random seed setup, loss functions (masked R², NSE, Pearson/Spearman correlation), and the full training loop (`train_model`) with early stopping and learning rate scheduling. |
| `basin_list_us_camels.txt` | **Basin list.** CAMELS-US basin IDs, one per line. Passed as the `--basin_list` argument. |

---

## Data Requirements

Data is sourced from the [Caravan](https://github.com/kratzert/Caravan) dataset. For more information about alternative data employed, gap-filling LAI, catchment filtering and other methodological steps, please refer to the manuscript. The expected directory structure is:

```
/path/to/Caravan/
├── forcing_and_target_csvs/
│   ├── camels_01013500_merged.csv   # one CSV per basin
│   └── ...
└── attributes/
    └── camels/
        ├── attributes_hydroatlas_camels.csv
        └── attributes_other_camels.csv
```

Each basin CSV must contain the following columns:

- **Forcings (inputs):** `total_precipitation_sum`, `temperature_2m_mean`, `surface_net_solar_radiation_mean`, `surface_net_thermal_radiation_mean`, `lai_GIMMS_filled`
- **Daily targets:** `streamflow`, `fluxcom_E` (ET), `nsidc_SWE`
- **Monthly target:** `grace_TWSA`

---

## Quick Start

### 1. Install dependencies

```bash
pip install torch numpy pandas scikit-learn tqdm
```

### 2. Edit `config.py`

Update the data paths to match your local setup:

```python
DATA_DIR = "/path/to/Caravan/forcing_and_target_csvs/"
ATTR_PATHS = [
    "/path/to/Caravan/attributes/camels/attributes_hydroatlas_camels.csv",
    "/path/to/Caravan/attributes/camels/attributes_other_camels.csv",
]
```

### 3. Run training

```bash
python main.py \
    --basin_list basin_list_us_camels.txt \
    --config config.py \
    --fold_id 0 \
    --seed 42 \
    --save_dir model_output \
    --num_epochs 100
```

### Command-line arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `--basin_list` | Yes | Path to basin list file |
| `--config` | Yes | Path to `config.py` |
| `--fold_id` | Yes | Cross-validation fold index (0–4) |
| `--seed` | No | Random seed (default: `42`) |
| `--save_dir` | No | Output directory (default: `model_output`) |
| `--num_epochs` | No | Number of training epochs (default: `100`) |
| `--show_progress` | No | Show per-timestep progress bar during forward pass |

---

## Output Files

After training, the following files are saved under `--save_dir`:

```
model_output/
├── best_model_seed42_fold0.pt            # Best model weights (lowest validation loss)
└── training_progress_seed42_fold0.txt    # Per-epoch train/val loss log
```

---

## Cross-Validation

Time-block cross-validation (`TimeBlockCV`) splits data by year:

- **Spin-up period:** 1996–2000 — used for model warm-up only, excluded from training/validation/test
- **CV period:** 2001–2020 — split into 5 folds of ~4 years each
- **Validation set:** one year sampled every 4 years from the non-test years (`val_stride=4, val_offset=3`)

Run fold indices 0–4 separately to cover the full dataset.

---

## Model Architecture

```
Input: meteorological forcings (P, T, Radiation, LAI) + static attributes (land cover, terrain, soil)
  │
  ├── StaticParamGenerator (MLP) --> basin-specific ecohydrological parameters
  │
  └── HydroProcess (sequential simulation)
        ├── Rain-snow partitioning
        ├── Snow bucket
        ├── Water partitioning
        ├── Ecosystem water bucket
        ├── Fast-flow bucket
        ├── Slow-flow bucket
        └── River routing (triangular MAXBAS kernel)

Output (train mode): Q, ET, SWE, TWSA (monthly anomaly)
Output (full mode):  all fluxes, states, and parameters
```

---

## Caching

On the first run, loading all basins can take several minutes. The processed dataset is automatically saved to a `cache/` directory. Subsequent runs with identical configuration will load directly from cache, significantly reducing startup time.

To force a fresh load, delete the contents of the `cache/` directory.
