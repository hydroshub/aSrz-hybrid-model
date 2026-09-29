# ----------------------------
# Data paths
# ----------------------------
DATA_DIR = "/path/to/Caravan/forcing_and_target_csvs/"
ATTR_PATHS = [
    "/path/to/Caravan/attributes/camels/attributes_hydroatlas_camels.csv",
    "/path/to/Caravan/attributes/camels/attributes_other_camels.csv",
]


# ----------------------------
# Input / Output columns
# ----------------------------
INPUT_COLS = [
    "total_precipitation_sum",
    "temperature_2m_mean",
    "surface_net_solar_radiation_mean",
    "surface_net_thermal_radiation_mean",
    "lai_GIMMS_filled"
]

TARGET_DAILY_COLS = [
    "streamflow",      # Q
    "fluxcom_E",       # ET
    "nsidc_SWE"        # Snow
]

TARGET_MONTHLY_COLS = [
    "grace_TWSA"       # TWS anomaly
]

ATTR_COLS = [
    "cover_forest",
    "cover_shrub",
    "cover_grass",
    "cover_crop",
    "cover_others",
    "slp_dg_sav",
    "ele_mt_sav",
    "cly_pc_sav",
    "snd_pc_sav",
    "snw_pc_syr",
    "ari_ix_sav",
]

# ----------------------------
# Time settings
# ----------------------------
START_YEAR = "1996"
END_YEAR = "2020"


# ----------------------------
# Model hyperparameters
# ----------------------------
PARAM_RANGES = {
    "snow_tsnow":   [-5.0, 0.0],       # lower bound for full snow
    "snow_train":   [0.0, 5.0],        # upper bound for full rain
    "snow_fmt":     [0.5, 8.0],        # snow melt factor (mm/°C/day)
    "split_k":      [0.1, 10],         # split 
    "avai_efmax":   [0.1, 1.0],        # max ET efficiency (unitless)
    "avai_cap_base":  [0.0, 1000.0],   # available water capacity (mm)
    "avai_wetpoint99":[0.01, 0.99],    # threshold for full ET activation (unitless)
    "avai_beta":    [0.05, 0.95],      # LAI response curve 
    "fast_kf":      [0.05, 0.95],      # fast flow coefficient
    "fast_perc":    [0.1, 20.0],       # percolation from fast to slow store (mm/day)
    "slow_ks":      [1e-4, 1e-1],      # slow flow recession rate
    "river_maxbas": [1.0, 5.0],        # MAXBAS routing kernel width (days)
}

# ----------------------------
# Training configuration
# ----------------------------
BATCH_SIZE = 2048
NUM_EPOCHS = 100
EARLY_STOP_PATIENCE = 10
LOSS_WEIGHTS = [1.0, 1.0, 1.0, 1.0]