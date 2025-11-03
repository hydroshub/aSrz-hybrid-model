import os
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import pickle
import pandas as pd


# Set up model parameter ranges, training configuration, attribute columns
param_ranges = {
    "snow_tsnow":   [-5.0, 0.0],       # lower bound for full snow
    "snow_train":   [0.0, 5.0],        # upper bound for full rain
    "snow_fmt":     [0.5, 8.0],        # snow melt factor (mm/°C/day)
    "split_b":      [0.5, 1.0],        # split
    "split_k":      [1, 10],           # split
    "avai_efmax":   [0.05, 0.95],      # max ET efficiency (unitless)
    "avai_cap_base":   [0.0, 1000.0],   # available water capacity (mm)
    "avai_wetpoint99": [0.30, 0.90],    # threshold for full ET activation (unitless)
    "avai_beta":    [0.05, 0.95],      # LAI response curve
    "fast_kf":      [0.05, 0.95],      # fast flow coefficient
    "fast_perc":    [0.1, 20.0],       # percolation from fast to slow store (mm/day)
    "slow_ks":      [1e-4, 1e-1],      # slow flow recession rate
    "river_maxbas": [1.0, 5.0],        # MAXBAS routing kernel width (days)
}

attr_cols = [
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

batch_size = 2048
early_stop_patience = 10
loss_weights = [1.0, 1.0, 1.0, 1.0]
idx_daily = pd.date_range(start='1996-01-01', end='2020-12-31', freq='D')


# Parser
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default="model_output")
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--show_progress", action="store_true")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the pickle file")

    return parser.parse_args()

# Main
def main():

    args = parse_args()
    from libs.model import HybridModel
    from libs.utils import set_seed, train_model

    # Setup
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    data_path = os.path.abspath(args.data_path)
    with open(data_path, "rb") as f:
        data_obj = pickle.load(f)
    train_data = data_obj["train_data"]
    val_data   = data_obj["val_data"]
    test_data  = data_obj["test_data"]

    # DataLoaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_data, batch_size=batch_size, shuffle=False) # currently unused

    # Model & Training
    model = HybridModel(param_range=param_ranges, attr_cols=attr_cols).to(device)
    date_seq = torch.tensor([d.toordinal() for d in idx_daily], dtype=torch.long, device=device)

    model_path = save_dir / f"best_model_seed{args.seed}.pt"
    log_path = save_dir / f"training_progress_seed{args.seed}.txt"

    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        model_path=model_path,
        log_path=log_path,
        num_epochs=args.num_epochs,
        seed=args.seed,
        early_stop_patience=early_stop_patience,
        loss_weights=loss_weights,
        date_seq=date_seq,
        show_progress=args.show_progress
    )

    model.load_state_dict(torch.load(model_path, weights_only=True))


if __name__ == "__main__":
    main()