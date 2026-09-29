import os
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import importlib
import sys

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--basin_list", type=str, required=True)
    parser.add_argument("--config", type=str, required=True, help="Path to config.py")
    parser.add_argument("--fold_id", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default="model_output_test")
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--show_progress", action="store_true")

    return parser.parse_args()

def main():
    args = parse_args()
    config_path = os.path.abspath(args.config)

    # ----- Load dynamic config -----
    spec = importlib.util.spec_from_file_location("config", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)

    # note data,model,utils need this config, so don't put them in the head
    sys.modules["config"] = config
    from data import load_dataset_with_cache, TimeBlockCV, UnifiedDataset
    from model import HybridModel
    from utils import set_seed, train_model
    
    # ----- Setup -----
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ----- Load data -----
    x_all, y_all, z_all, idx_daily, idx_monthly, valid_ids, processor = load_dataset_with_cache(
        data_dir=config.DATA_DIR,
        basin_list_path=args.basin_list,
        attr_paths=config.ATTR_PATHS,
        input_cols=config.INPUT_COLS,
        target_daily_cols=config.TARGET_DAILY_COLS,
        target_monthly_cols=config.TARGET_MONTHLY_COLS,
        attr_cols=config.ATTR_COLS,
        start_year=config.START_YEAR,
        end_year=config.END_YEAR
    )

    # ------------------------------------------------------
    # Setup cross-validation
    # ------------------------------------------------------
    cv = TimeBlockCV(years=idx_daily.year, num_folds=5, spinup_until=2001)
    cv.print_summary(args.fold_id)
    # ------------------------------------------------------
    # Generate masks
    # ------------------------------------------------------
    mask_daily   = cv.get_full_mask_dict(args.fold_id, idx_daily)
    mask_monthly = cv.get_full_mask_dict(args.fold_id, idx_monthly)
    # ------------------------------------------------------
    # Prepare datasets
    # ------------------------------------------------------
    train_data = UnifiedDataset(x_all, y_all, z_all, mask_daily["train_mask"], mask_monthly["train_mask"])
    val_data   = UnifiedDataset(x_all, y_all, z_all, mask_daily["val_mask"], mask_monthly["val_mask"])
    test_data  = UnifiedDataset(x_all, y_all, z_all, mask_daily["test_mask"], mask_monthly["test_mask"])
    # ------------------------------------------------------
    # DataLoaders
    # ------------------------------------------------------
    train_loader = DataLoader(train_data, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_data, batch_size=config.BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_data, batch_size=config.BATCH_SIZE, shuffle=False)

    # ----- Model & training -----
    model = HybridModel(param_range=config.PARAM_RANGES, attr_cols=config.ATTR_COLS).to(device)
    date_seq = torch.tensor([d.toordinal() for d in idx_daily], dtype=torch.long, device=device)

    model_path = save_dir / f"best_model_seed{args.seed}_fold{args.fold_id}.pt"
    log_path = save_dir / f"training_progress_seed{args.seed}_fold{args.fold_id}.txt"
    
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        model_path=model_path,
        log_path=log_path,
        num_epochs=args.num_epochs,
        seed=args.seed,
        early_stop_patience=config.EARLY_STOP_PATIENCE,
        loss_weights=config.LOSS_WEIGHTS,
        date_seq=date_seq,
        show_progress=args.show_progress
    )

    model.load_state_dict(torch.load(model_path, weights_only=True))

if __name__ == "__main__":
    main()