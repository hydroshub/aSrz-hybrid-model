import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import json
import hashlib
import pickle
import config

class TimeBlockCV:
    def __init__(self, years, num_folds=5, spinup_years=None, spinup_until=None):
        # Store raw years
        self.years = np.array(years).astype(int)
        self.num_folds = num_folds

        # Determine spinup years
        all_years = sorted(np.unique(self.years))
        if spinup_years is not None:
            spinup_years = list(sorted(set(spinup_years)))
        elif spinup_until is not None:
            spinup_years = [y for y in all_years if y < spinup_until]
        else:
            spinup_years = []

        self.spinup_years = list(map(int, spinup_years))

        # Remaining years used for CV
        self.unique_years = [y for y in all_years if y not in self.spinup_years]

        # Prepare folds
        self.fold_years = len(self.unique_years) // num_folds
        self.folds = self._generate_folds()

    # -----
    # Internal fold generation
    # -----
    def _generate_folds(self):
        return [
            self.unique_years[i:i + self.fold_years]
            for i in range(0, len(self.unique_years), self.fold_years)
        ]

    # -----
    # Get train/val/test years per fold
    # -----
    def get_masks(self, fold_idx, time_index=None, val_stride=4, val_offset=3):
        test_years = set(self.folds[fold_idx])
        non_test_years = [y for y in self.unique_years if y not in test_years]
        val_years = [y for i, y in enumerate(non_test_years) if (i % val_stride) == val_offset]
        train_years = [y for y in non_test_years if y not in val_years]

        result = {
            "years_train": list(map(int, train_years)),
            "years_val": list(map(int, val_years)),
            "years_test": sorted(map(int, test_years)),
        }

        if time_index is not None:
            result["spinup_mask"] = self.get_spinup_mask(time_index)

        return result

    # -----
    # Convert year list to mask
    # -----
    def get_mask(self, time_index, year_list):
        years = np.array([ts.year for ts in time_index])
        return np.isin(years, year_list)

    # -----
    # Get spinup mask
    # -----
    def get_spinup_mask(self, time_index):
        years = np.array([ts.year for ts in time_index])
        return np.isin(years, self.spinup_years)

    # -----
    # Convenience: get all masks (spinup/train/val/test)
    # -----
    def get_full_mask_dict(self, fold_idx, time_index, val_stride=4, val_offset=3):
        masks = self.get_masks(fold_idx, time_index, val_stride, val_offset)
        return {
            "spinup_mask": self.get_spinup_mask(time_index),
            "train_mask": self.get_mask(time_index, masks["years_train"]),
            "val_mask": self.get_mask(time_index, masks["years_val"]),
            "test_mask": self.get_mask(time_index, masks["years_test"]),
        }

    # -----
    # Print summary of current fold
    # -----
    def print_summary(self, fold_idx):
        m = self.get_masks(fold_idx)
        print("─" * 50)
        print(f"Cross-validation summary (Fold {fold_idx})")
        print(f"Total unique years (CV): {len(self.unique_years)} ({self.unique_years[0]}–{self.unique_years[-1]})")
        print(f"Fold rule: {self.fold_years}-year blocks, total folds: {self.num_folds}")
        print(f">> Spinup years: {self.spinup_years}")
        print(f">> Train years: {m['years_train']}")
        print(f">> Val years:   {m['years_val']}")
        print(f">> Test years:  {m['years_test']}")
        print("\u2500" * 50)


class UnifiedDataset(Dataset):
    def __init__(self, x, y, z, mask_daily, mask_monthly):
        self.x = x
        self.y = y.clone()
        self.z = z.clone()
        self.y[:, ~mask_daily, :] = float("nan")
        self.z[:, ~mask_monthly, :] = float("nan")

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, i):
        return self.x[i], self.y[i], self.z[i]

class AttributeProcessor:
    def __init__(self, attr_cols, area_cols=None, ele_cols=None):
        self.attr_cols = attr_cols

        self.cover_cols = [col for col in attr_cols if col.startswith("cover_")]
        self.non_cover_cols = [col for col in attr_cols if not col.startswith("cover_")]

        self.area_cols = [col for col in self.non_cover_cols if "area" in col.lower()] if area_cols is None else area_cols
        self.ele_cols = [col for col in self.non_cover_cols if "ele" in col.lower()] if ele_cols is None else ele_cols

        self.scaler = StandardScaler()

    def compute_landcover(self, df):
        df_proc = df.copy()

        try:
            df_proc['cover_forest'] = df_proc[[f'glc_pc_s{str(i).zfill(2)}' for i in [1,2,3,4,5,6,7,8,9,10]]].sum(axis=1)
            df_proc['cover_shrub']  = df_proc[[f'glc_pc_s{str(i).zfill(2)}' for i in [11,12,15]]].sum(axis=1)
            df_proc['cover_grass']  = df_proc[[f'glc_pc_s{str(i).zfill(2)}' for i in [13,14]]].sum(axis=1)
            df_proc['cover_crop']   = df_proc[[f'glc_pc_s{str(i).zfill(2)}' for i in [16,17,18]]].sum(axis=1)
            df_proc['cover_others'] = df_proc[[f'glc_pc_s{str(i).zfill(2)}' for i in [19,20,21,22]]].sum(axis=1)
            df_proc['cover_herb']   = df_proc[[f'glc_pc_s{str(i).zfill(2)}' for i in [11,12,13,14,15,16,17,18]]].sum(axis=1)
            df_proc['cover_nonforest'] = 100.0 - df_proc['cover_forest']
        except KeyError:
            print("Warning: Some glc_pc_sXX columns not found — skipping cover calculation.")

        return df_proc

    def transform_fixed(self, df):
        df_proc = df.copy()

        for col in self.area_cols:
            df_proc[col] = np.log1p(df_proc[col].clip(lower=0))

        for col in self.ele_cols:
            df_proc[col] = np.sqrt(df_proc[col].clip(lower=0, upper=5000))

        return df_proc

    def fit(self, df):
        df_proc = self.compute_landcover(df)
        df_proc = self.transform_fixed(df_proc)
        self.scaler.fit(df_proc[self.non_cover_cols])

    def transform(self, df):
        df_proc = self.compute_landcover(df)
        df_proc = self.transform_fixed(df_proc)
        df_proc[self.cover_cols] = df_proc[self.cover_cols].copy()
        df_proc[self.non_cover_cols] = self.scaler.transform(df_proc[self.non_cover_cols])
        return df_proc

    def fit_transform(self, df):
        df_proc = self.compute_landcover(df)
        df_proc = self.transform_fixed(df_proc)
        self.scaler.fit(df_proc[self.non_cover_cols])
        df_proc[self.cover_cols] = df_proc[self.cover_cols].copy()
        df_proc[self.non_cover_cols] = self.scaler.transform(df_proc[self.non_cover_cols])
        return df_proc

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump({
                'attr_cols': self.attr_cols,
                'area_cols': self.area_cols,
                'ele_cols': self.ele_cols,
                'scaler': self.scaler
            }, f)
    
    @classmethod
    def load(cls, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        proc = cls(data['attr_cols'], data['area_cols'], data['ele_cols'])
        proc.scaler = data['scaler']
        return proc


def load_attributes(attr_paths, basin_ids):

    if isinstance(attr_paths, str):
        attr_paths = [attr_paths]

    attr_dfs = []
    for path in attr_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Attribute file not found: {path}")
        df = pd.read_csv(path, index_col='gauge_id')
        attr_dfs.append(df)

    attr_df = pd.concat(attr_dfs, axis=1, join='inner')

    missing = [bid for bid in basin_ids if bid not in attr_df.index]
    if missing:
        raise ValueError(f"Missing basin(s) in attributes: {missing}")

    attr_df = attr_df.loc[basin_ids].copy()

    return attr_df


def load_dataset(
    data_dir,
    basin_list_path,
    attr_paths,
    input_cols,
    target_daily_cols,
    target_monthly_cols,
    attr_cols,
    start_year="2001",
    end_year="2020"
):
    basin_ids = pd.read_csv(basin_list_path, header=None)[0].to_list()
    attr_df_raw = load_attributes(attr_paths, basin_ids)
    processor = AttributeProcessor(attr_cols=attr_cols)
    attr_df = processor.fit_transform(attr_df_raw)

    x_all, y_all, z_all, valid_ids, skipped_ids = [], [], [], [], []
    idx_daily, idx_monthly = None, None

    for basin_id in tqdm(basin_ids, desc="Loading basins"):
        path = os.path.join(data_dir, f"{basin_id}_merged.csv")
        if not os.path.exists(path) or basin_id not in attr_df.index:
            skipped_ids.append(basin_id)
            continue

        try:
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            df = df.loc[start_year:end_year]

            if df.empty:
                skipped_ids.append(basin_id)
                continue

            if idx_daily is None:
                idx_daily = df.index
            elif not df.index.equals(idx_daily):
                print(f"Skipping {basin_id}: daily index mismatch")
                skipped_ids.append(basin_id)
                continue

            # Add static attributes to each time step
            for col in attr_cols:
                df[col] = attr_df.loc[basin_id, col]

            x = df[input_cols + attr_cols].values
            y = df[target_daily_cols].values
            y[y < 0] = np.nan
            y[:, 2][y[:, 2] <= 0.1] = np.nan  # SWE filter
            y[:, 0][y[:, 0] >= 100] = np.nan  # Q outlier removal

            df_monthly = df[target_monthly_cols].resample("ME").first()
            if idx_monthly is None:
                idx_monthly = df_monthly.index
            else:
                df_monthly = df_monthly.reindex(idx_monthly)

            z = df_monthly.values

            if x.shape[0] == y.shape[0]:
                x_all.append(torch.tensor(x, dtype=torch.float32))
                y_all.append(torch.tensor(y, dtype=torch.float32))
                z_all.append(torch.tensor(z, dtype=torch.float32))
                valid_ids.append(basin_id)
            else:
                print(f"Skipping {basin_id}: shape mismatch")
                skipped_ids.append(basin_id)

        except Exception as e:
            print(f"Error loading {basin_id}: {e}")
            skipped_ids.append(basin_id)
            continue

    if not x_all:
        raise ValueError("No valid basins found.")

    x_all = torch.stack(x_all)
    y_all = torch.stack(y_all)
    z_all = torch.stack(z_all)

    print("─" * 50)
    print("Data loading summary")
    print(f"> Loaded basins: {len(valid_ids)}")
    print(f"> Skipped basins: {len(skipped_ids)}")
    print(f"x_all: {x_all.shape}  [B, T, D]")
    print(f"y_all: {y_all.shape}  [B, T, V]")
    print(f"z_all: {z_all.shape}  [B, M, K]")
    print(f"Time (daily):   {idx_daily[0].date()} → {idx_daily[-1].date()}")
    print(f"Time (monthly): {idx_monthly[0].date()} → {idx_monthly[-1].date()}")
    print("─" * 50)

    return x_all, y_all, z_all, idx_daily, idx_monthly, valid_ids, processor


def get_config_hash(config_dict):
    config_str = json.dumps(config_dict, sort_keys=True)
    return hashlib.md5(config_str.encode("utf-8")).hexdigest()

def load_dataset_with_cache(
    data_dir,
    basin_list_path,
    attr_paths,
    input_cols,
    target_daily_cols,
    target_monthly_cols,
    attr_cols,
    start_year,
    end_year,
    cache_dir="cache"
):
    # Assemble config for hashing
    config = {
        "data_dir": data_dir,
        "basin_list_path": basin_list_path,
        "attr_paths": attr_paths,
        "input_cols": input_cols,
        "target_daily_cols": target_daily_cols,
        "target_monthly_cols": target_monthly_cols,
        "attr_cols": attr_cols,
        "start_year": start_year,
        "end_year": end_year
    }

    # Compute hash
    config_hash = get_config_hash(config)

    # Prepare cache paths
    os.makedirs(cache_dir, exist_ok=True)
    array_cache_path = os.path.join(cache_dir, f"cached_dataset_{config_hash}.npz")
    meta_cache_path = os.path.join(cache_dir, f"cached_meta_{config_hash}.pkl")
    processor_cache_path = os.path.join(cache_dir, f"cached_processor_{config_hash}.pkl")

    # Check cache
    if os.path.exists(array_cache_path) and os.path.exists(meta_cache_path) and os.path.exists(processor_cache_path):
        # Load arrays
        npzfile = np.load(array_cache_path)
        x_all = npzfile["x_all"]
        y_all = npzfile["y_all"]
        z_all = npzfile["z_all"]
        idx_daily = npzfile["idx_daily"]
        idx_monthly = npzfile["idx_monthly"]

        x_all = torch.tensor(x_all, dtype=torch.float32)
        y_all = torch.tensor(y_all, dtype=torch.float32)
        z_all = torch.tensor(z_all, dtype=torch.float32)
        idx_daily   = pd.to_datetime(idx_daily)
        idx_monthly = pd.to_datetime(idx_monthly)
        
        # Load metadata
        with open(meta_cache_path, "rb") as f:
            valid_ids = pickle.load(f)

        processor = AttributeProcessor.load(processor_cache_path)

        print(f"[Cache] Loaded dataset: {array_cache_path}")

    else:
        # Call original load_dataset
        x_all, y_all, z_all, idx_daily, idx_monthly, valid_ids, processor = load_dataset(
            data_dir=data_dir,
            basin_list_path=basin_list_path,
            attr_paths=attr_paths,
            input_cols=input_cols,
            target_daily_cols=target_daily_cols,
            target_monthly_cols=target_monthly_cols,
            attr_cols=attr_cols,
            start_year=start_year,
            end_year=end_year
        )

        # Save arrays
        np.savez_compressed(array_cache_path,
                            x_all=x_all,
                            y_all=y_all,
                            z_all=z_all,
                            idx_daily=idx_daily,
                            idx_monthly=idx_monthly)

        # Save metadata
        with open(meta_cache_path, "wb") as f:
            pickle.dump(valid_ids, f)

        processor.save(processor_cache_path)

        print(f"[Cache] Saved dataset: {array_cache_path}")

    return x_all, y_all, z_all, idx_daily, idx_monthly, valid_ids, processor
