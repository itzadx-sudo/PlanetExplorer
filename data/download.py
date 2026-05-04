import argparse
import os
from pathlib import Path

import pandas as pd
import requests
from sklearn.model_selection import train_test_split

URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+*+from+cumulative&format=csv"

KEEP_COLS = [
    "koi_dicco_msky", "koi_dikco_msky", "koi_prad", "koi_smet_err2", "koi_max_mult_ev", "koi_model_snr",
    "koi_steff_err1", "koi_smet_err1", "koi_prad_err2", "koi_steff_err2", "koi_ror", "koi_prad_err1",
    "koi_duration_err1", "koi_duration_err2", "koi_fittype_LS+MCMC", "koi_count", "koi_fwm_sdec_err",
    "koi_fwm_srao_err", "koi_fwm_sdeco_err", "koi_srad_err1", "koi_ror_err2", "koi_dor", "koi_smass_err1",
    "koi_fwm_stat_sig", "koi_ror_err1", "koi_fwm_sra_err", "koi_time0bk_err1", "koi_time0bk_err2",
    "koi_depth", "koi_time0_err1"
]


def download_data(force: bool = False, base_dir: Path = Path(".")):
    raw_file = base_dir / "data" / "raw" / "kepler_koi_raw.csv"
    raw_file.parent.mkdir(parents=True, exist_ok=True)
    if raw_file.exists() and not force:
        print(f"Raw file already exists, skipping download.")
        return
    print(f"Downloading from NASA Exoplanet Archive...")
    response = requests.get(URL)
    response.raise_for_status()
    raw_file.write_bytes(response.content)
    print(f"Saved to {raw_file}")


def preprocess_data(base_dir: Path = Path(".")):
    raw_file = base_dir / "data" / "raw" / "kepler_koi_raw.csv"
    train_file = base_dir / "data" / "processed" / "train.csv"
    test_file = base_dir / "data" / "processed" / "test.csv"
    train_file.parent.mkdir(parents=True, exist_ok=True)

    print("Preprocessing...")
    df = pd.read_csv(raw_file)

    df = df[df["koi_disposition"].isin({"CONFIRMED", "CANDIDATE", "FALSE POSITIVE"})]

    cols_to_keep = ["kepid", "koi_disposition"] + KEEP_COLS
    df = df[[c for c in cols_to_keep if c in df.columns]]

    cols_to_drop = df.columns[df.isnull().mean() > 0.0].tolist()
    if cols_to_drop:
        print(f"Dropping columns with missing values: {cols_to_drop}")
        df = df.drop(columns=cols_to_drop)

    train_df, test_df = train_test_split(
        df, test_size=0.10, stratify=df["koi_disposition"], random_state=42
    )

    train_df.to_csv(train_file, index=False)
    test_df.to_csv(test_file, index=False)

    print(f"Total rows: {len(df)}")
    print(f"Class distribution:\n{df['koi_disposition'].value_counts().to_string()}")
    print(f"Train: {train_df.shape}, Test: {test_df.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and preprocess Kepler dataset")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).parent.parent
    download_data(args.force, base_dir=root)
    preprocess_data(base_dir=root)
