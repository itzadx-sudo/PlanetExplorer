import os
import argparse
import pandas as pd
import requests
from sklearn.model_selection import train_test_split

URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+*+from+cumulative&format=csv"
RAW_FILE = "data/raw/kepler_koi_raw.csv"
TRAIN_FILE = "data/processed/train.csv"
TEST_FILE = "data/processed/test.csv"

KEEP_COLS = [
    "koi_dicco_msky", "koi_dikco_msky", "koi_prad", "koi_smet_err2", "koi_max_mult_ev", "koi_model_snr",
    "koi_steff_err1", "koi_smet_err1", "koi_prad_err2", "koi_steff_err2", "koi_ror", "koi_prad_err1",
    "koi_duration_err1", "koi_duration_err2", "koi_fittype_LS+MCMC", "koi_count", "koi_fwm_sdec_err",
    "koi_fwm_srao_err", "koi_fwm_sdeco_err", "koi_srad_err1", "koi_ror_err2", "koi_dor", "koi_smass_err1",
    "koi_fwm_stat_sig", "koi_ror_err1", "koi_fwm_sra_err", "koi_time0bk_err1", "koi_time0bk_err2",
    "koi_depth", "koi_time0_err1"
]

def download_data(force=False):
    os.makedirs(os.path.dirname(RAW_FILE), exist_ok=True)
    if os.path.exists(RAW_FILE) and not force:
        print(f"Raw file {RAW_FILE} already exists. Skipping download.")
        return
    print(f"Downloading data from {URL}...")
    response = requests.get(URL)
    response.raise_for_status()
    with open(RAW_FILE, 'wb') as f:
        f.write(response.content)
    print(f"Saved raw data to {RAW_FILE}")

def preprocess_data():
    print("Loading raw data...")
    df = pd.read_csv(RAW_FILE)
    
    print("Preprocessing data...")
    # Drop rows with dispositions we don't care about
    valid_dispositions = {"CONFIRMED", "CANDIDATE", "FALSE POSITIVE"}
    df = df[df["koi_disposition"].isin(valid_dispositions)]
    
    # Retain agreed columns + kepid and koi_disposition
    cols_to_keep = ["kepid", "koi_disposition"] + KEEP_COLS
    
    # Ensure all columns exist, if not, they'll just be handled normally (pandas will raise KeyError, but we only select present ones)
    present_cols = [c for c in cols_to_keep if c in df.columns]
    df = df[present_cols]
    
    # Count missing values before cleaning
    missing_before = df.isnull().sum().sum()
    
    # Drop any column with >0% missing values
    # Actually, we first need to identify such columns in the feature set (excluding kepid and koi_disposition)
    # The prompt says "across the retained set".
    missing_pct = df.isnull().mean()
    cols_to_drop = missing_pct[missing_pct > 0.0].index.tolist()
    if cols_to_drop:
        print(f"Dropping columns with >0% missing values: {cols_to_drop}")
        df = df.drop(columns=cols_to_drop)
    
    # Impute remaining NaNs with column median
    # If we dropped all columns with >0% missing values, there shouldn't be any, but we still execute this step.
    num_cols = df.select_dtypes(include=["number"]).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    
    missing_after = df.isnull().sum().sum()
    
    print("Splitting data...")
    os.makedirs(os.path.dirname(TRAIN_FILE), exist_ok=True)
    
    # Stratified 90/10 train-test split
    train_df, test_df = train_test_split(
        df, test_size=0.10, stratify=df["koi_disposition"], random_state=42
    )
    
    train_df.to_csv(TRAIN_FILE, index=False)
    test_df.to_csv(TEST_FILE, index=False)
    
    print("\n--- Summary ---")
    print(f"Total rows: {len(df)}")
    print(f"Class distribution:\n{df['koi_disposition'].value_counts().to_string()}")
    print(f"Missing values before cleaning: {missing_before}")
    print(f"Missing values after cleaning: {missing_after}")
    print(f"Train set shape: {train_df.shape}")
    print(f"Test set shape: {test_df.shape}")
    print("Preprocessing complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and preprocess Kepler dataset")
    parser.add_argument("--force", action="store_true", help="Force download even if raw file exists")
    args = parser.parse_args()
    
    download_data(args.force)
    preprocess_data()
