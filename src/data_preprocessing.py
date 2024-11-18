import random
import argparse

from pathlib import Path
import pandas as pd
from imblearn.over_sampling import RandomOverSampler

def parse_args():
    """Parse datasets paths and training hyper parameters.

    Returns:
        _type_: arguments
    """
    parser = argparse.ArgumentParser(description="Parse data and hyper params for training.")

    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to the dir with data."
    )
    parser.add_argument(
        "--new_data_dir",
        type=str,
        required=True,
        help="Path to the dir where new dataset will be stored into jsonl files."
    )
    return parser.parse_args()


def join_labels(x, cat_dict):
    for k, v in cat_dict.items():
        for category in v:
            if category == x:
                return k
    return x


def normalize_url(url: str) -> str:
    new_url = url.split("/")[-1].split(".")[0].rsplit("_")[0]
    return new_url


def resample_data(df: pd.DataFrame) -> pd.DataFrame:
    features = df.drop(columns=["category"])
    y = df["category"]  # Labels (category column)
    # Initialize RandomOverSampler
    oversampler = RandomOverSampler(sampling_strategy='auto', random_state=42)

    # Apply oversampling
    features_resampled, y_resampled = oversampler.fit_resample(features, y)

    # Combine the resampled features and labels back into a DataFrame
    df_resampled = pd.DataFrame(features_resampled, columns=features.columns)
    df_resampled['category'] = y_resampled
    return df_resampled

def main(args):
    # 1. process paths
    data_dir = Path(args.data_dir)
    new_data_dir = Path(args.new_data_dir)
    new_data_dir.mkdir(parents=True, exist_ok=True)

    splits = ["train.jsonl", "test.jsonl", "dev.jsonl"]

    # 2. Load data
    join_minor_categories = {
            "ENVIRONMENT": ["GREEN", "ENVIRONMENT"],
            "ARTS": ["ARTS & CULTURE", "ARTS", "CULTURE & ARTS"],
            "EDUCATION": ["EDUCATION", "COLLEGE"],
            "FOOD": ["TASTE", "FOOD & DRINK"],
            "WORLDPOST": ["THE WORLDPOST", "WORLDPOST", "WORLD NEWS"],
            "STYLE": ["STYLE", "STYLE & BEAUTY"],
            "PARENTING": ["PARENTING", "PARENTS"],
            "WELLNESS": ["WELLNESS", "HEALTHY LIVING"],
    }

    for split in splits:
        df = pd.read_json(data_dir / f"{split}", lines=True)
        df["category"] = df["category"].map(lambda x: join_labels(x, join_minor_categories))
        df["link"] = df["link"].map(normalize_url)
        if "train" in split:
            df = resample_data(df.copy())
        df.to_json(new_data_dir/split, orient="records", lines=True)

    
        

if __name__ == "__main__":
    random.seed(42)
    args = parse_args()
    main(args)




