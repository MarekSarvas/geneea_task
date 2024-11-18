"""Module for text and category pre-processing.
"""
import random
import argparse
from pathlib import Path
from collections import Counter
from typing import Dict

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
    parser.add_argument(
        "--drop_categories",
        nargs="+",
        default=[],
        help="Which categories to drop before preprocessing."
    )

    return parser.parse_args()


def join_labels(orig_label: str, cat_dict: Dict) -> str:
    """_summary_

    Args:
        orig_label (str): Original label from dataset.
        cat_dict (Dict): Dictionary of label categories that should be in same category.

    Returns:
        str: New or original label.
    """
    for new_label, labels_to_map in cat_dict.items():
        if orig_label in labels_to_map:
            return new_label
    return orig_label


def normalize_url(url: str) -> str:
    """Extracts text information from url.

    Args:
        url (str):

    Returns:
        str: Text information.
    """
    new_url = url.split("/")[-1].split(".")[0].rsplit("_")[0]
    url_text = new_url.replace("-", " ").replace("_", " ")
    return url_text


def resample_data(df: pd.DataFrame) -> pd.DataFrame:
    """Categories with small number of examples are copied so the
    dataset is more balanced.

    Args:
        df (pd.DataFrame): Original dataset.

    Returns:
        pd.DataFrame: Resampled bigger dataset.
    """
    features = df.drop(columns=["category"])
    y = df["category"]
    # Initialize RandomOverSampler
    category_counts = Counter(y)
    sampling_strategy = {key: max(value, 1000) for key, value in category_counts.items()}
    oversampler = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=42)

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
            "BUSINESS": ["BUSINESS", "MONEY"],
    }

    for split in splits:
        df = pd.read_json(data_dir / f"{split}", lines=True)
        df = df[~df["category"].isin(args.drop_categories)]

        df["category"] = df["category"].map(lambda x: join_labels(x, join_minor_categories))
        df["link"] = df["link"].map(normalize_url)
        if "train" in split:
            df = resample_data(df.copy())

        df.to_json(new_data_dir/split, orient="records", lines=True)


if __name__ == "__main__":
    random.seed(42)
    args = parse_args()
    main(args)

