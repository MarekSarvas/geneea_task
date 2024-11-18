import random
import argparse

from pathlib import Path
import pandas as pd

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
        df.to_json(new_data_dir/split, orient="records", lines=True)

if __name__ == "__main__":
    random.seed(42)
    args = parse_args()
    main(args)




