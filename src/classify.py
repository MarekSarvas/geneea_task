"""This model adds news categories into an input jsonfile as "category" key.
"""
import argparse
from pathlib import Path

import tqdm
import numpy as np
import pandas as pd
from datasets import load_dataset

from torch.utils.data import DataLoader

from utils import (
    prepare_dataset,
    prepare_onnx_batch,
    load_onnx_models,
    load_mappings
)

def parse_args():
    parser = argparse.ArgumentParser(description="Parse data and hyper params for training.")

    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Path to checkpoint of trained model."
    )

    parser.add_argument(
        "--data_to_label",
        type=str,
        default=None,
        required=False,
        help="Path to the validation data JSON file."
    )
    parser.add_argument(
        "--text_cols",
        nargs="+",
        default=["headline", "short_description"],
        help="Learning rate for the training."
    )
    return parser.parse_args()


def main(args):
    # 1. Load dataset
    data_path = Path(args.data_to_label)
    if data_path.exists() and data_path.suffix == ".jsonl":
        dataset = load_dataset("json", data_files={"test": str(data_path)})
    else:
        raise FileNotFoundError("Test data file does not exists.")

    # 2. Load models
    model_dir = Path(args.model_dir)
    onnx_model_path = model_dir/ "model.onnx"
    onnx_session, tokenizer = load_onnx_models(onnx_model_path, tokenizer_name=model_dir, local_files_only=True)

    # load label mappings
    label2id, id2label = load_mappings(model_dir/"label2id.json")
    # 3. preprocess and tokenize dataset
    dataset = prepare_dataset(
            dataset,
            tokenizer=tokenizer,
            text_cols=args.text_cols,
            label2id=label2id
    )

    # 4. Run prediction inference
    predictions = []
    dataloader = DataLoader(dataset["test"], batch_size=8, collate_fn=tokenizer.pad)
    for batch in tqdm.tqdm(dataloader, total=len(dataloader)):
        inputs = prepare_onnx_batch(batch)
        batch_preds = onnx_session.run(["logits"], inputs)
        predictions.append(batch_preds[0])

    #labels = dataset["test"]["labels"][:32]
    predictions = np.concatenate(predictions, axis=0)
    predictions = np.argmax(predictions, axis=1)
    predictions = [id2label[x] for x in predictions]

    new_json = data_path.with_name(data_path.stem + "_labeled.jsonl")
    orig_json = pd.read_json(data_path, lines=True)
    orig_json["category"] = predictions
    orig_json.to_json(new_json, orient='records', lines=True)



if __name__ == "__main__":
#- **inputs**: serialized model file, data to classify (the JSONL without `category` field)
#  - **outputs**: the input JSONL with `category` field set to model's predicted value
    args = parse_args()
    main(args)
