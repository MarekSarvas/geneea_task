"""
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import (
        TrainingArguments,
        Trainer,
        DataCollatorWithPadding,
)

from utils import (
    load_models,
    compute_metrics,
    prepare_dataset
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
    model, tokenizer = load_models(args.model_dir, args.model_dir, local_files_only=True)
    collator = DataCollatorWithPadding(tokenizer)

    label2id = model.config.label2id
    id2label= model.config.id2label

    # 3. preprocess and tokenize dataset
    dataset = prepare_dataset(
            dataset,
            tokenizer=tokenizer,
            text_cols=args.text_cols,
            label2id=label2id
    )

    # 3. Prepare params for training
    training_args = TrainingArguments(
        output_dir=args.model_dir,
        per_device_eval_batch_size=8,
    )

    # 4. Train the model
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=None,
        eval_dataset=None,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=lambda x: compute_metrics(x, id2label=id2label),
    )

    predictions, _, _= trainer.predict(dataset["test"])
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
