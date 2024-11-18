"""Module for trained model evaluation. Results are saved into eval/<timestamp>/ directory.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix
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
        "--test_data",
        type=str,
        required=True,
        help="Path to the test data JSON file."
    )
    parser.add_argument(
        "--exp_dir",
        type=str,
        default="exp/default_exp",
        help="Path to the dir where the model and logs will be stored."
    )
    #parser.add_argument(
    #    "--label2id",
    #    type=str,
    #    default="data/label2id.json",
    #    help="Path to the test data JSON file."
    #)
    parser.add_argument(
        "--text_cols",
        nargs="+",
        default=["headline", "short_description"],
        help="Learning rate for the training."
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="If set, prints also confusion matrix, precision and recall for each category"
    )

    return parser.parse_args()


def main(args):
    # 1. Load dataset
    test_path = Path(args.test_data)
    if test_path.exists() and test_path.suffix == ".jsonl":
        dataset = load_dataset("json", data_files={"test": str(test_path)})
    else:
        raise FileNotFoundError("Test data file does not exists.")

    # Load mappings to convert the labels in dataset
    #label2id_path = Path(args.label2id)
    #if label2id_path.exists():
    #    with label2id_path.open(mode='r', encoding='utf-8') as f:
    #        label2id = json.load(f)
    #else:
    #    raise FileNotFoundError(f"Label mappings not found at {label2id_path}")

    # 2. Load models
    model, tokenizer = load_models(args.exp_dir, args.exp_dir, local_files_only=True)
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

    eval_dir = Path(args.exp_dir) / "eval"
    if not eval_dir.exists():
        eval_dir.mkdir(parents=True, exist_ok=True)

    # 3. Prepare params for training
    training_args = TrainingArguments(
        output_dir=eval_dir,
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
        compute_metrics=lambda x: compute_metrics(x, args.verbose, id2label),
    )

    predictions, labels, metrics = trainer.predict(dataset["test"])
    predictions = np.argmax(predictions, axis=1)

    predictions = [id2label[x] for x in predictions]
    labels = [id2label[x] for x in labels]
    cm = confusion_matrix(y_true=labels, y_pred=predictions)
    np.save(eval_dir / "confusion_matrix.npy", cm)
    
    with (eval_dir/"metrics.json").open(mode='w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=4)

    print(json.dumps(metrics, indent=2))



if __name__ == "__main__":
#- **inputs**: serialized model file, evaluation data (in the JSONL format defined above)
#- **outputs**: accuracy; optionally confusion matrix and precision/recall for each category
    args = parse_args()
    main(args)
