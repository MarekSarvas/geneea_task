"""Module for trained model evaluation. Results are saved into eval/<timestamp>/ directory.
"""
import argparse
import json
from pathlib import Path

import tqdm
import numpy as np
from sklearn.metrics import confusion_matrix
from datasets import load_dataset
from torch.utils.data import DataLoader

from utils import (
    compute_metrics,
    prepare_dataset,
    prepare_onnx_batch,
    load_onnx_models,
    load_mappings
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
    test_path = Path(args.test_data)
    exp_dir = Path(args.exp_dir)

    # 1. Load dataset
    if test_path.exists() and test_path.suffix == ".jsonl":
        dataset = load_dataset("json", data_files={"test": str(test_path)})
    else:
        raise FileNotFoundError("Test data file does not exists.")

    # 2. Load models
    onnx_model_path = exp_dir/ "model.onnx"
    onnx_session, tokenizer = load_onnx_models(onnx_model_path, tokenizer_name=exp_dir, local_files_only=True)

    # load label mappings
    label2id, id2label = load_mappings(exp_dir/"label2id.json")


    # 3. preprocess and tokenize dataset
    dataset = prepare_dataset(
            dataset,
            tokenizer=tokenizer,
            text_cols=args.text_cols,
            label2id=label2id
    )

    eval_dir = exp_dir / "eval"
    if not eval_dir.exists():
        eval_dir.mkdir(parents=True, exist_ok=True)

    # 4. Run prediction inference
    predictions = []
    train_dataloader = DataLoader(dataset["test"], batch_size=32, collate_fn=tokenizer.pad)
    for batch in tqdm.tqdm(train_dataloader, total=len(train_dataloader)):
        inputs = prepare_onnx_batch(batch)
        batch_preds = onnx_session.run(["logits"], inputs)
        predictions.append(batch_preds[0])
        break

    labels = dataset["test"]["labels"][:32]
    predictions = np.concatenate(predictions, axis=0)

    # 5. Evaluate model performance
    metrics = compute_metrics((predictions, labels), args.verbose, id2label=id2label)

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
