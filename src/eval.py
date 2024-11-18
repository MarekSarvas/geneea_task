"""Module for trained model evaluation. Results are saved into eval/<timestamp>/ directory.
"""
import argparse
import json
from pathlib import Path
from typing import List

import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import seaborn as sns

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
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
        "--lowercase",
        action="store_true",
        help="If set, put the model input into lowercase."
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="If set, prints also confusion matrix, precision and recall for each category"
    )
    return parser.parse_args()


def plot_cm(cm: np.ndarray, labels: List[str], save_to: Path):
    #plt.figure(figsize=(8, 6))
    #sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
    #            xticklabels=labels, yticklabels=labels)
    #plt.title('Confusion Matrix')
    #plt.xlabel('Predicted Labels')
    #plt.ylabel('True Labels')
    #plt.show()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap='Blues')
    plt.savefig(save_to, dpi=300, bbox_inches='tight')


def plot_conf_mat(conf_mat_df, path: Path, size=12):
    fig, ax = plt.subplots(figsize=(size, size))
    ax.tick_params(axis='both', which='major', labelsize=10, labelbottom = False, bottom=False, top = False, labeltop=True)
    ax.tick_params(axis=u'both', which=u'both',length=0)
    ax.xaxis.set_label_position('top')
    sns.heatmap(conf_mat_df, annot=True, norm=LogNorm(), ax=ax, fmt='g', square=True, cbar=False, linewidth=0.1, linecolor="black")
    
    fig.tight_layout()
    plt.savefig(path, dpi=300)


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
            label2id=label2id,
            to_lower=args.lowercase,
    )

    eval_dir = exp_dir / "eval"
    if not eval_dir.exists():
        eval_dir.mkdir(parents=True, exist_ok=True)

    # 4. Run prediction inference
    predictions = []
    train_dataloader = DataLoader(dataset["test"], batch_size=16, collate_fn=tokenizer.pad)
    for batch in tqdm.tqdm(train_dataloader, total=len(train_dataloader)):
        inputs = prepare_onnx_batch(batch)
        batch_preds = onnx_session.run(["logits"], inputs)
        predictions.append(batch_preds[0])

    labels = dataset["test"]["labels"]
    predictions = np.concatenate(predictions, axis=0)

    # 5. Evaluate model performance
    metrics = compute_metrics((predictions, labels), args.verbose, id2label=id2label)

    predictions = np.argmax(predictions, axis=1)

    predictions = [id2label[x] for x in predictions]
    labels = [id2label[x] for x in labels]
    df = pd.DataFrame({
            "predictions": predictions,
            "labels": labels
    })

    cm = confusion_matrix(y_true=labels, y_pred=predictions)
    df_confusion = pd.crosstab(df["labels"], df["predictions"])

    cm_path = eval_dir / "confusion_matrix.png"
    #plot_cm(cm, label2id.keys(), cm_path)
    plot_conf_mat(df_confusion, cm_path)

    np.save(eval_dir / "confusion_matrix.npy", cm)

    with (eval_dir/"metrics.json").open(mode='w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=4)

    print(json.dumps(metrics, indent=2))



if __name__ == "__main__":
#- **inputs**: serialized model file, evaluation data (in the JSONL format defined above)
#- **outputs**: accuracy; optionally confusion matrix and precision/recall for each category
    args = parse_args()
    main(args)
