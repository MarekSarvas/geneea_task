"""Implementation of utility methods for data manipulation and metric computation.
"""
from typing import Dict, Tuple, List
from pathlib import Path

import numpy as np
from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer
)
from datasets import load_dataset, DatasetDict, Dataset, ClassLabel
import evaluate


def load_data(train: Path, val: Path) -> DatasetDict:
    """

    Args:
        train (Path): Path to json file with training data.
        val (Path): Path to json file with validation data.

    Returns:
        DatasetDict: HuggingFace format dataset.
    """
    data_files = {}
    for split, path in zip(["train", "val"], [train, val]):
        if path.exists():
            data_files[split] = str(path)
        else:
            print(f"Skipping {split} split, path {path} does not exists.")

    dataset = load_dataset("json", data_files=data_files)
    return dataset


def join_text_cols(row, text_cols: List[str], to_lower: bool):
    """Joins text of selected columns into one to use as input into
    the model. Optionally lowercases it.

    Args:
        row : One row of dataset.
        text_cols (List[str]): Columns to join.
        to_lower (bool):

    Returns:
        : updated dataset row
    """
    text = " ".join([row[col] for col in text_cols])
    if to_lower:
        text = text.lower()
    row["input_text"] = text
    return row


def get_labels(train_dataset: Dataset) -> List:
    """_

    Args:
        train_dataset (Dataset): Training split of the dataset.

    Returns:
        List: All labels in the dataset.
    """
    labels = sorted(list(set(train_dataset["category"])))
    return labels


def prepare_dataset(dataset: DatasetDict, tokenizer, text_cols: List[str], labels: List[str], max_length: int=256, padding: str="longest", to_lower: bool=True) -> DatasetDict:
    """Create text column (model input), tokenizes the text and maps
    string labels into numeric values.

    Args:
        dataset (DatasetDict): 
        tokenizer (): 
        text_cols (List[str]): Columns that will make model inputs. 
        labels (List[str]):
        max_length (int, optional): Max length of the tokenized inputs. Defaults to 256.
        padding (str, optional): Padding strategy for input tokenization. Defaults to "longest".
        to_lower (bool, optional): If lowercase the text before tokenization. Defaults to True.

    Returns:
        DatasetDict: Updated dataset.
    """
    # create text column that will be used as input to the model
    dataset = dataset.map(lambda data_slice:
        join_text_cols(data_slice, text_cols, to_lower)
    )
    # tokenize text
    dataset = dataset.map(lambda data_slice: tokenizer(
        data_slice["input_text"],
        truncation=True,
        max_length=max_length,
        padding=padding)
    )

    # assign ID to categorical features
    dataset = dataset.rename_column("category", "labels")
    cat_feats = ClassLabel(num_classes=len(labels), names=labels)
    dataset = dataset.cast_column("labels", cat_feats)

    return dataset



def load_models(model_name: str, num_labels: int) -> Tuple:
    """_summary_

    Args:
        model_name (str): Name of the pre-trained model.
        num_labels (int): Number of classes.

    Returns:
        Tuple[AutoModelForSequenceClassification, AutoTokenizer]: Model and tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=num_labels,
            )

    return model, tokenizer


def compute_metrics(output: Tuple):
    """Computes accuracy on predicted labels

    Args:
        output (Tuple): Output of the model.

    Returns:
        _type_: Accuracy metric.
    """
    metric = evaluate.load("accuracy")
    preds_logits, labels = output
    preds = np.argmax(preds_logits, axis=1)
    acc = metric.compute(predictions=preds, references=labels)
    return acc
