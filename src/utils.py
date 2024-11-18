"""Implementation of utility methods for data manipulation and metric computation.
"""
from pathlib import Path
import json
from typing import Dict, Tuple, List, Optional

import numpy as np
from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer
)
from datasets import DatasetDict, Dataset, ClassLabel
from sklearn.metrics import (
        precision_score,
        recall_score,
        accuracy_score
)


def create_label_mapping(labels: List[str], out_path: Path) -> Dict:
    """Save mappings between text labels and corresponding numeric IDs as json.

    Args:
        labels (List[str]): List of labels in train split.
        out_path (Path): Output json path.

    Returns:
        Dict: Label-to-ID mappings
    """
    labels.sort()
    label2id = {label: id for id, label in enumerate(labels)}
    out_path.touch(exist_ok=True)
    print(f"Saving label2id mapping into {out_path}")
    with out_path.open(mode='w', encoding='utf-8') as f:
        json.dump(label2id, f, indent=4)

    return label2id


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


def prepare_dataset(dataset: DatasetDict, tokenizer, text_cols: List[str], label2id: Dict, max_length: int=256, padding: str="longest", to_lower: bool=True) -> DatasetDict:
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
    cat_feats = ClassLabel(num_classes=len(label2id), names=list(label2id.keys()))
    dataset = dataset.cast_column("labels", cat_feats)

    return dataset



def load_models(model_name: str, tokenizer_name: str, num_labels: int, local_files_only=False) -> Tuple:
    """

    Args:
        model_name (str): Name of the pre-trained model.
        num_labels (int): Number of classes.

    Returns:
        Tuple[AutoModelForSequenceClassification, AutoTokenizer]: Model and tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=num_labels,
                local_files_only=local_files_only
            )

    return model, tokenizer


def prettyfy_metric(category_scores: np.ndarray, id2label: Optional[Dict] = None) -> Dict:
    """Store computed precision or recall as dictionary for each label.

    Args:
        category_scores (np.ndarray): Metric value for each category.
        id2label (Optional[Dict], optional): Mapping. Defaults to None.

    Returns:
        Dict: Metric for each category.
    """
    # if id2label dict is not provided, create arbitrary label names
    if id2label is None:
        id2label = {id: f"Label_{id}" for id in range(len(category_scores))}

    pretty_category_score = {}
    for i, score in enumerate(category_scores):
        pretty_category_score[id2label[i]] = score

    return pretty_category_score


def compute_metrics(output: Tuple, verbose: bool = False, id2label: Optional[Dict] = None) -> Dict:
    """Computes accuracy and optionally precision and recall on predicted labels.

    Args:
        output (Tuple): Labels and predictions of the model.
        verbose (bool, optional): If True, compute also precision and recall. Defaults to False.
        id2label (Optional[Dict], optional): Label mappings. Defaults to None.

    Returns:
        Dict: Dictionary containing metrics.
    """
    preds_logits, labels = output
    preds = np.argmax(preds_logits, axis=1)
    acc = accuracy_score(labels, preds)
    if verbose:
        p = precision_score(labels, preds, average=None, zero_division=0)
        p = prettyfy_metric(p, id2label)
        r = recall_score(labels, preds, average=None, zero_division=0)
        r = prettyfy_metric(r, id2label)
        return {"accuracy": acc, "precision": p, "recall": r} 
    return {"accuracy": acc}
