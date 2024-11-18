"""Module for training a transformer encoder model for 
sequence classification.
"""
import argparse
from pathlib import Path
import random

from transformers import (
        TrainingArguments,
        Trainer,
        DataCollatorWithPadding,
        onnx
)
from transformers.onnx import FeaturesManager
from datasets import load_dataset, disable_caching
from optimum.onnxruntime import ORTModelForSequenceClassification

from utils import (
        create_label_mapping,
        load_models,
        get_labels,
        compute_metrics,
        prepare_dataset,
)


def parse_args():
    """Parse datasets paths and training hyper parameters.

    Returns:
        _type_: arguments
    """
    parser = argparse.ArgumentParser(description="Parse data and hyper params for training.")

    parser.add_argument(
        "--train_data",
        type=str,
        required=True,
        help="Path to the training data JSON file."
    )
    parser.add_argument(
        "--val_data",
        type=str,
        default="",
        help="Path to the validation data JSON file."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="FacebookAI/xlm-roberta-base",
        help="Name of the pre-trained HF model and tokenizer."
    )
    parser.add_argument(
        "--exp_dir",
        type=str,
        default="exp/default_exp",
        help="Path to the dir where the model and logs will be stored."
    )
    # Data params
    parser.add_argument(
        "--text_cols",
        nargs="+",
        default=["headline", "short_description"],
        help="Learning rate for the training."
    )
    # Hyperparams
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate for the training."
    )
    return parser.parse_args()


def main(args):
    # for development purpose
    disable_caching()

    # 1. Load dataset
    train, val = Path(args.train_data), Path(args.val_data)
    data_files = {}
    for split, path in zip(["train", "val"], [train, val]):
        if path.exists() and path.suffix == ".jsonl":
            data_files[split] = str(path)
        else:
            print(f"Skipping {split} split, path {path} does not exists.")

    dataset = load_dataset("json", data_files=data_files)

    # Create experiment direcotry and store label mappings
    exp_dir = Path(args.exp_dir)
    if not exp_dir.exists():
        exp_dir.mkdir(parents=True, exist_ok=True)

    # store label2id mappings
    labels = get_labels(dataset["train"])
    label2id_path = exp_dir / "label2id.json"
    label2id = create_label_mapping(labels, label2id_path)
    id2label = {id: label for label, id in label2id.items()}


    # 2. Load models
    model, tokenizer = load_models(args.model, args.model, label2id=label2id, id2label=id2label)
    collator = DataCollatorWithPadding(tokenizer)

    # 3. preprocess and tokenize dataset
    dataset = prepare_dataset(
            dataset,
            tokenizer=tokenizer,
            text_cols=args.text_cols,
            label2id=label2id
    )

    # TODO: set correct hyperparams
    # 3. Prepare params for training
    training_args = TrainingArguments(
        output_dir=exp_dir,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=8,
        num_train_epochs=1,
        max_steps=5,
        save_strategy="steps",
        save_steps=5,
        eval_steps=5,
        load_best_model_at_end=False,
        learning_rate=1e-4,
        weight_decay=0.01,
    )

    # 4. Train the model
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["val"],
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=lambda x: compute_metrics(x, verbose=False, id2label=id2label),
    )
    trainer.train()
    trainer.save_model(exp_dir)



    # set parameters and paths
    feature = "sequence-classification"
    onnx_dir = exp_dir / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)
    onnx_model_path = onnx_dir / "model.onnx"

    _, model_onnx_config = FeaturesManager.check_supported_model_or_raise(model, feature=feature)
    onnx_config = model_onnx_config(model.config)
    model = model.cpu()
    # export
    _, _ = onnx.export(
            preprocessor=tokenizer,
            model=model,
            config=onnx_config,
            opset=14,
            output=onnx_model_path
    )
    tokenizer.save_pretrained(onnx_model_path.parent)



if __name__ == "__main__":
  #- **inputs**: train data, (optional) validation data, (optional) hyperparameter values
  #- **outputs**: serialized model file
    random.seed(42)
    args = parse_args()
    main(args)
