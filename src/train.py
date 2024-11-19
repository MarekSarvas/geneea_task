"""Module for training a transformer encoder model for 
sequence classification.
"""
import argparse
from pathlib import Path
import random
from typing import Dict

from transformers import (
        TrainingArguments,
        Trainer,
        DataCollatorWithPadding,
        onnx,
        AutoModelForSequenceClassification,
        AutoTokenizer,
)
from transformers.onnx import FeaturesManager
from datasets import load_dataset, disable_caching

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
    parser.add_argument(
        "--lowercase",
        action="store_true",
        default=False,
        help="If set, put the model input into lowercase."
    )
    # Hyperparams
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate for the training."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help=""
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=2,
        help=""
    )
    return parser.parse_args()


def optuna_hp_space(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 6e-5, 1e-4, log=True),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [8, 16]),
        "num_train_epochs": trial.suggest_int("num_train_epochs", 3, 7, log=True),
        "gradient_accumulation_steps": trial.suggest_int("gradient_accumulation_steps", 1, 4, log=True),
    }


def model_init(trial, model_name_or_path: str, id2label: Dict, label2id: Dict):
    return AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path,
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id,
    )

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
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    collator = DataCollatorWithPadding(tokenizer)

    # 3. preprocess and tokenize dataset
    dataset = prepare_dataset(
            dataset,
            tokenizer=tokenizer,
            text_cols=args.text_cols,
            label2id=label2id,
            to_lower=args.lowercase,
    )

    # 4.Initialize training arguments and trainer 
    training_args = TrainingArguments(
        output_dir=exp_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=2,
        save_strategy="epoch",
        eval_strategy="epoch",
        logging_steps=100,
        save_total_limit=5,
        load_best_model_at_end=True,
        learning_rate=args.lr,
        weight_decay=0.01,
    )
    trainer = Trainer(
        model=None,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["val"],
        tokenizer=tokenizer,
        data_collator=collator,
        model_init=lambda trial: model_init(trial, args.model, label2id=label2id, id2label=id2label),
        compute_metrics=lambda x: compute_metrics(x, verbose=False, id2label=id2label),
    )

    # 5. Perform hyperparam search
    best_run = trainer.hyperparameter_search(
        direction="maximize",
        hp_space=optuna_hp_space,
        backend="optuna",
        n_trials=10
    )

    for n, v in best_run.hyperparameters.items():
        setattr(training_args, n, v)

    # 6. Retrain the model
    model = model_init(None, args.model, label2id=label2id, id2label=id2label)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['val'],
        tokenizer=tokenizer,
        compute_metrics=lambda x: compute_metrics(x, verbose=False, id2label=id2label),
    )

    trainer.train()
    trainer.save_model(exp_dir)
    tokenizer.save_pretrained(exp_dir)

    # 7. Export to onnx format
    feature = "sequence-classification"
    onnx_model_path =  exp_dir / "model.onnx"

    print(f"Exporting the model into onnx to: {onnx_model_path}")
    _, model_onnx_config = FeaturesManager.check_supported_model_or_raise(model, feature=feature)
    onnx_config = model_onnx_config(model.config)
    model = model.cpu()
    _, _ = onnx.export(
            preprocessor=tokenizer,
            model=model,
            config=onnx_config,
            opset=14,
            output=onnx_model_path
    )
    print("TRAINING DONE")




if __name__ == "__main__":
  #- **inputs**: train data, (optional) validation data, (optional) hyperparameter values
  #- **outputs**: serialized model file
    random.seed(42)
    args = parse_args()
    main(args)
