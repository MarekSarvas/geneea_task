"""_summary_


"""
import argparse
from pathlib import Path

from transformers import (
        TrainingArguments,
        Trainer,
        DataCollatorWithPadding,
)
from datasets import disable_caching

from utils import (
        load_data,
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
    dataset = load_data(train, val)
    labels = get_labels(dataset["train"])

    # 2. Load models
    model, tokenizer = load_models(args.model, len(labels))
    collator = DataCollatorWithPadding(tokenizer)

    # 3. preprocess and tokenize dataset
    dataset = prepare_dataset(
            dataset,
            tokenizer=tokenizer,
            text_cols=args.text_cols,
            labels=labels
    )

    # 3. Prepare params for training
    training_args = TrainingArguments(
        output_dir=args.exp_dir,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=8,
        num_train_epochs=20,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
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
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.save_model(args.exp_dir)








if __name__ == "__main__":
  #- **inputs**: train data, (optional) validation data, (optional) hyperparameter values
  #- **outputs**: serialized model file
    args = parse_args()
    main(args)
