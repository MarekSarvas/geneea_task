# News classification
- The main approach to classify news based on the news title and description was to take a pre-trained transformer model and fine-tune it for the classification task.
- I used the HuggingFace transformers library and specificaly roberta-base model. This model was chosen based on the data it was trained on, should have good representation of the text and HW reasons so it can be comfortably trained on one 8GB GPU.
- After initiall experiments, the model achieved only around ~60% accuracy mainly due to the unbalanced dataset and overlapping classes.
- I added some data pre-processing where I:
    - joined similar categories together such as: PARENT + PARENTING etc.
    - added text information extracted from url address as sometimes it is different to the title
    - removed some categories based on their content
    - oversample the categories with few examples
- For the input data, roberta-based was trained on cased text so I kept the text as is, I tried to remove punctuation and stopwords using Spacy and ntlk but it did not have any significant improvement on the performance.
- The data pre-processing boosted the performance to ~70% on dev set.
- Looking at a precision, recall scores and confusion matrix which categories are miss-classified, content of description or title for some categories is not exact and can fit into multiple categories.

# Installation
- Create conda environment from **environment.yml** file by running
```bash
conda env create -f environment.yml
```

# How to use the scripts
Every script beside the optional data preparation expects that the serialized trained model will be saved in the **path/to/experiments/<EXP_NAME>** directory as model.onnx, tokenizer is saved in the same directory with HuggingFace convention.
1. Optionally pre-process the given data unziped in **data/** dir by running:
```bash
python3 src/data_preprocessing.py \
    --data_dir data \
    --new_data_dir data_clean \
    --drop_categories "WEIRD NEWS" "GOOD NEWS" "FIFTY"
```
2. To train the model, activate the environment and run **src/train.py** with correct path to the directories, e.g.:
```bash
python3 src/train.py \
    --train_data data_clean/train.jsonl \
    --val_data path/to/data/dir/dev.jsonl \
    --model <name-of-pretrained-model_roberta-base> \
    --exp_dir exp/<name-of-the-experiemnt> \
    --text_cols headline short_description link \
    --lr 8e-5 \
    --batch_size 16 \
    --gradient_accumulation_steps 4
```
3. Evaluate the trained model on test set:
```bash
python3 src/eval.py \
    --test_data data_clean/test.jsonl \
    --exp_dir exp/<name-of-the-experiemnt>  \
    --text_cols headline short_description link \
    --verbose

```
4. Add predicted labels into the jsonl file. The labeled json will be stored as given file with "_labeled" name suffix, e.g. test.jsonl -> test_labeled.jsonl in the same directory.
```bash
python3 src/classify.py \
    --model_dir exp/<name-of-the-experiemnt>  \
    --data_to_label data_clean/test.jsonl
```