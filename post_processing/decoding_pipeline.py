# Utils
import os
import argparse
import pandas as pd
import numpy as np

# transformers essentials
from transformers import BertTokenizerFast, BertConfig, BertForTokenClassification
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

# Self-defined utils
import sys
sys.path.append(os.path.abspath("../utils"))
from prepare_dataset import prepare_data

# Global variables
MAX_LEN = 128
DATA_PATH = "../processed_notes.csv"

# Arguments
parser = argparse.ArgumentParser(
    description="Run NER inference using a Transformer-based model"
)

# Define inputs
parser.add_argument(
    "--model_name",
    type=str,
    required=True,
    help="Options: bert, clinicalbert, deberta"
)

parser.add_argument(
    "--finetune",
    action="store_true",
    help="Use the fine-tuned model (if set), otherwise use the baseline"
)

parser.add_argument(
    "--input_file",
    type=str,
    required=True,
    help="Path to input .txt file containing clinical text"
)

parser.add_argument(
    "--output_file",
    type=str,
    required=True,
    help="Path to output .txt file where predictions will be saved"
)

# Parse arguments
args = parser.parse_args()

MODEL = args.model_name
TEXT_FILE_PATH = args.input_file
FINE_TUNE = args.finetune

# Step 1: Load Fine-tuned model and Tokenier
model_path = # TODO: add model path according to model input
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForTokenClassification.from_pretrained(
    model_path,
    low_cpu_mem_usage=True,
    device_map="auto")

# Step 2: Load the labels and the input texts
data_dict = prepare_data("../processed_notes.csv")
data = data_dict["data"]
labels_to_ids = data_dict["labels_to_ids"]
ids_to_labels = data_dict["ids_to_labels"]

# load input text
text = # TODO: load input texts

# Inject labels into model config
model.config.id2label = {i: ids_to_labels[i] for i in ids_to_labels}
model.config.label2id = {v: k for k, v in model.config.id2label.items()}

# Step 3: Decoding Pipeline
ner_pipe = pipeline(
    "ner",
    model=model,
    tokenizer=tokenizer,
    aggregation_strategy="simple",  # combines B/I tokens into full spans
)

predictions = ner_pipe(text)

# Print predictions
print(f"{'Entity':<30} {'Tag':<10} {'Score':<6}")
print("-" * 50)

for ent in predictions:
    if ent["entity_group"] != "O":  # Optional: filter out "O"
        print(f"{ent['word']:<30} {ent['entity_group']:<10} {ent['score']:.2f}")

# Step 4: save the outputs
