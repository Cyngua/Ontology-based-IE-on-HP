# Utils
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# torch essentials
import torch
from torch.utils.data import Dataset
from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'
print(device)


# Model evaluation
from transformers import EvalPrediction
from seqeval.metrics import accuracy_score, precision_score, recall_score, f1_score

# model and tokenizer
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import Trainer, TrainingArguments

MODEL_NAME = "medicalai/ClinicalBERT"

# Step I: load data
file_path = 'processed_notes.csv'
notes_df = pd.read_csv(file_path)

# Step II: clean data
def expand_token_tag_rows(df):
    """
    Expand each token and tag to its own row.
    Input: df with columns ['note_id', 'sentence_id', 'sentence', 'tags']
    Output: exploded DataFrame with one token and tag per row
    """
    # Ensure 'sentence' and 'tags' are lists (if loaded from CSV they might be strings)
    df["sentence"] = df["sentence"].apply(eval) if isinstance(df["sentence"].iloc[0], str) else df["sentence"]
    df["tags"] = df["tags"].apply(eval) if isinstance(df["tags"].iloc[0], str) else df["tags"]

    # Expand the sentence and tag columns
    exploded_df = df.explode(["sentence", "tags"]).reset_index(drop=True)

    # Rename columns for clarity
    exploded_df = exploded_df.rename(columns={
        "sentence": "token",
        "tags": "tag"
    })

    return exploded_df

notes_expand_df = expand_token_tag_rows(notes_df)

# label maps
labels_to_ids = {k: v for v, k in enumerate(notes_expand_df.tag.unique())}
ids_to_labels = {v: k for v, k in enumerate(notes_expand_df.tag.unique())}

# Define tokens to exclude
junk_tokens = {" ", "", "_", "___", "\t", "\n"}
# Filter out rows where token is in junk_tokens
notes_clean_df = notes_expand_df[~notes_expand_df["token"].isin(junk_tokens)].copy()
notes_clean_df = notes_clean_df[notes_clean_df["token"].str.strip() != ""]
# Group back to sentence-level
notes_grouped_df = (
    notes_clean_df
    .groupby(["note_id", "sentence_id"])
    .agg({
        "token": list,
        "tag": list
    })
    .reset_index()
    .rename(columns={"token": "sentence", "tag": "tags"})
)

# Step III: transformer dataset setup
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME) # TODO: add tokenizer
MAX_LEN = 128

class dataset(Dataset):
  def __init__(self, dataframe, tokenizer, max_len):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len

  def __getitem__(self, index):
        # step 1: get the sentence and word labels
        sentence = self.data.sentence[index]
        word_labels = self.data.tags[index]

        # step 2: use tokenizer to encode sentence (includes padding/truncation up to max length)
        encoding = self.tokenizer(sentence,
                             is_split_into_words=True,
                             return_offsets_mapping=True,
                             padding='max_length',
                             truncation=True,
                             max_length=self.max_len)

        # step 3: create token labels only for first word pieces of each tokenized word
        labels = [labels_to_ids[label] for label in word_labels]
        # create an empty array of -100 of length max_length
        encoded_labels = np.ones(len(encoding["offset_mapping"]), dtype=int) * -100
        # set only labels whose first offset position is 0 and the second is not 0
        i = 0
        for idx, mapping in enumerate(encoding["offset_mapping"]):
          if mapping[0] == 0 and mapping[1] != 0:
              if i < len(labels):  # avoid IndexError
                # overwrite label
                encoded_labels[idx] = labels[i]
                i += 1
        # step 4: turn everything into PyTorch tensors
        item = {key: torch.as_tensor(val) for key, val in encoding.items()}
        item['labels'] = torch.as_tensor(encoded_labels)

        return item

  def __len__(self):
        return self.len

data = notes_grouped_df
train_size = 0.8
train_dataset = data.sample(frac=train_size,random_state=200)
test_dataset = data.drop(train_dataset.index).reset_index(drop=True)
train_dataset = train_dataset.reset_index(drop=True)

print("FULL Dataset: {}".format(data.shape))
print("TRAIN Dataset: {}".format(train_dataset.shape))
print("TEST Dataset: {}".format(test_dataset.shape))

training_set = dataset(train_dataset, tokenizer, MAX_LEN)
testing_set = dataset(test_dataset, tokenizer, MAX_LEN)


# Step IV: Load Model
model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME, num_labels=len(labels_to_ids)) # TODO: load model
model.to(device)

# sanity Check
inputs = training_set[2]
input_ids = inputs["input_ids"].unsqueeze(0)
attention_mask = inputs["attention_mask"].unsqueeze(0)
labels = inputs["labels"].unsqueeze(0)
input_ids = input_ids.to(device)
attention_mask = attention_mask.to(device)
labels = labels.to(device)
outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
initial_loss = outputs[0]
print(initial_loss)

# Step V: Set up trainer
def compute_metrics(p: EvalPrediction):
    predictions = p.predictions
    labels = p.label_ids

    # Get the class with highest probability for each token
    predicted_ids = np.argmax(predictions, axis=-1)

    true_labels = []
    true_predictions = []

    for pred, label in zip(predicted_ids, labels):
        curr_preds = []
        curr_labels = []
        for p_id, l_id in zip(pred, label):
            if l_id != -100:
                curr_preds.append(ids_to_labels[p_id])  # predicted label string
                curr_labels.append(ids_to_labels[l_id])  # true label string
        true_predictions.append(curr_preds)
        true_labels.append(curr_labels)

    return {
        "accuracy": accuracy_score(true_labels, true_predictions),
        "precision": precision_score(true_labels, true_predictions),
        "recall": recall_score(true_labels, true_predictions),
        "f1": f1_score(true_labels, true_predictions)
    }

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    learning_rate=2e-5, # smaller learning rate
    seed=42, # for deterministic results
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=100,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=500,
    fp16=True, # mixed precision training
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=training_set,
    eval_dataset=testing_set,
    compute_metrics=compute_metrics,
)

trainer.train()

# Step VI: Save model
trainer.save_model("./saved_model")  # Save model, tokenizer, config, etc.
tokenizer.save_pretrained("./saved_model")

# Step VII: Plot evaluation
log_history = trainer.state.log_history

train_steps, train_loss = [], []
eval_steps, eval_loss, f1s = [], [], []

for entry in log_history:
    if "loss" in entry and "step" in entry:
        train_steps.append(entry["step"])
        train_loss.append(entry["loss"])
    if "eval_loss" in entry:
        eval_steps.append(entry["step"])
        eval_loss.append(entry["eval_loss"])
        f1s.append(entry.get("eval_f1", None))  # only if F1 was logged

plt.figure(figsize=(12, 5))

# Plot losses
plt.subplot(1, 2, 1)
plt.plot(train_steps, train_loss, label="Train Loss")
plt.plot(eval_steps, eval_loss, label="Eval Loss")
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.title("Training vs. Validation Loss")
plt.legend()

# Plot F1
plt.subplot(1, 2, 2)
if f1s[0] is not None:
    plt.plot(eval_steps, f1s, label="Eval F1", color='green')
    plt.xlabel("Steps")
    plt.ylabel("F1 Score")
    plt.title("Evaluation F1 over Time")
    plt.legend()

plt.tight_layout()
plt.savefig("figures/medical_ner_output.png")
