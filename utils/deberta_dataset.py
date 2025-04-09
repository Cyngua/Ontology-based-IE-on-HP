import numpy as np
import torch
from torch.utils.data import Dataset

class DeBERTaDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len, labels_to_ids):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.labels_to_ids = labels_to_ids

    def __getitem__(self, index):
        sentence = self.data.sentence[index]
        word_labels = self.data.tags[index]

        encoding = self.tokenizer(sentence,
                                  is_split_into_words=True,
                                  padding='max_length',
                                  truncation=True,
                                  max_length=self.max_len)

        labels = [self.labels_to_ids[label] for label in word_labels]

        encoded_labels = np.ones(len(encoding["input_ids"]), dtype=int) * -100
        word_ids = encoding.word_ids()

        previous_word_idx = None
        for idx, word_idx in enumerate(word_ids):
            if word_idx is None:
                continue
            elif word_idx != previous_word_idx:
                encoded_labels[idx] = labels[word_idx]
            previous_word_idx = word_idx

        item = {key: torch.tensor(val) for key, val in encoding.items()}
        item['labels'] = torch.tensor(encoded_labels)

        return item

    def __len__(self):
        return self.len