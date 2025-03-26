'''
Character span annotation into BIO format
'''
import pandas as pd
import spacy
from spacy.tokens import Doc
import tqdm

# spacy nlp pipeline
nlp = spacy.blank("en")

# load the semantic group map
semantic_group_map = pd.read_csv('prepare/semantic_group_map.csv')
semantic_dict = dict(zip(semantic_group_map['concept_id'], semantic_group_map['group_abbr']))

# load notes df
notes_df = pd.read_csv('data/benchmark_data/mimic-iv_notes_training_set.csv')
# load annotations df
annotations_df = pd.read_csv('data/benchmark_data/train_annotations.csv')


def char_spans_to_bio(note_id):
    '''
    Given a single note id convert the span to bio
    Returns a list of (token.text, label) tuples.
    '''

    text_row = notes_df[notes_df['note_id'] == note_id]
    if text_row.empty:
        return []

    text = text_row['text'].iloc[0]
    doc = nlp(text)
    labels = ["O"] * len(doc)

    # extract the annotations spans according to note id
    spans = annotations_df[annotations_df['note_id'] == note_id]
    for _, row in spans.iterrows():
        start_idx = row["start"]
        end_idx = row["end"]
        concept_id = row["concept_id"]

        group = semantic_dict.get(concept_id, "UNK")  # fallback if missing

        labels[start_idx] = f"B-{group}"
        if end_idx > start_idx:
            labels[start_idx+1: end_idx+1] = f"I-{group}"

    return list(zip([token.text for token in doc], labels))

def main():
    pass

if __name__ == "__main__":
    main()
