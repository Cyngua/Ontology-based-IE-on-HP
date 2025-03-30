'''
Character span annotation into BIO format
'''
import pandas as pd
import spacy
import tqdm

# spacy nlp pipeline
nlp = spacy.blank("en")
nlp.add_pipe('sentencizer')

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
    Returns a list of list (token.text, label) tuples.
    '''

    text_row = notes_df[notes_df['note_id'] == note_id]
    if text_row.empty:
        return []

    text = text_row['text'].iloc[0]
    doc = nlp(text.replace('\n', ' '))
    labels = ["O"] * len(doc)

    # extract the annotations spans according to note id
    spans = annotations_df[annotations_df['note_id'] == note_id]
    for _, row in spans.iterrows():
        start_idx = row["start"]
        end_idx = row["end"]
        concept_id = row["concept_id"]
        span = doc.char_span(start_idx, end_idx)
        if span:
            group = semantic_dict.get(concept_id, "UNK")  # fallback if missing
            labels[span.start] = f"B-{group}"
            labels[span.start+1: span.end] = [f"I-{group}"]*(span.end-span.start-1)

    # Split token-labels by sentence
    sentence_bio_pairs = []
    for sent in doc.sents:
        sent_tokens = [token.text for token in sent]
        sent_labels = labels[sent.start : sent.end]
        sentence_bio_pairs.append(list(zip(sent_tokens, sent_labels)))

    return sentence_bio_pairs

def main():
    rows = []  # list to collect output rows
    note_id_list = notes_df['note_id'].unique()
    for note_id in tqdm.tqdm(note_id_list):
        sentence_bio_pairs = char_spans_to_bio(note_id)
        for sentence_id, token_label_pairs in enumerate(sentence_bio_pairs):
            tokens = [token for token, _ in token_label_pairs]
            labels = [label for _, label in token_label_pairs]
            row_entries = [note_id, sentence_id, tokens, labels]
            rows.append(row_entries)
    df = pd.DataFrame(rows, columns=["note_id", "sentence_id", "sentence", "tags"])
    print(df.head())
    df.to_csv('processed_notes.csv', index=None)

def test():
    print(char_spans_to_bio('10097089-DS-8')[60])

if __name__ == "__main__":
    main()
