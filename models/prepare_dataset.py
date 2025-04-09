import pandas as pd

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

def prepare_data(file_path='processed_notes.csv'):
    # Step I: load data
    notes_df = pd.read_csv(file_path)

    # Step II: clean data
    notes_expand_df = expand_token_tag_rows(notes_df)

    # label maps
    labels_to_ids = {k: v for v, k in enumerate(notes_expand_df.tag.unique())}
    ids_to_labels = {v: k for v, k in enumerate(notes_expand_df.tag.unique())}

    # Define tokens to exclude
    junk_tokens = {" ", "", "_", "___", "\t", "\n"}
    # Filter out rows where the token is in junk_tokens
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

    return {
        "data": notes_grouped_df,
        "labels_to_ids": labels_to_ids,
        "ids_to_labels": ids_to_labels
    }