from transformers import pipeline

# Input I: model --> load model and tokenizer
# Input II: text file (txt/doc)

ner_pipe = pipeline(
    "ner",
    model=model,
    tokenizer=tokenizer,
    aggregation_strategy="simple",  # combines B/I tokens into full spans
    device=-1  # ensure it's using CPU
)

text = "The patient was prescribed metformin for type 2 diabetes."

predictions = ner_pipe(text)

for ent in predictions:
    print(f"{ent['word']} → {ent['entity_group']} (score: {ent['score']:.2f})")
