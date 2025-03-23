# Clinical Information Extraction from H&P Notes

This project focuses on extracting clinically relevant entities (e.g., conditions, procedures, findings) from History & Physical (H&P) notes using named entity recognition (NER) techniques. It leverages annotated clinical data with SNOMED CT concept IDs and aims to build a high-performance information extraction pipeline for real-world applications.

---

## Dataset
- **Source:** mimic-iv de-identified clinical notes.
- **Annotations:** Entity spans annotated with `start`, `end`, and `SNOMED CT concept_id`.
- **Example:**
  ```
  note_id,start,end,concept_id
  10060142-DS-9,179,190,91936005
  ```
- **Semantic Labels:** Concept IDs are mapped to entity types such as `CONDITION`, `PROCEDURE`, `FINDING`, etc.

---

## Workflow Overview

### 1. **Preprocessing**
- Load raw H&P text notes.
- Tokenize using `spaCy` or `Hugging Face` tokenizer.
- Align token spans with annotated character offsets.

### 2. **Concept Mapping**
- Map each `concept_id` to an entity group (e.g., `PROCEDURE`, `CONDITION`) using: SNOMED CT hierarchy

### 3. **NER Data Formatting**
- Convert aligned tokens into BIO-tagged format (e.g., `B-CONDITION`, `I-PROCEDURE`, `O`).
- Export training data in CoNLL format or JSON (for spaCy or HuggingFace).

### 4. **Model Training**
- Fine-tune a pretrained clinical language model (e.g., `BioClinicalBERT`, `SciBERT`) using token classification.
- Train using Hugging Face Transformers or spaCy pipelines.

### 5. **Evaluation**
- Evaluate using precision, recall, and F1-score at the entity level.
- Perform qualitative comparison against baseline models (e.g., `SciSpacy`, `QuickUMLS`).

---

## ✅ Goals

- Build a robust, domain-adapted NER model for clinical entity extraction.
- Normalize extracted terms to SNOMED CT for downstream applications.
- Enable accurate information extraction from narrative clinical text, particularly H&P sections.

---

## 🚀 Future Work

- Extend to relation extraction (e.g., link conditions to medications).
- Integrate negation and temporality detection.
- Deploy as a clinical text mining tool or API.

---

## 📎 Dependencies

- Python 3.8+
- `spaCy`, `transformers`, `pandas`, `scikit-learn`, `nltk`
- (Optional) `QuickUMLS`, `UMLS API`, `BioPortal`
