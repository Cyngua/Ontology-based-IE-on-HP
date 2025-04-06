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
- Align token spans with annotated character offsets using `spaCy`tokenizer..

### 2. **Concept Mapping**
- Map each `concept_id` to an entity group (e.g., `PROCEDURE`, `CONDITION`) using: UMLS API

### 3. **NER Data Formatting**
- Convert aligned tokens into BIO-tagged format (e.g., `B-CONDITION`, `I-PROCEDURE`, `O`).
- Export training data in csv format.

### 4. **Experimental Design**
- ✅ Model Baseline: pretrained `BERT`, `Clinical-BERT`, `DeBERTa`.
- ✅ Model Finetuning: `BERT`, `Clinical-BERT`, `DeBERTa` using Huggingface Transformer Training Pipeline.
- Model Fusion: the best baseline model + CRF.
- Model Ensemble: integrate prediction results from the three finetuning models.
- (Optional)Hybrid approaches: (GNN + Transformer embeddings).

Justification: This experimental design (1) systematically **evaluates transformer-based clinical NER** by benchmarking general and clinical-domain models, (2) enhancing them via **CRF-based fusion**, **ensemble modeling**, and (optional) integrating transformer embeddings with graph neural networks for semantic enrichment using clinical ontologies.

### 5. **Evaluation**
- Evaluate using precision, recall, and F1-score at the entity level.
- Perform qualitative comparison against baseline model (`BERT`).

### 6. **Post-processing**
* Minimal: Decode model outputs back to text and semantic tags (merge B/I, skip O)
* Good to have: Extract word-tag pairs
* Best to have: Map words to SNOMED CT terms

| **Goal**            | **Tool/Library**                                                                 | **Purpose**                                           |
|---------------------|----------------------------------------------------------------------------------|-------------------------------------------------------|
| Decoding            | Model's tokenizer (e.g. HuggingFace's `tokenizer.decode`)                  | Turn token indices back into words                    |
| Post-processing     | `seqeval`, `pandas`                                                              | Convert IOB/IO format to structured entities          |
| Extraction          | Python dictionary/set logic, `collections.defaultdict`                           | Aggregate by tags                                     |
| Normalization       | 🔹 **ScispaCy**, 🔹 **QuickUMLS**, 🔹 **BioPortal API**                            | Map raw terms to SNOMED CT                            |
| SNOMED CT Integration | 🔸 `pysnomed` (if you have a license) or 🔸 FHIR terminology server              | Lookup and standardize with SNOMED CT concept codes   |

---

## AIM

- Build a robust, domain-adapted NER model for clinical entity extraction.
- Normalize extracted terms to SNOMED CT for downstream applications.
- Enable accurate information extraction from narrative clinical text, particularly H&P sections.

---
## Results - Baseline (Feature Extraction)
|               | Accuracy | Precision | Recall | F1     |
|---------------|----------|-----------|--------|--------|
| BERT          | 0.8732	 | 0.4909    | 0.4690 | 0.4797 |
| Clinical-BERT | 0.8123	 | 0.2500    | 0.0184 | 0.0344 |
| DeBERTa       | **0.8787**	 | 0.5207    | 0.4942 | **0.5071** |

Notes: `Clinical-BERT` validation loss has reached its "local minimum", but the evaluation metrics are still terrible. `deBerta` has very fluctuating training loss curve, while the evaluation metrics shows that it truely has the best potential to predict the token classification on the baseline.

Figure 1: Bert baseline
![bert baseline](figures/baseline_bert.png)
Figure 2: DeBerta baseline
![deberta baseline](figures/baseline_deberta.png)

## Results - Model Fine-tuning
|               | Accuracy | Precision | Recall | F1     | Validation Loss | Train Runtime |
|---------------|----------|-----------|--------|--------|-----------------|---------------|
| BERT          | 0.9288   | 0.7144    | 0.7751 | 0.7435 | 0.2834          | 1239.1881     |
| Clinical-BERT | 0.9284   | 0.7129    | 0.7629 | 0.7370 | 0.2256          | 433.2273      |

Training progress of clinical-BERT:

Figure 3: Clinical-Bert Finetuning
![clinical bert finetuning](figures/clinical_bert_output_2.png)

---

## Future Work

- Extend to relation extraction (e.g., link conditions to medications).
- Integrate negation and temporality detection.
- Deploy as a clinical text mining tool or API.

---

## Dependencies

- Python 3.8+
- `spaCy`, `transformers`, `pandas`, `scikit-learn`
- (Optional) `UMLS API`

Virtual environment setup:
```
# 1. Create the environment
module load miniconda
conda create -n clinical-ner python=3.9 -y
conda activate clinical-ner

# 2. Install PyTorch and CUDA support
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 3. Install the rest libraries via pip
pip install -r requirements.txt
```


## Tips and Issue Handling

### Project Data Mart

We built a lightweight project data mart using SQLite to store and manage structured information related to annotated clinical notes, concept mappings, and semantic group metadata. The database schema captures relationships between notes, concept annotations, and semantic groupings to support downstream data analysis and querying. To populate the database with CSV data, you can use the SQLite command-line interface as follows:
```
sqlite3 database.sqlite
sqlite> .mode csv
sqlite> .import <csv_file> <table_name>
sqlite> .exit
```

![alt text](figures/schema.png)

### Setup jobs on HPC

Shell script template
```sh
#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gpus=a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G                               # Request 10 GiB of memory
#SBATCH --time=6:00:00                         # Set a maximum runtime of 6 hours
#SBATCH --output=logs/medical_ner_train_%j.out  # Save standard output to log file
#SBATCH --error=logs/medical_ner_train_%j.err   # Save error output to log file

# Run the Python script
python models/medical_ner_train.py

# Print job finish time
echo "Job finished at $(date)"
```

Authorize and run the job
```sh
chmod +x ~/run_medical_ner_train.sh
```
```sh
sbatch ~/run_medical_ner_train.sh
```
### Issue I: deBerta attention score overflow
Open the file directly in Terminal with an editor (replace the virtual environment name with yours)
```bash
vim ~/.conda/envs/clinical-ner/lib/python3.9/site-packages/transformers/models/deberta/modeling_deberta.py
```
To search for specific line number: type `:290` and press Enter.<br>
In vim: Press `i` to switch to `__INSERT__` mode.<br>
Replace the following code:
```
attention_scores = attention_scores.masked_fill(~(attention_mask), torch.finfo(query_layer.dtype).min)
```
with this line:
```
attention_scores = attention_scores.masked_fill(~(attention_mask), -1e04)
```
Save and exit the file: Press `Esc`, type `:wq`, then press `Enter`.

