# NLP for healthcare
## Predicting 30-day ICU readmissions using structured and unstructured data in MIMIC III
### Data Processing for unstructured text
* All data processing scripts for unstructured data are contained in the *dataproc* directory.
* Process NOTEEVENTS to get word vectors using *data_processing_script.py*.
    * Write Discharge summaries using *get_discharge_summaries.py*
    * Build vocab from discharge summaries using *build_vocab.py*.
    * Train word embeddings on all words using *word_embeddings.py*.
    * Write trained word embeddings with our vocab using *gensim_to_embeddings* method in *extract_wvs.py*.

