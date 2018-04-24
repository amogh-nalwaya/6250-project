# NLP for healthcare
## Predicting 30-day ICU readmissions using structured and unstructured data in MIMIC III

###Structured data
* Logic for the structured network modeling is contained within files in the *Structured* directory
* The ETL logic for structured network can be found in *structured_etl_part1.scala* and *structured_etl_part2.py*
* Modeling logic is contained within *struc_net.py*
* Modeling scripts are run through the wrapper *py_train_struc.py* which also allows for random search for hyperparameter tuning for the network.

###Unstructured data

#### Data Processing for unstructured text
* All data processing scripts for unstructured data are contained in the *dataproc* directory.
* Process NOTEEVENTS to get word vectors using *data_processing_script.py*.
    * Write Discharge summaries using *get_discharge_summaries.py*
    * Build vocab from discharge summaries using *build_vocab.py*.
    * Train word embeddings on all words using *word_embeddings.py*.
    * Write trained word embeddings with our vocab using *gensim_to_embeddings* method in *extract_wvs.py*.

#### Modeling scripts
* Modeling scripts can be found in the *models* directory.
* To run the modeling scripts, arguments ca



