#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 20:07:33 2018

@author: miller
"""
from sklearn.metrics import *
import pandas as pd
import numpy as np

data_path = "/home/miller/Documents/BDH NLP/Data/"
data = pd.read_csv(data_path + "disch_summary_readmission_labels.csv")

### Randomly split data into train, validation and test sets
train, validation, test = np.split(data.sample(frac=1), [int(.89999*len(data)), int(.95*len(data))])

print("Size of train set: " + str(len(train)))
print("Size of val set: " + str(len(validation)))
print("Size of test set: " + str(len(test)))

### Sorting by length for batching
for splt, df in zip(["train","val","test"],[train,validation,test]): 
    df['length'] = df.apply(lambda row: len(str(row['TEXT']).split()), axis=1)
    df.sort_values(['length'], inplace=True)
    df.reset_index(inplace=True, drop=True)
    df.to_csv('%s/%s_sorted.csv' % (data_path, splt), index=False)
    
### Appendind data into single data set
all_data = train.append(validation)
all_data = all_data.append(test)
all_data.to_csv(data_path + 'all_data_sorted.csv', index=False)

### Saving only subj id and hadm id
all_data.drop(['TEXT', 'READMISSION', 'length'], axis=1, inplace=True)
all_data.to_csv(data_path + 'subj_hadm_ids_sorted.csv', index=False)

all_data = pd.read_csv(data_path + 'subj_hadm_ids_sorted.csv')

