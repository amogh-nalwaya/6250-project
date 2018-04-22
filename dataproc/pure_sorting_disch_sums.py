#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 18:51:54 2018

@author: miller
"""
import numpy as np
import pandas as pd

data_path = "/home/miller/Documents/BDH NLP/Data/"
sorted_data = pd.read_csv(data_path + 'all_data_sorted.csv')

#sorted_data.sort_values(['length'], inplace=True)

sorted_data.to_csv(data_path + 'sorted_disch_sums_no_split.csv')

### Ordering training set (longest discharge summaries first for gpu memory purposes) ###
train = pd.read_csv(data_path + "sorted_sums_matched_struc_train_full.csv")
train.sort_values("length", inplace=True, ascending = False)

train.to_csv(data_path + 'sorted_sums_matched_struc_train_reversed.csv', index=False)
#test[["SUBJECT_ID", "HADM_ID"]].to_csv('sorted_sums_matched_struc_train_reversed_ids.csv')


### Ordering validation set ###
val = pd.read_csv(data_path + "sorted_sums_matched_struc_val.csv")
val.sort_values("length", inplace=True, ascending = False)
val.to_csv(data_path + 'sorted_sums_matched_struc_val_reversed.csv', index=False)

### Ordering test set ###
test = pd.read_csv(data_path + "sorted_sums_matched_struc_test.csv")
test.sort_values("length", inplace=True, ascending = False)
test.to_csv(data_path + 'sorted_sums_matched_struc_test_reversed.csv', index=False)


# Combining data sets
all_data = train.append(val)
all_data = all_data.append(test)
all_data[['SUBJECT_ID','HADM_ID']].to_csv(data_path + 'all_reversed_ids.csv')


### Testing to make sure lengths were sorted properly
lens = np.array(all_data['length'])
lens_shift = np.array(all_data['length'].shift(-1))

if np.sum( (lens - lens_shift) < 0) == 2:
    print("All good")



