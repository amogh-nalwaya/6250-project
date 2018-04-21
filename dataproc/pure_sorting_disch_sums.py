#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 18:51:54 2018

@author: miller
"""

import pandas as pd

data_path = "/home/miller/Documents/BDH NLP/Data/"
sorted_data = pd.read_csv(data_path + 'all_data_sorted.csv')

test = pd.read_csv(data_path + "sorted_sums_matched_struc_train_full.csv")

sorted_data.sort_values(['length'], inplace=True)

sorted_data.to_csv(data_path + 'sorted_disch_sums_no_split.csv')
