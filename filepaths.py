#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 19:54:37 2018

@author: miller
"""

import csv
import argparse
import os 
import numpy as np
import sys
import time
from tqdm import tqdm
from collections import defaultdict

# Adding path to import files from --> need to make more general, add to constants.py
abs_path = os.path.abspath(__file__)
file_dir = os.path.dirname(abs_path)
parent_dir = os.path.dirname(file_dir)

print(__file__)
print("abs_path: " + str(abs_path))
print("file dir: " + str(file_dir))
print("parent dir: " + str(parent_dir))