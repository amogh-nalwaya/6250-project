"""
    Data loading methods
"""
from collections import defaultdict
import csv
import math
import numpy as np
import sys

from constants import *

class Batch:
    def __init__(self):
        self.docs = []
        self.labels = []
        self.hadm_ids = []
        #self.code_set = set()
        self.length = 0
        self.max_length = MAX_LENGTH
        #self.desc_embed = desc_embed
        #self.descs = []

    def add_instance(self, row, w2ind):
        """
            Makes an instance to add to this batch from given row data, with a bunch of lookups
        """
        #labels = set()
        hadm_id = int(row[1])
        text = row[2]
        
        #length = int(row[4]) # WHERE DOES THIS COME FROM?
        
        label = int(row[3]) # NEW --> PUT LABEL IN ith (3rd?) COLUMN of FINAL DATASET
                
        #OOV words are given a unique index at end of vocab lookup
        text = [int(w2ind[w]) if w in w2ind else len(w2ind)+1 for w in text.split()]
        
        # ADDED TO REPLACE AMBIGUOUS LENGTH ASSIGNMENT ABOVE
        length = len(text)
        
        #truncate long documents
        if len(text) > self.max_length:
            text = text[:self.max_length]

        #build instance
        self.docs.append(text)
        self.labels.append(label)
        self.hadm_ids.append(hadm_id)
                
        #reset length
        self.length = min(self.max_length, length) # NEED TO PAD TO EITHER MAX LENGTH OR LONGEST DOC LENGTH (?)
           
        
    def pad_docs(self):
        #pad all docs to have self.length
        padded_docs = []
        for doc in self.docs:
            if len(doc) < self.length:
                doc.extend([0] * (self.length - len(doc)))
            padded_docs.append(doc)
        self.docs = padded_docs

    def to_ret(self):
        return np.array(self.docs), np.array(self.labels), np.array(self.hadm_ids) #, self.code_set, np.array(self.descs)

def data_generator(filename, dicts, batch_size):

    """
        Inputs:
            filename: holds data sorted by sequence length, for best batching
            dicts: holds all needed lookups
            batch_size: the batch size for train iterations

        Output:
            Batch containing np array of data for training loop.
    """
    #ind2w, w2ind, ind2c, c2ind, dv_dict = dicts[0], dicts[1], dicts[2], dicts[3], dicts[5]
    ind2w, w2ind  = dicts[0], dicts[1] # DONT NEED ind2c, c2ind, or dv_dict
    
    with open(filename, 'r') as infile:
        r = csv.reader(infile)
        #header
        next(r) # ERROR HANDLING WHEN ON LAST ROW?
        cur_inst = Batch()
        for row in r:
            #find the next `batch_size` instances
            if len(cur_inst.docs) == batch_size:
                cur_inst.pad_docs()
                yield cur_inst.to_ret()
                # CREATE NEW BATCH INSTANCE
                cur_inst = Batch()
            #cur_inst.add_instance(row, ind2c, c2ind, w2ind, dv_dict, num_labels)
            cur_inst.add_instance(row, w2ind) # HAVE TO CHECK if we need to return w2ind
        cur_inst.pad_docs()
        yield cur_inst.to_ret()


def load_vocab_dict(vocab_file): # SHOULD CHANGE TO load_vocab_dicts*
    
    ''' Input: Path to vocabulary file.
    
        Output: Index:Word dictionary, Word:Index dictionary'''
        
    #reads vocab_file into two lookups
    ind2w = defaultdict(str)
    with open(vocab_file, 'r') as vocabfile:
        for i,line in enumerate(vocabfile):
            line = line.rstrip()
            if line != '':
                ind2w[i+1] = line.rstrip()
    w2ind = {w:i for i,w in ind2w.items()} # CHANGED FROM iteritems --> items
    return ind2w, w2ind

