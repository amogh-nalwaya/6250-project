"""
    Various methods are kept here to keep the other code files simple
"""
import csv
import json
import math

import torch

#import models
from models import models_vM4 as models
from constants import *
import datasets
import persistence
import numpy as np

def pick_model(args, dicts):
    """
        Use args to initialize the appropriate model
    """
#    if "," in args.kernel_sizes:
    kernel_sizes = [size for size in args.kernel_sizes if size != ","] # Removing commas if multiple filter sizes passed
    print(kernel_sizes)
    
    if args.model == "rnn":
        model = models.VanillaRNN(args.embed_file, dicts, args.rnn_dim, args.cell_type, args.rnn_layers, args.gpu, args.embed_size,
                                  args.bidirectional)
    
    elif args.model == "conv_encoder":
                
        model = models.ConvEncoder(args.embed_file, args.kernel_sizes, args.num_filter_maps, args.gpu, dicts, args.embed_size, args.dropout)
    
    elif args.model == "conv_attn":
        model = models.ConvAttnPool(Y, args.embed_file, filter_size, args.num_filter_maps, args.lmbda, args.gpu, dicts,
                                    embed_size=args.embed_size, dropout=args.dropout)
    elif args.model == "saved":
        model = torch.load(args.test_model)
    
    if args.gpu:
        model.cuda()
        
    return model

def make_param_dict(args):
    """
        Make a list of parameters to save for future reference
    """
    #param_vals = [args.Y, args.filter_size, args.dropout, args.num_filter_maps, args.rnn_dim, args.cell_type, args.rnn_layers,
    #              args.lmbda, args.command, args.weight_decay, args.version, args.data_path, args.vocab, args.embed_file, args.lr]
    param_vals = [args.kernel_sizes, args.dropout, args.num_filter_maps,
        args.command, args.weight_decay, args.data_path,
        args.embed_file, args.lr]
    
    param_names = ["kernel_sizes", "dropout", "num_filter_maps", "command",
        "weight_decay", "data_path", "embed_file", "lr"]
    params = {name:val for name, val in zip(param_names, param_vals) if val is not None}
    return params

#def build_code_vecs(code_inds, dicts):
#    """
#        Get vocab-indexed arrays representing words in descriptions of each *unseen* label
#    """
#    code_inds = list(code_inds)
#    ind2w, ind2c, dv_dict = dicts[0], dicts[2], dicts[5]
#    vecs = []
#    for c in code_inds:
#        code = ind2c[c]
#        if code in dv_dict.keys():
#            vecs.append(dv_dict[code])
#        else:
#            #vec is a single UNK if not in lookup
#            vecs.append([len(ind2w) + 1])
#    #pad everything
#    vecs = datasets.pad_desc_vecs(vecs)
#    return (torch.cuda.LongTensor(code_inds), vecs)
#
#def get_num_labels(Y, version):
#    #get appropriate number of labels based on input Y and version
#    if Y == 'full':
#        if version == 'mimic2':
#            num_labels = FULL_LABEL_SIZE_II
#        elif version == 'mimic3':
#            num_labels = FULL_LABEL_SIZE
#        else:
#            print(version)
#    else:
#        num_labels = int(Y)
#    return num_labels
