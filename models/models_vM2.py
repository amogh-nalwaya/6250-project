import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform
from torch.autograd import Variable

import numpy as np

import math
import random
import sys
import time

sys.path.append('../')
#from constants import *
#from dataproc import extract_wvs
import extract_wvs_edited

class BaseModel(nn.Module):

    def __init__(self, embed_file, dicts, lmbda=0, dropout=0.5, gpu=True, embed_size=100):
        super(BaseModel, self).__init__()
        self.gpu = gpu
        self.embed_size = embed_size
        self.embed_drop = nn.Dropout(p=dropout)

        #make embedding layer
        if embed_file:
            print("loading pretrained embeddings...")
            W = torch.Tensor(extract_wvs_edited.load_embeddings(embed_file))
            self.embed = nn.Embedding(W.size()[0], W.size()[1])
            self.embed.weight.data = W.clone()
            
        else:
            #add 2 to include UNK and PAD
            vocab_size = len(dicts[0])
            self.embed = nn.Embedding(vocab_size+2, embed_size)


    def get_loss(self, yhat, target, diffs=None):
        
        #calculate the BCE
        loss = F.binary_cross_entropy(yhat, target)

        #keep track of this while training
        print("loss: %.5f" % loss.data[0])

        return loss

    def params_to_optimize(self):
        return self.parameters()
    
    
class VanillaConv(BaseModel):

    def __init__(self, embed_file, kernel_size, num_filter_maps, gpu=True, dicts=None, embed_size=100, dropout=0.5):
        super(VanillaConv, self).__init__(embed_file, dicts, dropout=dropout, embed_size=embed_size) 
        
        #initialize conv layer as in 2.1
        # PyTorch SYNTAX: 
        # self.embed_size = in_channels
        # num_filter_maps = out_channels
        # kernel_size of 3 = trigrams
        self.conv = nn.Conv1d(self.embed_size, num_filter_maps, kernel_size=kernel_size)
        xavier_uniform(self.conv.weight)

        #linear output
        self.fc = nn.Linear(num_filter_maps, ) # Need to make size based on number of kernels
        xavier_uniform(self.fc.weight)

    def forward(self, x, y, target, desc_data=None, get_attention=False):
        #embed
        x = self.embed(x)
        x = self.embed_drop(x)
        x = x.transpose(1, 2)
        
        #conv/max-pooling
        c = self.conv(x)
        
        x = F.max_pool1d(F.tanh(c), kernel_size=c.size()[2])
        x = x.squeeze(dim=2) # SQUEEZE?

        #linear output
        x = self.fc(x)

        #final sigmoid to get predictions
        yhat = F.sigmoid(x)
        loss = self.get_loss(yhat, target)
        return yhat, loss
