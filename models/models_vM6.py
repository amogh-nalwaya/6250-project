import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform
from torch.autograd import Variable
from torch.optim import Adam

import numpy as np

import math
import random
import sys
import time

sys.path.append("/home/miller/Documents/BDH NLP/Code/Github/6250-project/")
sys.path.append("/home/miller/Documents/BDH NLP/Code/Github/6250-project/datasets/")
from constants import *
from dataproc import extract_wvs

class BaseModel(nn.Module):

    def __init__(self, embed_file, dicts, dropout=0.5, gpu=True, embed_size=100):
        super(BaseModel, self).__init__()
        self.gpu = gpu
        self.embed_size = embed_size
#        self.embed_drop = nn.Dropout(p=dropout)

        #make embedding layer
        if embed_file:
            print("loading pretrained embeddings...")
            W = torch.Tensor(extract_wvs.load_embeddings(embed_file))
            print("Size of embedding matrix")
            print(W.size())
            self.embed = nn.Embedding(W.size()[0], W.size()[1])
            self.embed.weight.data = W.clone()
            
        else:
            #add 2 to include UNK and PAD
            vocab_size = len(dicts[0])
#            vocab_size = 10 # TEMP
            self.embed = nn.Embedding(vocab_size+2, embed_size)


####### NEW #########

    def weighted_bce(self, output, target, weights=None):
        
        '''Computes weighted binary cross entropy given predictions and ground truths (target).
           weights: [weight_neg_class, weight_pos_class]'''
        
        
        if weights is not None:
            assert len(weights) == 2
            
            loss = weights[1] * (target * torch.log(output)) + \
                   weights[0] * ((1 - target) * torch.log(1 - output))
                   
#            print("New loss fx")
#            print("Weight for negative class: " + str(weights[0]))
#            print("Weight for positive class: " + str(weights[1]))
        else:
            loss = target * torch.log(output) + (1 - target) * torch.log(1 - output)
    
        return torch.neg(torch.mean(loss))

#######################################################################

    def get_loss(self, yhat, target, diffs=None):
        
        #calculate the BCE
        loss = F.binary_cross_entropy(yhat, target)

        #keep track of this while training
#        print("loss: %.5f" % loss.data[0])

        return loss

    def params_to_optimize(self):
        return self.parameters()
        
class ConvEncoder(BaseModel):

    def __init__(self, embed_file, kernel_sizes, num_filter_maps, gpu=True, dicts=None, embed_size=100, dropout=0.5, conv_activation = "selu"):
        super(ConvEncoder, self).__init__(embed_file, dicts, dropout=dropout, embed_size=embed_size) 
        
        self.kernel_sizes = kernel_sizes        
        
        # Setting non-linear activation on feature maps
        self.conv_activation = getattr(F, conv_activation) # Equivalent to F.[conv_activation]
        print(self.conv_activation)
        
        # Initializing convolutional layers
        self.conv_layers = [nn.Conv1d(in_channels=embed_size, out_channels=num_filter_maps, 
                            kernel_size=int(kernel_size)) for kernel_size in kernel_sizes]
        
        for i, conv_layer in enumerate(self.conv_layers):
            self.add_module('conv_%d' % i, conv_layer) # Add convolutional modules, one for each kernel size
            
        # dropout
        self.dropout = nn.Dropout(p=dropout)

        # fully-connected layer         
        maxpool_output_dim = num_filter_maps * len(kernel_sizes)
        self.fc = nn.Linear(maxpool_output_dim, 1) 
        xavier_uniform(self.fc.weight)

    def forward(self, x, target):
        
        #embed
        x = self.embed(x)
#        x = self.embed_drop(x)
        x = x.transpose(1, 2) # Transposing from word vectors as rows --> word vectors as columns  ???
                  
        # Aggregating outputs for each filter for each kernel size
        filter_outputs = []
        for i in range(len(self.conv_layers)):
            conv_layer = getattr(self, 'conv_{}'.format(i)) # Equivalent to self.conv_i
            filter_outputs.append(
                    self.conv_activation(conv_layer(x)).max(dim=2)[0]) # .max() returns 2 arrays; 0. max vals, 1. argmax (max indices across desired dimension --> dim=2 is across columns in 3d tensor)
                
        
        x = torch.cat(filter_outputs, dim=1) if len(filter_outputs) > 1 else filter_outputs[0]

        #linear output
        x = self.dropout(x) # Multiplying by 2 when dropout = 0.5? Should test
        x = self.fc(x)

        #final sigmoid to get predictions
        yhat = F.sigmoid(x)
        y = yhat.squeeze()
#        loss = self.get_loss(y, target)
        loss = self.weighted_bce(y,target, [.1,1])
        
        return y, loss



































