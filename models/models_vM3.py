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

sys.path.append('../')
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
            self.embed = nn.Embedding(W.size()[0], W.size()[1])
            self.embed.weight.data = W.clone()
            
        else:
            #add 2 to include UNK and PAD
#            vocab_size = len(dicts[0])
            vocab_size = 10 # TEMP
            self.embed = nn.Embedding(vocab_size+2, embed_size)


    def get_loss(self, yhat, target, diffs=None):
        
        #calculate the BCE
        loss = F.binary_cross_entropy(yhat, target)

        #keep track of this while training
#        print("loss: %.5f" % loss.data[0])

        return loss

    def params_to_optimize(self):
        return self.parameters()
    
    
class ConvEnc(BaseModel):

    def __init__(self, embed_file, kernel_sizes, num_filter_maps, gpu=True, dicts=None, embed_size=100, dropout=0.5, conv_activation = "selu"):
        super(ConvEnc, self).__init__(embed_file, dicts, dropout=dropout, embed_size=embed_size) 
        
        self.kernel_sizes = kernel_sizes        
        
        # Setting activation on convolutional layer
        if conv_activation == "relu":
            self.conv_activation = F.relu
        if conv_activation == "elu":
            self.conv_activation = F.elu
        if conv_activation == "selu":
            self.conv_activation = F.selu
        if conv_activation == "tanh":
            self.conv_activation = F.tanh
        
        # Initializing convolutional layer(s)
        if len(self.kernel_sizes) == 0: # If only one filter/kernel size..
        
            self.conv = nn.Conv1d(self.embed_size, num_filter_maps, kernel_size=kernel_sizes)
            xavier_uniform(self.conv.weight)
            
        else: # If multiple filter sizes..
            self.conv_layers = [nn.Conv1d(in_channels=embed_size,
                                    out_channels=num_filter_maps,
                                    kernel_size=kernel_size)
                                    for kernel_size in kernel_sizes]
            for i, conv_layer in enumerate(self.conv_layers):
                self.add_module('conv_%d' % i, conv_layer)
            
        # dropout
        self.dropout = nn.Dropout(p=dropout)

        # fully-connected layer         
        maxpool_output_dim = num_filter_maps * len(kernel_sizes)
        self.fc = nn.Linear(maxpool_output_dim, 1) 
        xavier_uniform(self.fc.weight)

    def forward(self, x, target):
        
        #embed
#        x = self.embed(x)
#        x = self.embed_drop(x)
#        x = x.transpose(1, 2) # Transposing from word vectors as rows --> word vectors as columns  ???
                  
        if len(self.kernel_sizes) == 0: # If only one filter..
                                
            #conv/max-pooling
            c = self.conv(x)
            x = F.max_pool1d(self.conv_activation(c), kernel_size=c.size()[2]) # RELU OR ELU > tanh?
            x = x.squeeze(dim=2) # squeeze eliminates singleton dimensions

        else:
            
            filter_outputs = []
            for i in range(len(self.conv_layers)):
                conv_layer = getattr(self, 'conv_{}'.format(i))
                filter_outputs.append(
                        self.conv_activation(conv_layer(x)).max(dim=2)[0])
            
#            print(len(filter_outputs))
#            print(filter_outputs[:3])
            
            # max-pooling
            x = torch.cat(filter_outputs, dim=1) if len(filter_outputs) > 1 else filter_outputs[0]

        #linear output
        x = self.dropout(x) # Multiplying by 2 when dropout = 0.5? Should test
        x = self.fc(x)

        #final sigmoid to get predictions
        yhat = F.sigmoid(x)
        y = yhat.squeeze()
#        print(y.size())
        loss = self.get_loss(y, target)
        return y, loss

########################################

### Testing ConvEnc Class

# Initializing fake data

embed_size = 100
num_docs = 150
doc_len = 20
num_feat_maps = 5
kernels = [3]

data = np.random.random(embed_size * num_docs * doc_len).reshape(num_docs,embed_size,doc_len)
x = Variable(torch.FloatTensor(data))
y = Variable(torch.FloatTensor(np.random.random(num_docs)), requires_grad=False)

start = time.time()

# Initializing model and optimizer
model = ConvEnc(False, kernels, num_feat_maps, True, None, embed_size, 0.5, "selu")
optimizer = Adam(model.params_to_optimize())
model.train() # PUTS MODEL IN TRAIN MODE 
    
      
for i in range(1000):
           
    optimizer.zero_grad()
    
    pred, loss = model(x,y)      # FORWARD PASS
    
    loss.backward()
    optimizer.step()
    
end = time.time()
print(end-start)

########################################

### Max pool --> dropout --> FC  Example
data = np.random.random(8).reshape(2,2,2)
test = Variable(torch.FloatTensor(data))
drop = nn.Dropout(p=0.5)
fc = nn.Linear(2, 1) 

x = F.max_pool1d(test, kernel_size=test.size()[2]) # RELU OR ELU > tanh?
           
x_sq = x.squeeze(dim=2)

drop_x = drop(x_sq)

fc_x = fc(drop_x)

F.sigmoid(fc_x)
































