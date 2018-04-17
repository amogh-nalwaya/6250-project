import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import MaxAbsScaler
from sklearn.metrics import roc_curve, auc
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


data_path = "/home/miller/Documents/BDH NLP/Data/"

X, y = load_svmlight_file(data_path + "struc_data.svmlight")
print("data loaded")

# Converting to csr sparse matrix form
X = X.tocsr()

# Splitting into train, val and test
X_train = X[:-5000]
X_val = X[-5000:-2500]
X_test = X[-2500:]

X_test.shape[0]

y_train = y[:-5000]
y_val = y[-5000:-2500]
y_test = y[-2500:]

# Standardizing features
scaler = MaxAbsScaler().fit(X_train)
X_train_std = scaler.transform(X_train)
X_val_std = scaler.transform(X_val)
X_test_std = scaler.transform(X_test)

class FeedForwardNet(nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super(FeedForwardNet, self).__init__()
        self.hidden1 = nn.Linear(n_input, n_hidden)
        self.hidden2 = nn.Linear(n_hidden, n_hidden)
        self.out = nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = self.out(x)
        return x

net = FeedForwardNet(n_input=X.shape[1], n_hidden = 100, n_output=1)
print("defined network")

criterion = nn.BCEWithLogitsLoss()
#optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
optimizer = optim.Adam(net.parameters(), weight_decay=0, lr=0.001)

print("starting training")
for epoch in range(10):  # loop over the dataset multiple times
    running_loss = 0.0
    
    i = 0
    k = 0
    batchsize = 25
    while i < X_train.shape[0]/2:
        
        batch_size_safe = min(batchsize, X_train.shape[0] - i) # Avoiding going out of range
        
        Xtrainsample = X_train_std[i:i+batch_size_safe].todense()
        ytrainsample = y_train[i:i+batch_size_safe]

        inputs = torch.from_numpy(Xtrainsample.astype('float32'))
        labels = torch.from_numpy(ytrainsample.astype('float32')).view(-1,1)

        # wrap them in Variable
        inputs, labels = Variable(inputs), Variable(labels)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        # backward
        loss.backward()
        # optimize
        optimizer.step()

        # print statistics
        running_loss += loss.data[0]

        if k % 100 == 99:    # print every 100 mini-batches
            print('[epoch:%d, batch:%5d] loss over last 100 batches: %.3f' %
                  (epoch + 1, k + 1, running_loss / 100))
            running_loss = 0.0
        
        k = k+1
        i = i+batchsize
        
    print('Finished Epoch')
    print("Predicting on validation set")
    
    y_true = []
    y_scores = []
    
    i = 0
    
    while i < X_test.shape[0]:
        
        batch_size_safe = min(batchsize, X_val.shape[0] - i) # Avoiding going out of range
        
        Xtestsample = X_val_std[i:i+batch_size_safe].todense()
        ytestsample = y_val[i:i+batch_size_safe]
    
    
        inputs = torch.from_numpy(Xtestsample.astype('float32'))
        labels = torch.from_numpy(ytestsample.astype('float32')).view(-1,1)
        
        outputs = net(Variable(inputs))
        outputs = F.sigmoid(outputs)
        y_true.extend(labels.numpy().flatten().tolist())
        y_scores.extend(outputs.data.numpy().flatten().tolist())
        i = i + batchsize
        
    print("finished predicting on validation set")
    
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    auc_ffnet = auc(fpr, tpr)
    print("AUC: " + str(auc_ffnet))
    

