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

#X, y = load_svmlight_file("data.svmlight")
#print("data loaded")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

scaler = MaxAbsScaler().fit(X)
X_train_transformed = scaler.transform(X_train)
X_test_transformed = scaler.transform(X_test)
print("data scaled")


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
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

print("starting training")
for epoch in range(10):  # loop over the dataset multiple times
    running_loss = 0.0
    
    i = 0
    k = 0
    batchsize = 32
    while i<=(X_train.shape[0] - batchsize):
        Xtrainsample = X_train_transformed.tocsr()[i:i+batchsize].todense()
        ytrainsample = y_train[i:i+batchsize]

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

        if k % 100 == 99:    # print every 10 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, k + 1, running_loss / 100))
            running_loss = 0.0
        
        k = k+1
        i = i+batchsize
        
    print('Finished Training')
    print("start testing")
    
    y_true = []
    y_scores = []
    
    i = 0
    while i<=(X_test.shape[0]-batchsize):
        Xtestsample = X_test_transformed.tocsr()[i:i+batchsize].todense()
        ytestsample = y_test[i:i+batchsize]
    
    
        inputs = torch.from_numpy(Xtestsample.astype('float32'))
        labels = torch.from_numpy(ytestsample.astype('float32')).view(-1,1)
        
        outputs = net(Variable(inputs))
        outputs = F.sigmoid(outputs)
        y_true.extend(labels.numpy().flatten().tolist())
        y_scores.extend(outputs.data.numpy().flatten().tolist())
        i = i + batchsize
    print("finished testing")

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    auc_ffnet = auc(fpr, tpr)
    print("AUC: " + str(auc_ffnet))