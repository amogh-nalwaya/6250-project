import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import MaxAbsScaler
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
import argparse
import json
from collections import defaultdict
import datetime
import os
from torch.nn.init import xavier_uniform

# Adding relative path to python path
abs_path = os.path.abspath(__file__)
file_dir = os.path.dirname(abs_path)
parent_dir = os.path.dirname(file_dir)
sys.path.append(parent_dir)

#code_path = "/home/miller/Documents/BDH NLP/Code/Github/6250-project"
#sys.path.append(code_path)

from evaluation import evaluation_vM3 as evaluation

# Need to pass array of hidden layer sizes

class FeedForwardNet(nn.Module):
    def __init__(self, n_input, n_output, fc_layer_size_list, fc_dropout_list, fc_activation):
        
        super(FeedForwardNet, self).__init__()
        self.fc_layer_size_list = fc_layer_size_list
        self.fc_dropout_list = fc_dropout_list
        self.output = nn.Linear(self.fc_layer_size_list[len(self.fc_layer_size_list)-1], n_output)
                
        # Initializing dropouts on each hidden layer
        self.fc_dropouts = [nn.Dropout(p = fc_dropout_list[idx]) for idx in range(len(fc_dropout_list))]
        
        if len(self.fc_dropouts) > 0: # If employing dropout..
            for idx, fc_dropout in enumerate(self.fc_dropouts):
                self.add_module('dropout_%d' % idx, fc_dropout)                 
               
        # Initializing fully-connected layers layers
        self.fc_layer_size_list.insert(0, n_input) # Inserting num_inputs to serve as input dim for first hidden layer                 
        
        self.fc_layers = [nn.Linear(self.fc_layer_size_list[idx], self.fc_layer_size_list[idx+1]) for idx in range(len(self.fc_layer_size_list) - 1)]
        for idx, fc_layer in enumerate(self.fc_layers):
            self.add_module('fc_%d' % idx, fc_layer)
            
            # NEED TO INVESTIGATE
            ff_layer = getattr(self, 'fc_{}'.format(idx)) # ~ self.fc_i
            xavier_uniform(ff_layer.weight)
            
        # Setting non-linear activation on fc layers
        self.fc_activation = getattr(F, fc_activation) # Equivalent to F.[fc_activation]
            
    def forward(self, x):
        
        for i in range(len(self.fc_layers)): # Passing input through each fully connected layer
            
            fc_layer = getattr(self, 'fc_{}'.format(i)) # ~ self.fc_i
            
            if i <= (len(self.fc_dropout_list) - 1) : # If dropout applied to this layer (assuming first value in fc_dropout_list is for first layer)
                
                dropout = getattr(self, 'dropout_{}'.format(i)) # ~ self.dropout_i
                x = dropout(self.fc_activation(fc_layer(x)))
            else:
                x = self.fc_activation(fc_layer(x))

        x = self.output(x) # Prediction
                                       
        return x


def train_and_test_net(X, y, fc_network, criterion, optimizer, num_epochs, batchsize, train_frac, test_frac, gpu_bool):

    # Converting to csr sparse matrix form
    X = X.tocsr()
    
    # Splitting into train, val and test
    X_train = X[:-5000]
    X_val = X[-5000:-2500]
    X_test = X[-2500:]
        
    y_train = y[:-5000]
    y_val = y[-5000:-2500]
    y_test = y[-2500:]
    
    # Standardizing features
    scaler = MaxAbsScaler().fit(X_train)
    X_train_std = scaler.transform(X_train)
    X_val_std = scaler.transform(X_val)
    X_test_std = scaler.transform(X_test)
        
    metrics_dict = defaultdict(lambda : defaultdict(float))

    ### Training Set ###
    
    print("starting training")
    for epoch in range(num_epochs):  # loop over the dataset multiple times
    
        running_loss = 0.0 # Loss over a set of mini batches
        
        i = 0
        k = 0
        losses = []
        
        print("Number of training examples: " + str(int(X_train.shape[0]*train_frac)))
        
        while i < X_train.shape[0]*train_frac:
            
            batch_size_safe = min(batchsize, X_train.shape[0] - i) # Avoiding going out of range
            
            Xtrainsample = X_train_std[i:i+batch_size_safe].todense()
            ytrainsample = y_train[i:i+batch_size_safe]
    
            inputs = torch.from_numpy(Xtrainsample.astype('float32'))
            labels = torch.from_numpy(ytrainsample.astype('float32')).view(-1,1)           
    
            # wrap them in Variable
            inputs, labels = Variable(inputs), Variable(labels)
            
            if gpu_bool:
                inputs = inputs.cuda()
                labels = labels.cuda()
    
            # zero the parameter gradients
            optimizer.zero_grad()
    
            # forward
            outputs = fc_network(inputs)
            loss = criterion(outputs, labels)
            
            # backward
            loss.backward()
            losses.append(loss.data)
            
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
        
        metrics_dict["tr_loss_{}".format(epoch)] = np.mean(losses)
            
        print('Finished Epoch')
        print("Predicting on validation set")
        
        ### Validation Set ###
        
        y_true = []
        y_pred = []
        losses = []
        
        i = 0
        
        while i < X_val_std.shape[0]*test_frac:
            
            batch_size_safe = min(batchsize, X_val_std.shape[0] - i) # Avoiding going out of range
            
            Xtestsample = X_val_std[i:i+batch_size_safe].todense()
            ytestsample = y_val[i:i+batch_size_safe]
        
            inputs = torch.from_numpy(Xtestsample.astype('float32'))
            labels = torch.from_numpy(ytestsample.astype('float32')).view(-1,1)
            
            if gpu_bool:
                inputs = inputs.cuda()
                labels = labels.cuda()
            
            outputs = fc_network(Variable(inputs))
                        
            loss = criterion(outputs, Variable(labels))
            losses.append(loss.data)
            
            outputs = F.sigmoid(outputs)
            
            

            # Converting to numpy format
            outputs = outputs.data.cpu().numpy()

            y_true.extend(labels.numpy().flatten().tolist())
            y_pred.extend(outputs.flatten().tolist())
            i = i + batchsize
            
        print("finished predicting on validation set")
                
        y_pred = np.array(y_pred)
        y_true = np.array(y_true)
        
        epoch_key = "epoch_{}".format(epoch+1)
        
        metrics_dict["val_loss_{}".format(epoch)] = np.mean(losses)
        
        metrics_dict[epoch_key]["f1"], metrics_dict[epoch_key]["opt_thresh"] = evaluation.find_opt_thresh_f1(y_pred, y_true, 0.01, 0.5, 50)
        metrics_dict[epoch_key]["auc"] = evaluation.auc_metrics(y_pred, y_true, metrics_dict[epoch_key]["opt_thresh"]) # auc_metrics() returns a dictionary    
        print("AUC on epoch {}: ".format(epoch+1) + str(metrics_dict[epoch_key]["auc"]))
        print("F1 on epoch {}: ".format(epoch+1) + str(metrics_dict[epoch_key]["f1"]))
        print("opt f1 thresh on epoch {}: ".format(epoch+1) + str(metrics_dict[epoch_key]["opt_thresh"]))
           
        ### Test Set ###
        
        y_true = []
        y_pred = []
        i = 0
    
        while i < X_test_std.shape[0]*test_frac:
        
            batch_size_safe = min(batchsize, X_test_std.shape[0] - i) # Avoiding going out of range
            
            Xtestsample = X_test_std[i:i+batch_size_safe].todense()
            ytestsample = y_test[i:i+batch_size_safe]
        
            inputs = torch.from_numpy(Xtestsample.astype('float32'))
            labels = torch.from_numpy(ytestsample.astype('float32')).view(-1,1)
            
            if gpu_bool:
                inputs = inputs.cuda()
            
            outputs = fc_network(Variable(inputs))
            outputs = F.sigmoid(outputs)
            
            # Converting to numpy format
            outputs = outputs.data.cpu().numpy()
            
            y_true.extend(labels.numpy().flatten().tolist())
            y_pred.extend(outputs.flatten().tolist())
            i = i + batchsize
        
        print("\nfinished predicting on test set")
                
        y_pred = np.array(y_pred)
        y_true = np.array(y_true)
        
        epoch_key = "epoch_{}".format(epoch+1)
        
        metrics_dict[epoch_key]["test_f1"], metrics_dict[epoch_key]["test_opt_thresh"] = evaluation.find_opt_thresh_f1(y_pred, y_true, 0.01, 0.5, 50)
        metrics_dict[epoch_key]["test_auc"] = evaluation.auc_metrics(y_pred, y_true, metrics_dict[epoch_key]["test_opt_thresh"]) # auc_metrics() returns a dictionary    
        print("test AUC on epoch {}: ".format(epoch+1) + str(metrics_dict[epoch_key]["test_auc"]))
        print("test F1 on epoch {}: ".format(epoch+1) + str(metrics_dict[epoch_key]["test_f1"]))
        print("test opt f1 thresh on epoch {}: ".format(epoch+1) + str(metrics_dict[epoch_key]["test_opt_thresh"]))
                
    return metrics_dict


def main(data_path, fc_layer_size_list, fc_dropout_list, fc_activation, num_epochs, batch_size, train_frac, test_frac, gpu_bool):
    
    # Loading data
    X, y = load_svmlight_file(data_path)
    print("data loaded")
        
    net = FeedForwardNet(n_input=X.shape[1], n_output=1, fc_layer_size_list=fc_layer_size_list, fc_dropout_list=fc_dropout_list, fc_activation = fc_activation)
    print("defined network")
        
    print("\nGPU: " + str(gpu_bool))
    
    if gpu_bool:
        net.cuda()
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(net.parameters(), weight_decay=0, lr=0.001)

    metrics_dict = train_and_test_net(X, y, net, criterion, optimizer, num_epochs, batch_size, train_frac, test_frac, gpu_bool)

    return  metrics_dict


if __name__ == "__main__":
    
    current_time = datetime.datetime.now()
    month = str(current_time).split('-')[1]
    day = str(current_time).split('-')[2][:-10].split()[0]
    mins = str(current_time).split('-')[2][:-10].split()[1]
    date =  month + '_' + day + "_" + mins
    
    running_ide = False
    
    if running_ide == True:
        
        data_path = "/home/miller/Documents/BDH NLP/Data/"
#
        auc_dict, f1_dict = main(data_path, [1024], [0.6] , "relu", 1, 32)
        
    else:
                            
        parser = argparse.ArgumentParser(description="train a neural network on some clinical documents")
        parser.add_argument("data_path", type=str,
                            help="path to a file containing sorted train data. dev/test splits assumed to have same name format with 'train' replaced by 'dev' and 'test'")
        parser.add_argument("write_path", type=str,
                            help="path to write file to")
        parser.add_argument("num_epochs", type=int, help="number of epochs to train")
        parser.add_argument("--train-frac", type=float, help="fraction of training split to train on", required=False, dest="train_frac", default=1.0)
        parser.add_argument("--test-frac", type=float, help="fraction of test split to test on", required=False, dest="test_frac", default=1.0)
#        parser.add_argument("--bce-weights", type=str, required=False, dest="bce_weights", default = None,
#                            help="Weights applied to negative and positive classes respectively for Binary Cross entropy loss. Ex: 0.1, 1 --> 10x more weight to positive instances")
        parser.add_argument("--fc-layer-size-list", type=str, required=False, dest="fc_layer_size_list", default=3,
                            help="Number of units in each hidden layer Ex: 3,4,5)")
        parser.add_argument("--fc-activation", type=str, required=False, dest="fc_activation", default="selu",
                            help="non-linear activation to be applied to fc layers. Must match PyTorch documentation for torch.nn.functional.[conv_activation]")
        parser.add_argument("--dropout-list", type=str, required=False, dest="dropout_list", default=None,
                            help=" Dropout proportion on each hidden layer. First number assumed to correspond to first hidden layer. Ex: 0.5,0.1")
        parser.add_argument("--batch-size", type=int, required=False, dest="batch_size", default=32,
                            help="size of training batches")
        parser.add_argument("--patience", type=int, default=2, required=False,
                            help="how many epochs to wait for improved criterion metric before early stopping (default: 3)")
        parser.add_argument("--gpu", dest="gpu", action="store_const", required=False, const=True,
                            help="optional flag to use GPU if available")
        args = parser.parse_args()
        command = ' '.join(['python'] + sys.argv)
        args.command = command
        
        if args.dropout_list:
            dropouts = [float(size) for size in args.dropout_list.split(",")]
        else:
            dropouts = []
                
        fc_layer_sizes = [int(size) for size in args.fc_layer_size_list.split(",")]
        
        print(fc_layer_sizes)
        
        metrics_dict = main(args.data_path, fc_layer_sizes, dropouts, args.fc_activation, args.num_epochs, args.batch_size, args.train_frac, args.test_frac, args.gpu)
                                    
        params = defaultdict(list)
        params["dropout"] = dropouts
        params["hidden_layers"] = fc_layer_sizes[1:] # Number of inputs gets inserted at 0th index in main function
              
        metrics_dict.update(params)
                          
        print(metrics_dict)
            
        with open(args.write_path + "fc_results_{}.txt".format(date), "w") as text_file:
            text_file.write(str(metrics_dict))
          


        