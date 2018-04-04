"""
    Main training code. Loads data, builds the model, trains, tests, evaluates, writes outputs, etc.
"""
import torch
import torch.optim as optim
from torch.autograd import Variable

import csv
import argparse
import os 
import numpy as np
import sys
import time
from tqdm import tqdm
from collections import defaultdict

# Adding relative path to python path
abs_path = os.path.abspath(__file__)
file_dir = os.path.dirname(abs_path)
parent_dir = os.path.dirname(file_dir)
sys.path.append(parent_dir)
#sys.path.append("/home/miller/Documents/BDH NLP/Code/Github/6250-project/")

from constants import *
from datasets import datasets_vM2 as datasets
from evaluation import evaluation_vM2 as evaluation
from persistence import persistence_vM1 as persistence
from tools import tools_vM1 as tools

def main(args):
    start = time.time()
    args, model, optimizer, params, dicts = init(args)
    epochs_trained = train_epochs(args, model, optimizer, params, dicts)
    print("TOTAL ELAPSED TIME FOR %s MODEL AND %d EPOCHS: %f" % (args.model, epochs_trained, time.time() - start))

def init(args):
    """
        Load data, build model, create optimizer, create vars to hold metrics, etc.
    """
    #need to handle really large text fields
    csv.field_size_limit(sys.maxsize) # Sets field size to max available for strings

    # LOAD VOCAB DICTS
    dicts = datasets.load_vocab_dict(args.vocab_path)

    model = tools.pick_model(args, dicts)
    print(model)
    
    print("\nGPU: " + str(args.gpu))

    optimizer = optim.Adam(model.params_to_optimize(), weight_decay=args.weight_decay, lr=args.lr)

    params = tools.make_param_dict(args)
    
    return args, model, optimizer, params, dicts

def train_epochs(args, model, optimizer, params, dicts):
    """
        Main loop. does train and test
    """
    metrics_hist = defaultdict(lambda: [])
    metrics_hist_te = defaultdict(lambda: [])
    metrics_hist_tr = defaultdict(lambda: [])

    test_only = args.test_model is not None
    
    print("\n\ntest_only: " + str(test_only))
        
    #train for n_epochs unless criterion metric does not improve for [patience] epochs
    for epoch in range(args.n_epochs):
        
        #only test on train/test set on very last epoch
        if epoch == 0 and not args.test_model:
            model_dir = os.path.join(MODEL_DIR, '_'.join([args.model, time.strftime('%b_%d_%H:%M', time.gmtime())]))
            os.mkdir(model_dir) 
            
        elif args.test_model:
            
            model_dir = os.getcwd() #just save things to where this script was called
            
        metrics_all = one_epoch(model, optimizer, epoch, args.n_epochs, args.batch_size, args.data_path,test_only, dicts, model_dir, 
                                                  args.samples, args.gpu, args.debug, args.quiet)

        # DISTRIBUTING results from metrics_all to respective dicts
        for name in metrics_all[0].keys():
            metrics_hist[name].append(metrics_all[0][name])
        for name in metrics_all[1].keys():
            metrics_hist_te[name].append(metrics_all[1][name])
        for name in metrics_all[2].keys():
            metrics_hist_tr[name].append(metrics_all[2][name])
        metrics_hist_all = (metrics_hist, metrics_hist_te, metrics_hist_tr)

        #save metrics, model, params
        persistence.save_everything(args, metrics_hist_all, model, model_dir, params, args.criterion) # SHOULD SAVE MODEL PARAMS AT EACH EPOCH, BELIEVE IS HAPPENING

        if test_only:
            break

        if args.criterion in metrics_hist.keys():
            if early_stop(metrics_hist, args.criterion, args.patience):
                #stop training, do tests on test and train sets, and then stop the script
                print("%s hasn't improved in %d epochs, early stopping..." % (args.criterion, args.patience))
                test_only = True
                model = torch.load('%s/model_best_%s.pth' % (model_dir, args.criterion)) # LOADING BEST MODEL FOR FINAL TEST
                
    return epoch+1

def early_stop(metrics_hist, criterion, patience):
    if not np.all(np.isnan(metrics_hist[criterion])):
        if criterion == 'loss-dev': 
            
            ### EPOCH NUM W/ MIN DEV LOSS < (CURR EPOCH NUM - PATIENCE) ?? RETURNS ANSWER AS BOOL --> IF TRUE, STOP TRAINING
            ### EX: 5 < 9 - 3 = TRUE = EARLY STOP
            return np.nanargmin(metrics_hist[criterion]) < len(metrics_hist[criterion]) - patience
        else:
            
            ### EPOCH NUM W/ MAX CRITERION VAL < (CURR EPOCH NUM - PATIENCE) ?? RETURNS ANSWER AS BOOL
            return np.nanargmax(metrics_hist[criterion]) < len(metrics_hist[criterion]) - patience
    else:
        return False
        
def one_epoch(model, optimizer, epoch, n_epochs, batch_size, data_path, testing_only, dicts, model_dir, samples, gpu, debug, quiet):
    """
        Basically a wrapper to do a training epoch and test on dev
    """
    if not testing_only:        
        losses = train(model, optimizer, epoch, batch_size, data_path, gpu, dicts, debug, quiet)
        loss = np.mean(losses)
        print("epoch loss: " + str(loss))
        
    else:
        loss = np.nan

    pred_fold = "val" # fold to predict on

    #test on dev --> NEED TO MAKE SURE THIS IS HAPPENING ON EVERY EPOCH, PROBLY IS
    metrics = test(model, epoch, batch_size, data_path, pred_fold, gpu, dicts, samples, model_dir, testing_only, debug)
    
    # REPETITIVE WITH CODE ABOVE
    if testing_only or epoch == n_epochs - 1:
        print("evaluating on test")
        metrics_te = test(model, epoch, batch_size, data_path, "test", gpu, dicts, samples, model_dir, True, debug)

    else:
        metrics_te = defaultdict(float)
#        fpr_te = defaultdict(lambda: []) # PURPOSE?
#        tpr_te = defaultdict(lambda: [])
    metrics_tr = {'loss': loss}
    metrics_all = (metrics, metrics_te, metrics_tr)
    
    return metrics_all

def train(model, optimizer, epoch, batch_size, data_path, gpu, dicts, debug, quiet):
    """
        Training loop.
        output: losses for each example for this iteration
    """
    losses = []
    
    #how often to print some info to stdout
    print_interval = 25

    model.train() # PUTS MODEL IN TRAIN MODE
                   
    gen = datasets.data_generator(data_path, dicts, batch_size)
    for batch_idx, tup in tqdm(enumerate(gen)):
                
        data, target, hadm = tup
                
        data, target = Variable(torch.LongTensor(data)), Variable(torch.FloatTensor(target))
        
        if gpu:
            data = data.cuda()
            target = target.cuda()
            
        optimizer.zero_grad()

        output, loss = model(data, target) # FORWARD PASS

        loss.backward()
        optimizer.step()

        losses.append(loss.data[0])

        if not quiet and batch_idx % print_interval == 0:
            #print the average loss of the last 100 batches
            print("Train epoch: {} [batch #{}, batch_size {}, seq length {}]\tLoss: {:.6f}".format(
                epoch+1, batch_idx, data.size()[0], data.size()[1], np.mean(losses[-100:])))

    return losses


def test(model, epoch, batch_size, data_path, fold, gpu, dicts, samples, model_dir, testing, debug):

    """
        Testing loop.
        Returns metrics
    """
    filename = data_path.replace('train', fold)
    print('\nfile for evaluation: %s' % filename)
    
    #initialize stuff for saving attention samples
    if samples:
        tp_file = open('%s/tp_%s_examples_%d.txt' % (model_dir, fold, epoch), 'w')
        fp_file = open('%s/fp_%s_examples_%d.txt' % (model_dir, fold, epoch), 'w')

    y, yhat, yhat_raw, hids, losses = [], [], [], [], []
    
    model.eval()
    gen = datasets.data_generator(filename, dicts, batch_size)
    for batch_idx, tup in tqdm(enumerate(gen)):
        if debug and batch_idx > 50:
            break
        
        data, target, hadm_ids = tup
        
        data, target = Variable(torch.LongTensor(data), volatile=True), Variable(torch.FloatTensor(target))
        if gpu:
            data = data.cuda()
            target = target.cuda()
            
        model.zero_grad()

        output, loss = model(data, target) # Forward pass

        output = output.data.cpu().numpy()
        losses.append(loss.data[0])
        target_data = target.data.cpu().numpy()
        
        #save predictions, target, hadm ids
        yhat_raw.append(output) 
        output = np.round(output) # Rounds to 0 for <= 0.5, up to one for > 0.5
        yhat.append(output)
        
        y.append(target_data)
        hids.extend(hadm_ids)

    if samples:
        tp_file.close()
        fp_file.close()
    
    y = np.concatenate(y, axis=0)
    yhat = np.concatenate(yhat, axis=0)
    yhat_raw = np.concatenate(yhat_raw, axis=0)
    
    print("\nMax Prediction:")
    print(max(yhat_raw))

#    print("y shape: " + str(y.shape))
#    print("yhat shape: " + str(yhat.shape))

    #write the predictions
    persistence.write_preds(yhat, model_dir, hids, fold, yhat_raw)
        
    metrics = evaluation.all_metrics(yhat, y, yhat_raw=yhat_raw)
    evaluation.print_metrics(metrics)
    metrics['loss_%s' % fold] = np.mean(losses)
    
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train a neural network on some clinical documents")
    parser.add_argument("data_path", type=str,
                        help="path to a file containing sorted train data. dev/test splits assumed to have same name format with 'train' replaced by 'dev' and 'test'")
    parser.add_argument("vocab_path", type=str, help="path to a file holding vocab word list for discretizing words")
    parser.add_argument("model", type=str, choices=["conv_encoder", "rnn", "conv_attn", "multi_conv_attn", "saved"], help="model")
    parser.add_argument("n_epochs", type=int, help="number of epochs to train")
    parser.add_argument("--embed-file", type=str, required=False, dest="embed_file",
                        help="path to a file holding pre-trained embeddings")
    parser.add_argument("--loss-weights", type=str, required=False, dest="loss_weights", default = None,
                        help="Weights associated with neg and pos class for BCE calculation. Ex: 0.1, 1")
    parser.add_argument("--embed-size", type=int, required=False, dest="embed_size", default=100,
                        help="size of embedding dimension. (default: 100)")
    parser.add_argument("--kernel-sizes", type=list, required=False, dest="kernel_sizes", default=3,
                        help="Size(s) of convolutional filter(s)/kernel(s) to use. Ex: 3,4,5)")
    parser.add_argument("--num-filter-maps", type=int, required=False, dest="num_filter_maps", default=50,
                        help="size of conv output (default: 50)")
    parser.add_argument("--conv-activation", type=str, required=False, dest="conv_activation", default="selu",
                        help="non-linear activation to be applied to feature maps. Must match PyTorch documentation for torch.nn.functional.[conv_activation]")
    parser.add_argument("--weight-decay", type=float, required=False, dest="weight_decay", default=0,
                        help="coefficient for penalizing l2 norm of model weights (default: 0)")
    parser.add_argument("--lr", type=float, required=False, dest="lr", default=1e-3,
                        help="learning rate for Adam optimizer (default=1e-3)")
    parser.add_argument("--batch-size", type=int, required=False, dest="batch_size", default=16,
                        help="size of training batches")
    parser.add_argument("--dropout", dest="dropout", type=float, required=False, default=0.5,
                        help="optional specification of dropout (default: 0.5)")
    parser.add_argument("--test-model", type=str, dest="test_model", required=False, help="path to a saved model to load and evaluate")
    parser.add_argument("--criterion", type=str, default='f1_micro', required=False, dest="criterion",
                        help="which metric to use for early stopping (default: f1_micro)")
    parser.add_argument("--patience", type=int, default=3, required=False,
                        help="how many epochs to wait for improved criterion metric before early stopping (default: 3)")
    parser.add_argument("--gpu", dest="gpu", action="store_const", required=False, const=True,
                        help="optional flag to use GPU if available")
    parser.add_argument("--debug", dest="debug", action="store_const", required=False, const=True,
                        help="optional flag to set debug mode (run train/test for only 50 batches)")
    parser.add_argument("--stack-filters", dest="stack_filters", action="store_const", required=False, const=True,
                        help="optional flag for multi_conv_attn to instead use concatenated filter outputs, rather than pooling over them")
    parser.add_argument("--samples", dest="samples", action="store_const", required=False, const=True,
                        help="optional flag to save samples of good / bad predictions")
    parser.add_argument("--quiet", dest="quiet", action="store_const", required=False, const=True,
                        help="optional flag not to print so much during training")
    args = parser.parse_args()
    command = ' '.join(['python'] + sys.argv)
    args.command = command
    main(args)

