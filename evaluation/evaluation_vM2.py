"""
    This file contains evaluation methods that take in a set of predicted labels 
        and a set of ground truth labels and calculate precision, recall, accuracy, and f1 score
"""
from collections import defaultdict
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix
from constants import *

def all_metrics(yhat, y, yhat_raw=None):
    """
        Inputs:
            yhat: binary predictions matrix 
            y: binary ground truth matrix
            yhat_raw: prediction scores matrix (floats)
        Outputs:
            dict holding relevant metrics
    """
    names = ["acc", "prec", "rec", "f1"]

    #micro
    ymic = y.ravel() # RAVEL = FLATTEN TO 1D
    yhatmic = yhat.ravel()
    micro = all_micro(yhatmic, ymic)

    metrics = {names[i] + "_micro": micro[i] for i in range(len(micro))}

    #AUC
    if yhat_raw is not None:
        
        roc_auc = auc_metrics(yhat_raw, ymic)
        metrics.update(roc_auc)
        
    return metrics

def all_micro(yhatmic, ymic):
    return micro_accuracy(yhatmic, ymic), micro_precision(yhatmic, ymic), micro_recall(yhatmic, ymic), micro_f1(yhatmic, ymic)

###################
# INSTANCE-AVERAGED
###################

#def inst_precision(yhat, y):
#    num = intersect_size(yhat, y, 1) / yhat.sum(axis=1)
#    #correct for divide-by-zeros
#    num[np.isnan(num)] = 0.
#    return np.mean(num)
#
#def inst_recall(yhat, y):
#    num = intersect_size(yhat, y, 1) / y.sum(axis=1)
#    #correct for divide-by-zeros
#    num[np.isnan(num)] = 0.
#    return np.mean(num)
#
#def inst_f1(yhat, y):
#    prec = inst_precision(yhat, y)
#    rec = inst_recall(yhat, y)
#    f1 = 2*(prec*rec)/(prec+rec)
#    return f1

##########################################################################
#MICRO METRICS: treat every prediction as an individual binary prediction
##########################################################################

def micro_accuracy(yhatmic, ymic):
    return np.sum(yhatmic == ymic) / ymic.shape[0]

def micro_precision(yhatmic, ymic):
    if np.sum(yhatmic) == 0:
        print("Predicted all 0s, precision is null (0/0+0)")
        return 0
    else:
        return intersect_size(yhatmic, ymic, 0) / yhatmic.sum(axis=0)

def micro_recall(yhatmic, ymic):
    return intersect_size(yhatmic, ymic, 0) / ymic.sum(axis=0)

def micro_f1(yhatmic, ymic):
    prec = micro_precision(yhatmic, ymic)
    rec = micro_recall(yhatmic, ymic)
    if prec + rec == 0:
        f1 = 0.
    else:
        f1 = 2*(prec*rec)/(prec+rec)
    return f1

def auc_metrics(yhat_raw, ymic):
    if yhat_raw.shape[0] <= 1:
        return
    
    roc_auc = {}
    
    #micro-AUC
    yhatmic = yhat_raw.ravel()
    roc_auc['true_neg'], roc_auc['false_pos'], roc_auc['false_neg'], roc_auc['true_pos'] = confusion_matrix(ymic, np.round(yhatmic)).ravel() # Rounding to get binary preds
    
    # Converting to float to allow serialization to json
    roc_auc['true_neg'] = np.float(roc_auc['true_neg'])
    roc_auc['false_neg'] = np.float(roc_auc['false_neg'])  
    roc_auc['true_pos'] = np.float(roc_auc['true_pos'])  
    roc_auc['false_pos'] = np.float(roc_auc['false_pos'])         
           
    roc_auc["auc"] = roc_auc_score(ymic, yhatmic) # (groundTruth, Preds)

    return roc_auc

def union_size(yhat, y, axis):
    #axis=0 for label-level union (macro). axis=1 for instance-level
    return np.logical_or(yhat, y).sum(axis=axis).astype(float)

def intersect_size(yhat, y, axis):
    #axis=0 for label-level union (macro). axis=1 for instance-level
    return np.logical_and(yhat, y).sum(axis=axis).astype(float)

def print_metrics(metrics):
    #annoyingly complicated printing, to keep track of progress during training
        
    if "true_pos" in metrics.keys():
        print("\naccuracy, prec, recall, f-measure, AUC, True Pos, False Pos, True Neg, False Neg")
        print("%.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f" % (metrics["acc_micro"], metrics["prec_micro"], metrics["rec_micro"], metrics["f1_micro"], metrics["auc"], metrics["true_pos"], metrics["false_pos"],metrics["true_neg"], metrics["false_neg"]))
          
    elif "auc" in metrics.keys():
        print("\naccuracy, precision, recall, f-measure, AUC")
        print("%.4f, %.4f, %.4f, %.4f, %.4f" % (metrics["acc_micro"], metrics["prec_micro"], metrics["rec_micro"], metrics["f1_micro"], metrics["auc"]))
    else:
        print("[MICRO] accuracy, precision, recall, f-measure")
        print("%.4f, %.4f, %.4f, %.4f" % (metrics["acc_micro"], metrics["prec_micro"], metrics["rec_micro"], metrics["f1_micro"]))

