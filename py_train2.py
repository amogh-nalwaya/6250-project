import subprocess
import shlex

var_list = ["./train_v2.sh",                   # Shell script
            "training/training_vM7.py",        # Training script
            "../../../Data/dis_sum_train.csv", # Data path
            "../../../Data/vocab.csv",         # Vocab path
            "conv_encoder",                    # Model
            "5",                               # Num epochs
            "--desc",                          # Description tag
            "new_vocab_02_1_wts",
            "--embed-dropout-bool",
            "True",
            "--embed-dropout-p",
            "0.3",
            "--bce-weights",
            "0.2,1",
            "--batch-size",
            "50",
            "--kernel-sizes",
            "3,5",
            "--num-filter-maps",
            "100",
            "--fc-dropout-p",
            "0.5",
            "--embed-file",
            "../../../Data/processed_full.embed"]

string = ""

for idx in range(len(var_list)):
        
    if idx < (len(var_list) - 1):
            
        string = string + var_list[idx] + ' '
    
    else:
        
        string += var_list[idx]
        
        
subprocess.call(shlex.split(string))