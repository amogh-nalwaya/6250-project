LEN_TRAIN='wc -l ../../../Data/sorted_sums_matched_struc_train_reversed.csv'
len_train=`$LEN_TRAIN | cut -f1 -d' '`
len_train_final=`expr $len_train - 1`

LEN_VAL='wc -l ../../../Data/sorted_sums_matched_struc_val.csv'
len_val=`$LEN_VAL | cut -f1 -d' '`
len_val_final=`expr $len_val - 1`

LEN_TEST='wc -l ../../../Data/sorted_sums_matched_struc_test.csv'
len_test=`$LEN_TEST | cut -f1 -d' '`
len_test_final=`expr $len_test - 1`

python training/training_mmnet_vM1.py ../../../Data/sorted_sums_matched_struc_train.csv ../../../Data/struc_data_reversed.svmlight ../../../Data/vocab.csv mmnet 5 $len_train_final $len_val_final $len_test_final --train-frac 0.01 --test-frac 0.01 --struc-aux-loss-wt 0.3 --conv-aux-loss-wt 0.3 --post-merge-layer-size-list 8 --embed-dropout-bool True --embed-dropout-p 0.1 --batch-size 4 --kernel-sizes 2,3,5 --num-filter-maps 150 --fc-dropout-p 0.5 --embed-file ../../../Data/processed_full.embed --struc-layer-size-list 128,32 --struc-activation selu --struc-dropout-list 0.5 --post-conv-fc-bool True --batch-norm-bool True
