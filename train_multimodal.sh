LEN_TRAIN='wc -l ../../../Data/sorted_sums_matched_w_struc_train.csv'
len_train=`$LEN_TRAIN | cut -f1 -d' '`
len_train_final=`expr $len_train - 1`

LEN_VAL='wc -l ../../../Data/sorted_sums_matched_w_struc_val.csv'
len_val=`$LEN_VAL | cut -f1 -d' '`
len_val_final=`expr $len_val - 1`

LEN_TEST='wc -l ../../../Data/sorted_sums_matched_w_struc_test.csv'
len_test=`$LEN_TEST | cut -f1 -d' '`
len_test_final=`expr $len_test - 1`

python training/training_multimodal_vM1.py ../../../Data/sorted_sums_matched_w_struc_train.csv ../../../Data/struc_data.svmlight ../../../Data/vocab.csv mmnet 5 $len_train_final $len_val_final $len_test_final --train-frac 0.1 --post-merge-layer-size-list 4 --embed-dropout-bool True --embed-dropout-p 0.3 --batch-size 32 --kernel-sizes 3,5 --num-filter-maps 100 --fc-dropout-p 0.5 --embed-file ../../../Data/processed_full.embed --struc-layer-size-list 256,32 --struc-activation relu --struc-dropout-list 0.3
