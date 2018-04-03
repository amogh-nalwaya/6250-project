python training/training_vM5.py ../../../Data/disch_sum_train.csv ../../../Data/vocab.csv conv_encoder 3 --batch-size 50 --kernel-sizes 3,5 --num-filter-maps 50 --dropout 0.5 --embed-file ../../../Data/processed_full.embed
python training/training_vM5.py ../../../Data/disch_sum_train.csv ../../../Data/vocab.csv conv_encoder 15 --batch-size 25 --kernel-sizes 3,5 --num-filter-maps 100 --dropout 0.5 --embed-file ../../../Data/processed_full.embed

