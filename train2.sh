python training/training_vM6.py ../../../Data/train.csv ../../../Data/vocab.csv conv_encoder 5 --loss-weights 1,10 --embed-dropout-bool True   --batch-size 25 --kernel-sizes 2,3,5 --num-filter-maps 100 --fc-dropout-p 0.5 --embed-file ../../../Data/processed_full.embed
python training/training_vM5.py ../../../Data/disch_sum_train.csv ../../../Data/vocab.csv conv_encoder 5 --batch-size 25 --kernel-sizes 3,5 --num-filter-maps 100 --dropout 0.5 --embed-file ../../../Data/processed_full.embed

