python Structured/struc_net_vM3.py ../../../Data/struc_data.svmlight ../../../Models/ 3 --gpu --fc-layer-size-list 128,32 --fc-activation selu --dropout-list 0.5 --batch-size 16 --train-frac 0.01
python Structured/struc_net_vM3.py ../../../Data/struc_data.svmlight ../../../Models/ 3 --gpu --fc-layer-size-list 128 --fc-activation selu --batch-size 16
python Structured/struc_net_vM3.py ../../../Data/struc_data.svmlight ../../../Models/ 3 --gpu --fc-layer-size-list 64 --fc-activation selu --batch-size 32
python Structured/struc_net_vM3.py ../../../Data/struc_data.svmlight ../../../Models/ 3 --gpu --fc-layer-size-list 128 --fc-activation selu --dropout-list 0.2 --batch-size 32
python Structured/struc_net_vM3.py ../../../Data/struc_data.svmlight ../../../Models/ 3 --gpu --fc-layer-size-list 256,32 --fc-activation relu --dropout-list 0.3 --batch-size 32
python Structured/struc_net_vM3.py ../../../Data/struc_data.svmlight ../../../Models/ 3 --gpu --fc-layer-size-list 256,64 --fc-activation selu --dropout-list 0.4 --batch-size 32
python Structured/struc_net_vM3.py ../../../Data/struc_data.svmlight ../../../Models/ 3 --gpu --fc-layer-size-list 512,32 --fc-activation relu --dropout-list 0.5 --batch-size 32
python Structured/struc_net_vM3.py ../../../Data/struc_data.svmlight ../../../Models/ 3 --gpu --fc-layer-size-list 128,16 --fc-activation relu --dropout-list 0.3 --batch-size 32
