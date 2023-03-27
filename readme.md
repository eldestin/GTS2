## This Repository is the GTS2 Source code

### Data 

Two different datasets we used in paper are in data files. Notice that if you want to change the dataset, please go into train.py and changes **wiki = False**

### Layer

All the laye used in GTS2 can be checked in Layer files.

### Train

Used python3 train.py to run the training strategy of GTS2.

### Requirement

All the experiments are runned in:

1. torch V1.12.1+cu116
2. torch-scatter V2.0.9

with a single 24G 3090 GPU.
