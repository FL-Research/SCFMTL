# Clustered Federated Multi-Task Learning with Non-IID Data

## Environment

python 3.9.1

pytorch 1.7.1+cu101

## Run Experiments and Get Figures

### Performance on MNIST

python main.py --dataset mnist --iid iid --ep 50 --local_ep 1 --frac 0.2 --num_batch 10 --num_clients 60 --clust 10 --L 1.0 --experiment performance-mnist --filename fig

python main.py --dataset mnist --iid non-iid --ratio 0.25 --ep 50 --local_ep 1 --frac 0.2 --num_batch 10 --num_clients 60 --clust 10 --L 0.1 --experiment performance-mnist --filename fig

python main.py --dataset mnist --iid non-iid --ratio 0.5 --ep 50 --local_ep 1 --frac 0.2 --num_batch 10 --num_clients 60 --clust 10 --L 0.1 --experiment performance-mnist --filename fig

python main.py --dataset mnist --iid non-iid --ratio 0.75 --ep 50 --local_ep 1 --frac 0.2 --num_batch 10 --num_clients 60 --clust 10 --L 0.001 --experiment performance-mnist --filename fig

python main.py --dataset mnist --iid non-iid-single_class --ep 50 --local_ep 1 --frac 0.2 --num_batch 10 --num_clients 60 --clust 10 --L 0.001 --experiment performance-mnist --filename fig

python fig.py --experiment performance-mnist --filename fig

### Performance on CIFAR-10

python main.py --dataset cifar --iid iid --ep 200 --local_ep 1 --frac 0.2 --num_batch 10 --num_clients 50 --clust 10 --L 1.0 --experiment performance-cifar --filename fig

python main.py --dataset cifar --iid non-iid --ratio 0.25 --ep 200 --local_ep 1 --frac 0.2 --num_batch 10 --num_clients 50 --clust 10 --L 1.0 --experiment performance-cifar --filename fig

python main.py --dataset cifar --iid non-iid --ratio 0.5 --ep 200 --local_ep 1 --frac 0.2 --num_batch 10 --num_clients 50 --clust 10 --L 0.1 --experiment performance-cifar --filename fig

python main.py --dataset cifar --iid non-iid --ratio 0.75 --ep 200 --local_ep 1 --frac 0.2 --num_batch 10 --num_clients 50 --clust 10 --L 0.1 --experiment performance-cifar --filename fig

python main.py --dataset cifar --iid non-iid-single_class --ep 200 --local_ep 1 --frac 0.2 --num_batch 10 --num_clients 50 --clust 10 --L 0.1 --experiment performance-cifar --filename fig

### performance on caltech101 with running in the background

<!-- ratio:0.25; L:1;  -->
nohup python main.py --dataset caltech101 --iid non-iid --ratio 0.25 --ep 100 --local_ep 1 --frac 0.2 --num_batch 64 --num_clients 10 --clust 10 --L 1 --experiment performance-caltech101  --filename caltech101_CFMTL_0.25 >> log/caltech101_CFMTL_0.25.log 2>&1 &
<!-- ratio:0.5; L:1;  -->
nohup python main.py --dataset caltech101 --iid non-iid --ratio 0.5 --ep 100 --local_ep 1 --frac 0.2 --num_batch 64 --num_clients 10 --clust 10 --L 1 --experiment performance-caltech101  --filename caltech101_CFMTL_0.50 >> log/caltech101_CFMTL_0.50.log 2>&1 &
<!-- ratio:0.75; L:1;  dist:L2-->
nohup python main.py --dataset caltech101 --iid non-iid --ratio 0.75 --ep 100 --local_ep 1 --frac 0.2 --num_batch 64 --num_clients 10 --clust 10 --L 1 --experiment performance-caltech101  --filename caltech101_CFMTL_0.75_1 >> log/caltech101_CFMTL_0.75.log 2>&1 &
<!-- non-iid:non-iid-single_class; L:1; -->
nohup python main.py --dataset caltech101 --iid non-iid-single_class --ep 100 --local_ep 1 --frac 0.2 --num_batch 64 --num_clients 10 --clust 10 --L 1 --experiment performance-caltech101  --filename caltech101_CFMTL_sc >> log/caltech101_CFMTL_sc.log 2>&1 &
<!-- ratio:0.75; L:1; dist:Equal -->
<!-- 检测程序所耗费时间 -->
python -m cProfile -s time main.py --dataset caltech101 --iid non-iid --ratio 0.75 --ep 2 --local_ep 1 --frac 0.2 --num_batch 64 --num_clients 10 --clust 10 --L 1 --dist="Equal" --experiment performance-caltech101  --filename caltech101_CFMTL_0.75_equal

nohup python main.py --dataset caltech101 --iid non-iid --ratio 0.75 --ep 100 --local_ep 1 --frac 0.2 --num_batch 64 --num_clients 10 --clust 10 --L 1 --dist="Equal" --experiment performance-caltech101  --filename caltech101_CFMTL_0.75_equal >> log/caltech101_CFMTL_0.75_equal.log 2>&1 &

<!-- ratio:0.75; L:1; dist:Equal; clust:5 -->
python main.py --dataset caltech101 --iid non-iid --ratio 0.75 --ep 100 --local_ep 1 --frac 0.2 --num_batch 64 --num_clients 10 --clust 5 --L 1 --dist="Equal" --experiment performance-caltech101  --filename caltech101_CFMTL_0.75_equal_clust5

<!-- iid; L1:0.1 -->
nohup python main.py --dataset caltech101 --iid iid --ep 100 --local_ep 1 --frac 0.2 --num_batch 64 --num_clients 10 --clust 10 --L 0.1 --experiment performance-caltech101  --filename caltech101_CFMTL_iid_L_0.1 >> log/caltech101_CFMTL_iid_L0.1.log 2>&1 &
<!-- non-iid:0.25; L1:0.1 -->
python main.py --dataset caltech101 --iid non-iid --ratio 0.25 --ep 100 --local_ep 1 --frac 0.2 --num_batch 64 --num_clients 10 --clust 10 --L 0.1 --experiment performance-caltech101  --filename caltech101_CFMTL_0.25_L0.1 >> log/caltech101_CFMTL_0.25_L0.1.log 2>&1
<!-- non-iid:0.50; L1:0.1 -->
python main.py --dataset caltech101 --iid non-iid --ratio 0.50 --ep 100 --local_ep 1 --frac 0.2 --num_batch 64 --num_clients 10 --clust 10 --L 0.1 --experiment performance-caltech101  --filename caltech101_CFMTL_0.50_L0.1 >> log/caltech101_CFMTL_0.50_L0.1.log 2>&1
<!-- non-iid:0.75; L1:0.1 -->
python main.py --dataset caltech101 --iid iid --ep 100 --local_ep 1 --frac 0.2 --num_batch 64 --num_clients 10 --clust 10 --L 0.1 --experiment performance-caltech101  --filename caltech101_CFMTL_0.75_L0.1 >> log/caltech101_CFMTL_0.75_L0.1.log 2>&1
<!-- non-iid:non-iid-single_class; L1:0.1 -->
python main.py --dataset caltech101 --iid non-iid-single_class --ep 100 --local_ep 1 --frac 0.2 --num_batch 64 --num_clients 10 --clust 10 --L 0.1 --experiment performance-caltech101  --filename caltech101_CFMTL_sc_L0.1 >> log/caltech101_CFMTL_sc_L0.1.log 2>&1

<!-- iid; L:0.01 -->
nohup python main.py --dataset caltech101 --iid iid --ep 60 --local_ep 1 --frac 0.2 --num_batch 64 --num_clients 10 --clust 10 --L 0.01 --experiment performance-caltech101  --filename caltech101_CFMTL_iid_L_0.01 >> log_simple_net/caltech101_CFMTL_iid_L0.01.log 2>&1 &
<!-- non-iid:0.25; L:0.01 -->
python main.py --dataset caltech101 --iid non-iid --ratio 0.25 --ep 100 --local_ep 1 --frac 0.2 --num_batch 64 --num_clients 10 --clust 10 --L 0.01 --experiment performance-caltech101  --filename caltech101_CFMTL_0.25_L0.01 >> log/caltech101_CFMTL_0.25_L0.01.log 2>&1
<!-- non-iid:0.50; L:0.01 -->
python main.py --dataset caltech101 --iid non-iid --ratio 0.50 --ep 100 --local_ep 1 --frac 0.2 --num_batch 64 --num_clients 10 --clust 10 --L 0.01 --experiment performance-caltech101  --filename caltech101_CFMTL_0.50_L0.01 >> log/caltech101_CFMTL_0.50_L0.01.log 2>&1
<!-- non-iid:0.75; L:0.01 -->
python main.py --dataset caltech101 --iid iid --ep 100 --local_ep 1 --frac 0.2 --num_batch 64 --num_clients 10 --clust 10 --L 0.01 --experiment performance-caltech101  --filename caltech101_CFMTL_0.75_L0.01 >> log/caltech101_CFMTL_0.75_L0.01.log 2>&1
<!-- non-iid:non-iid-single_class; L:0.01 -->
python main.py --dataset caltech101 --iid non-iid-single_class --ep 100 --local_ep 1 --frac 0.2 --num_batch 64 --num_clients 10 --clust 10 --L 0.01 --experiment performance-caltech101  --filename caltech101_CFMTL_sc_L0.01 >> log/caltech101_CFMTL_sc_L0.01.log 2>&1


python fig.py --experiment performance-cifar --filename fig

### Communication cost

python main.py --dataset cifar --iid iid --ep 50 --local_ep 1 --frac 0.2 --num_batch 10 --num_clients 50 --clust 10 --L 1.0 --experiment communication --filename fig

python main.py --dataset cifar --iid non-iid --ratio 0.25 --ep 50 --local_ep 1 --frac 0.2 --num_batch 10 --num_clients 50 --clust 10 --L 1.0 --experiment communication --filename fig

python main.py --dataset cifar --iid non-iid --ratio 0.5 --ep 50 --local_ep 1 --frac 0.2 --num_batch 10 --num_clients 50 --clust 10 --L 0.1 --experiment communication --filename fig

python main.py --dataset cifar --iid non-iid --ratio 0.75 --ep 50 --local_ep 1 --frac 0.2 --num_batch 10 --num_clients 50 --clust 10 --L 0.1 --experiment communication --filename fig

python main.py --dataset cifar --iid non-iid-single_class --ep 50 --local_ep 1 --frac 0.2 --num_batch 10 --num_clients 50 --clust 10 --L 0.1 --experiment communication --filename fig

python fig.py --experiment communication --filename fig

### Various hyperparameters

python main.py --dataset mnist --iid non-iid --ratio 0.5 --ep 50 --local_ep 1 --frac 0.5 --num_batch 10 --num_clients 250 --clust 50 --L 0.1 --experiment hyperparameters --filename fig

python main.py --dataset mnist --iid non-iid --ratio 0.5 --ep 50 --local_ep 1 --frac 1.0 --num_batch 10 --num_clients 250 --clust 50 --L 0.1 --experiment hyperparameters --filename fig

python main.py --dataset mnist --iid non-iid --ratio 0.5 --ep 50 --local_ep 1 --frac 0.2 --num_batch 50 --num_clients 250 --clust 50 --L 0.1 --experiment hyperparameters --filename fig

python main.py --dataset mnist --iid non-iid --ratio 0.5 --ep 50 --local_ep 1 --frac 0.2 --num_batch 1 --num_clients 250 --clust 50 --L 0.1 --experiment hyperparameters --filename fig

python main.py --dataset mnist --iid non-iid --ratio 0.5 --ep 50 --local_ep 5 --frac 0.2 --num_batch 10 --num_clients 250 --clust 50 --L 0.1 --experiment hyperparameters --filename fig

python fig.py --experiment hyperparameters --filename fig

### Different metrics

python main.py --dataset mnist --iid iid --ep 50 --local_ep 1 --frac 0.2 --num_batch 10 --num_clients 250 --clust 50 --L 1.0 --experiment metric --filename fig

python main.py --dataset mnist --iid non-iid --ratio 0.25 --ep 50 --local_ep 1 --frac 0.2 --num_batch 10 --num_clients 250 --clust 50 --L 0.1 --experiment metric --filename fig

python main.py --dataset mnist --iid non-iid --ratio 0.5 --ep 50 --local_ep 1 --frac 0.2 --num_batch 10 --num_clients 250 --clust 50 --L 0.1 --experiment metric --filename fig

python main.py --dataset mnist --iid non-iid --ratio 0.75 --ep 50 --local_ep 1 --frac 0.2 --num_batch 10 --num_clients 250 --clust 50 --L 0.001 --experiment metric --filename fig

python main.py --dataset mnist --iid non-iid-single_class --ep 50 --local_ep 1 --frac 0.2 --num_batch 10 --num_clients 250 --clust 50 --L 0.001 --experiment metric --filename fig

python fig.py --experiment metric --filename fig
