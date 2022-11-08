CFMTL：clustered federated multi-task learning

crypten：MPC computation open-source framework

# secure MNIST experiments
python main.py --dataset mnist --iid iid --ep 50 --local_ep 1 --frac 0.2 --num_batch 10 --num_clients 200 --clust 20 --L 1.0 --experiment performance-mnist

python main.py --dataset mnist --iid non-iid --ratio 0.25 --ep 50 --local_ep 1 --frac 0.2 --num_batch 10 --num_clients 200 --clust 20 --L 0.1 --experiment performance-mnist

python main.py --dataset mnist --iid non-iid --ratio 0.5 --ep 50 --local_ep 1 --frac 0.2 --num_batch 10 --num_clients 200 --clust 20 --L 0.1 --experiment performance-mnist

python main.py --dataset mnist --iid non-iid --ratio 0.75 --ep 50 --local_ep 1 --frac 0.2 --num_batch 10 --num_clients 200 --clust 20 --L 0.001 --experiment performance-mnist

python main.py --dataset mnist --iid non-iid-single_class --ep 50 --local_ep 1 --frac 0.2 --num_batch 10 --num_clients 200 --clust 20 --L 0.001 --experiment performance-mnist

# secure CIFAR experiments
python main.py --dataset cifar --iid iid --ep 100 --local_ep 1 --frac 0.2 --num_batch 10 --num_clients 50 --clust 10 --L 1.0 --experiment performance-cifar

python main.py --dataset cifar --iid non-iid --ratio 0.25 --ep 100 --local_ep 1 --frac 0.2 --num_batch 10 --num_clients 50 --clust 10 --L 1.0 --experiment performance-cifar

python main.py --dataset cifar --iid non-iid --ratio 0.5 --ep 100 --local_ep 1 --frac 0.2 --num_batch 10 --num_clients 50 --clust 10 --L 0.1 --experiment performance-cifar

python main.py --dataset cifar --iid non-iid --ratio 0.75 --ep 100 --local_ep 1 --frac 0.2 --num_batch 10 --num_clients 50 --clust 10 --L 0.1 --experiment performance-cifar

python main.py --dataset cifar --iid non-iid-single_class --ep 100 --local_ep 1 --frac 0.2 --num_batch 10 --num_clients 50 --clust 10 --L 0.1 --experiment performance-cifar



