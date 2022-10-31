# 批量后台运行主函数 nohup sh run.sh &
# # <!-- iid; L1:0.1 -->
# # python main.py --dataset caltech101 --iid iid --ep 100 --local_ep 1 --frac 0.2 --num_batch 64 --num_clients 10 --clust 10 --L 0.1 --experiment performance-caltech101  --filename caltech101_CFMTL_iid_L(0.1) >> log/caltech101_CFMTL_iid_L(0.1).log 2>&1
# # <!-- non-iid:0.25; L1:0.1 -->
# # python main.py --dataset caltech101 --iid non-iid --ratio 0.25 --ep 100 --local_ep 1 --frac 0.2 --num_batch 64 --num_clients 10 --clust 10 --L 0.1 --experiment performance-caltech101  --filename caltech101_CFMTL_0.25_L0.1 >> log/caltech101_CFMTL_0.25_L0.1.log 2>&1;
# # # <!-- non-iid:0.50; L1:0.1 -->
# # python main.py --dataset caltech101 --iid non-iid --ratio 0.50 --ep 100 --local_ep 1 --frac 0.2 --num_batch 64 --num_clients 10 --clust 10 --L 0.1 --experiment performance-caltech101  --filename caltech101_CFMTL_0.50_L0.1 >> log/caltech101_CFMTL_0.50_L0.1.log 2>&1;
# # <!-- non-iid:0.75; L1:0.1 -->
# python main.py --dataset caltech101 --iid non-iid --ratio 0.75 --ep 100 --local_ep 1 --frac 0.2 --num_batch 64 --num_clients 10 --clust 10 --L 0.1 --experiment performance-caltech101  --filename caltech101_CFMTL_0.75_L0.1 >> log/caltech101_CFMTL_0.75_L0.1.log 2>&1;
# # <!-- non-iid:non-iid-single_class; L1:0.1 -->
# python main.py --dataset caltech101 --iid non-iid-single_class --ep 100 --local_ep 1 --frac 0.2 --num_batch 64 --num_clients 10 --clust 10 --L 0.1 --experiment performance-caltech101  --filename caltech101_CFMTL_sc_L0.1 >> log/caltech101_CFMTL_sc_L0.1.log 2>&1;
# # <!-- iid; L:0.01 -->
# python main.py --dataset caltech101 --iid iid --ep 100 --local_ep 1 --frac 0.2 --num_batch 64 --num_clients 10 --clust 10 --L 0.01 --experiment performance-caltech101  --filename caltech101_CFMTL_iid_L_0.01 >> log/caltech101_CFMTL_iid_L0.01.log 2>&1;
# # <!-- non-iid:0.25; L:0.01 -->
# python main.py --dataset caltech101 --iid non-iid --ratio 0.25 --ep 100 --local_ep 1 --frac 0.2 --num_batch 64 --num_clients 10 --clust 10 --L 0.01 --experiment performance-caltech101  --filename caltech101_CFMTL_0.25_L0.01 >> log/caltech101_CFMTL_0.25_L0.01.log 2>&1;
# # <!-- non-iid:0.50; L:0.01 -->
# python main.py --dataset caltech101 --iid non-iid --ratio 0.50 --ep 100 --local_ep 1 --frac 0.2 --num_batch 64 --num_clients 10 --clust 10 --L 0.01 --experiment performance-caltech101  --filename caltech101_CFMTL_0.50_L0.01 >> log/caltech101_CFMTL_0.50_L0.01.log 2>&1;
# # <!-- non-iid:0.75; L:0.01 -->
# python main.py --dataset caltech101 --iid iid --ep 100 --local_ep 1 --frac 0.2 --num_batch 64 --num_clients 10 --clust 10 --L 0.01 --experiment performance-caltech101  --filename caltech101_CFMTL_0.75_L0.01 >> log/caltech101_CFMTL_0.75_L0.01.log 2>&1;
# # <!-- non-iid:non-iid-single_class; L:0.01 -->
# python main.py --dataset caltech101 --iid non-iid-single_class --ep 100 --local_ep 1 --frac 0.2 --num_batch 64 --num_clients 10 --clust 10 --L 0.01 --experiment performance-caltech101  --filename caltech101_CFMTL_sc_L0.01 >> log/caltech101_CFMTL_sc_L0.01.log 2>&1

# <!-- non-iid:0.75; L:1 ; opt:sgd-->
python main.py --dataset caltech101 --iid non-iid --ratio 0.75 --ep 100 --local_ep 1 --frac 0.2 --num_batch 64 --num_clients 10 --clust 10 --L 1 --dist="Equal" --opt="sgd" --experiment performance-caltech101  --filename caltech101_CFMTL_0.75_equal_L1_sgd >> log/caltech101_CFMTL_0.75_equal_L1_sgd.log 2>&1;
# <!-- non-iid:0.75; L:0.1 ; opt:sgd-->
python main.py --dataset caltech101 --iid non-iid --ratio 0.75 --ep 100 --local_ep 1 --frac 0.2 --num_batch 64 --num_clients 10 --clust 10 --L 0.1 --dist="Equal" --opt="sgd" --experiment performance-caltech101  --filename caltech101_CFMTL_0.75_equal_L0.1_sgd >> log/caltech101_CFMTL_0.75_equal_L0.1_sgd.log 2>&1;
# <!-- non-iid:0.75; L:0.01; opt:sgd -->
python main.py --dataset caltech101 --iid non-iid --ratio 0.75 --ep 100 --local_ep 1 --frac 0.2 --num_batch 64 --num_clients 10 --clust 10 --L 0.01 --dist="Equal" --opt="sgd" --experiment performance-caltech101  --filename caltech101_CFMTL_0.75_equal_L0.01_sgd >> log/caltech101_CFMTL_0.75_equal_L0.01_sgd.log 2>&1;
# <!-- non-iid:0.75; L:0.1; opt:adam -->
python main.py --dataset caltech101 --iid non-iid --ratio 0.75 --ep 100 --local_ep 1 --frac 0.2 --num_batch 64 --num_clients 10 --clust 10 --L 0.1 --dist="Equal" --opt="adam" --experiment performance-caltech101  --filename caltech101_CFMTL_0.75_equal_L0.1_adam >> log/caltech101_CFMTL_0.75_equal_L0.1_adam.log 2>&1;
# <!-- non-iid:0.75; L:0.01; opt:adam -->
python main.py --dataset caltech101 --iid non-iid --ratio 0.75 --ep 100 --local_ep 1 --frac 0.2 --num_batch 64 --num_clients 10 --clust 10 --L 0.01 --dist="Equal" --opt="adam" --experiment performance-caltech101  --filename caltech101_CFMTL_0.75_equal_L0.01_adam >> log/caltech101_CFMTL_0.75_equal_L0.01_adam.log 2>&1

