#!/bin/bash
# bash ./scripts/baseline.sh
echo script name: $0
echo $# arguments

#digit
# CUDA_VISIBLE_DEVICES=2 python haokun/main.py --lr 0.01 --alg fedavg --dataset digit --num_users 5 --rounds 200 --num_bb 3 --feature_iid 0 --label_iid 0 --alpha 1 >digit_fedavg_fnln_3bb_5u_a1_001.log
# CUDA_VISIBLE_DEVICES=2 python haokun/main.py --lr 0.01 --alg fedavg --dataset digit --num_users 5 --rounds 200 --num_bb 3 --feature_iid 1 --label_iid 0 --alpha 1 >digit_fedavg_filn_3bb_5u_a1_001.log
# CUDA_VISIBLE_DEVICES=2 python haokun/main.py --lr 0.01 --alg fedavg --dataset digit --num_users 5 --rounds 200 --num_bb 3 --feature_iid 0 --label_iid 1 --alpha 1 >digit_fedavg_fnli_3bb_5u_a1_001.log

# CUDA_VISIBLE_DEVICES=2 python haokun/main.py --lr 0.01 --alg fedavg --dataset digit --num_users 5 --rounds 200 --num_bb 1 --feature_iid 0 --label_iid 0 --alpha 1 >digit_fedavg_fnln_1bb_5u_a1_001.log
# CUDA_VISIBLE_DEVICES=2 python haokun/main.py --lr 0.01 --alg fedavg --dataset digit --num_users 5 --rounds 200 --num_bb 1 --feature_iid 1 --label_iid 0 --alpha 1 >digit_fedavg_filn_1bb_5u_a1_001.log
# CUDA_VISIBLE_DEVICES=2 python haokun/main.py --lr 0.01 --alg fedavg --dataset digit --num_users 5 --rounds 200 --num_bb 1 --feature_iid 0 --label_iid 1 --alpha 1 >digit_fedavg_fnli_1bb_5u_a1_001.log

python exps/federated_main.py --alg fedavg --lr 0.005 --dataset office --num_users 5 --rounds 200 --num_bb 1 --feature_iid 1 --label_iid 0 --alpha 1 >office_fedavg_filn_1bb_5u_a1.log
python exps/federated_main.py --alg fedavg --lr 0.005 --dataset office --num_users 5 --rounds 200 --num_bb 3 --feature_iid 1 --label_iid 0 --alpha 1 >office_fedavg_filn_3bb_5u_a1.log
python exps/federated_main.py --alg fedavg --lr 0.005 --dataset office --num_users 5 --rounds 200 --num_bb 1 --feature_iid 0 --label_iid 0 --alpha 1 >office_fedavg_fnln_1bb_5u_a1.log
python exps/federated_main.py --alg fedavg --lr 0.005 --dataset office --num_users 5 --rounds 200 --num_bb 3 --feature_iid 0 --label_iid 0 --alpha 1 >office_fedavg_fnln_3bb_5u_a1.log
python exps/federated_main.py --alg fedavg --lr 0.005 --dataset office --num_users 5 --rounds 200 --num_bb 1 --feature_iid 0 --label_iid 1 --alpha 1 >office_fedavg_fnli_1bb_5u_a1.log
python exps/federated_main.py --alg fedavg --lr 0.005 --dataset office --num_users 5 --rounds 200 --num_bb 3 --feature_iid 0 --label_iid 1 --alpha 1 >office_fedavg_fnli_3bb_5u_a1.log
