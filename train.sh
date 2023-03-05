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

python haokun/main.py --alg fedavg --lr 0.005 --dataset office --num_users 5 --rounds 200 --num_bb 1 --feature_iid 1 --label_iid 0 --alpha 1 --train_method "none" >office_fedavg_filn_1bb_5u_a1_none.log
python haokun/main.py --alg fedavg --lr 0.005 --dataset office --num_users 5 --rounds 200 --num_bb 3 --feature_iid 1 --label_iid 0 --alpha 1 --train_method "none" >office_fedavg_filn_3bb_5u_a1_none.log
python haokun/main.py --alg fedavg --lr 0.005 --dataset office --num_users 5 --rounds 200 --num_bb 1 --feature_iid 0 --label_iid 0 --alpha 1 --train_method "none" >office_fedavg_fnln_1bb_5u_a1_none.log
python haokun/main.py --alg fedavg --lr 0.005 --dataset office --num_users 5 --rounds 200 --num_bb 3 --feature_iid 0 --label_iid 0 --alpha 1 --train_method "none" >office_fedavg_fnln_3bb_5u_a1_none.log
python haokun/main.py --alg fedavg --lr 0.005 --dataset office --num_users 5 --rounds 200 --num_bb 1 --feature_iid 0 --label_iid 1 --alpha 1 --train_method "none" >office_fedavg_fnli_1bb_5u_a1_none.log
python haokun/main.py --alg fedavg --lr 0.005 --dataset office --num_users 5 --rounds 200 --num_bb 3 --feature_iid 0 --label_iid 1 --alpha 1 --train_method "none" >office_fedavg_fnli_3bb_5u_a1_none.log

python haokun/main.py --alg fedavg --lr 0.005 --dataset office --num_users 5 --rounds 200 --num_bb 1 --feature_iid 1 --label_iid 0 --alpha 1 --train_method "bn" >office_fedavg_filn_1bb_5u_a1_bn.log
python haokun/main.py --alg fedavg --lr 0.005 --dataset office --num_users 5 --rounds 200 --num_bb 3 --feature_iid 1 --label_iid 0 --alpha 1 --train_method "bn" >office_fedavg_filn_3bb_5u_a1_bn.log
python haokun/main.py --alg fedavg --lr 0.005 --dataset office --num_users 5 --rounds 200 --num_bb 1 --feature_iid 0 --label_iid 0 --alpha 1 --train_method "bn" >office_fedavg_fnln_1bb_5u_a1_bn.log
python haokun/main.py --alg fedavg --lr 0.005 --dataset office --num_users 5 --rounds 200 --num_bb 3 --feature_iid 0 --label_iid 0 --alpha 1 --train_method "bn" >office_fedavg_fnln_3bb_5u_a1_bn.log
python haokun/main.py --alg fedavg --lr 0.005 --dataset office --num_users 5 --rounds 200 --num_bb 1 --feature_iid 0 --label_iid 1 --alpha 1 --train_method "bn" >office_fedavg_fnli_1bb_5u_a1_bn.log
python haokun/main.py --alg fedavg --lr 0.005 --dataset office --num_users 5 --rounds 200 --num_bb 3 --feature_iid 0 --label_iid 1 --alpha 1 --train_method "bn" >office_fedavg_fnli_3bb_5u_a1_bn.log

python haokun/main.py --alg fedavg --lr 0.005 --dataset office --num_users 5 --rounds 200 --num_bb 1 --feature_iid 1 --label_iid 0 --alpha 1 --train_method "full" >office_fedavg_filn_1bb_5u_a1_full.log
python haokun/main.py --alg fedavg --lr 0.005 --dataset office --num_users 5 --rounds 200 --num_bb 3 --feature_iid 1 --label_iid 0 --alpha 1 --train_method "full" >office_fedavg_filn_3bb_5u_a1_full.log
python haokun/main.py --alg fedavg --lr 0.005 --dataset office --num_users 5 --rounds 200 --num_bb 1 --feature_iid 0 --label_iid 0 --alpha 1 --train_method "full" >office_fedavg_fnln_1bb_5u_a1_full.log
python haokun/main.py --alg fedavg --lr 0.005 --dataset office --num_users 5 --rounds 200 --num_bb 3 --feature_iid 0 --label_iid 0 --alpha 1 --train_method "full" >office_fedavg_fnln_3bb_5u_a1_full.log
python haokun/main.py --alg fedavg --lr 0.005 --dataset office --num_users 5 --rounds 200 --num_bb 1 --feature_iid 0 --label_iid 1 --alpha 1 --train_method "full" >office_fedavg_fnli_1bb_5u_a1_full.log
python haokun/main.py --alg fedavg --lr 0.005 --dataset office --num_users 5 --rounds 200 --num_bb 3 --feature_iid 0 --label_iid 1 --alpha 1 --train_method "full" >office_fedavg_fnli_3bb_5u_a1_full.log

python haokun/main.py --alg fedavg --lr 0.005 --dataset office --num_users 5 --rounds 200 --num_bb 1 --feature_iid 1 --label_iid 0 --alpha 1 --train_method "adapter" >office_fedavg_filn_1bb_5u_a1_adapter.log
python haokun/main.py --alg fedavg --lr 0.005 --dataset office --num_users 5 --rounds 200 --num_bb 3 --feature_iid 1 --label_iid 0 --alpha 1 --train_method "adapter" >office_fedavg_filn_3bb_5u_a1_adapter.log
python haokun/main.py --alg fedavg --lr 0.005 --dataset office --num_users 5 --rounds 200 --num_bb 1 --feature_iid 0 --label_iid 0 --alpha 1 --train_method "adapter" >office_fedavg_fnln_1bb_5u_a1_adapter.log
python haokun/main.py --alg fedavg --lr 0.005 --dataset office --num_users 5 --rounds 200 --num_bb 3 --feature_iid 0 --label_iid 0 --alpha 1 --train_method "adapter" >office_fedavg_fnln_3bb_5u_a1_adapter.log
python haokun/main.py --alg fedavg --lr 0.005 --dataset office --num_users 5 --rounds 200 --num_bb 1 --feature_iid 0 --label_iid 1 --alpha 1 --train_method "adapter" >office_fedavg_fnli_1bb_5u_a1_adapter.log
python haokun/main.py --alg fedavg --lr 0.005 --dataset office --num_users 5 --rounds 200 --num_bb 3 --feature_iid 0 --label_iid 1 --alpha 1 --train_method "adapter" >office_fedavg_fnli_3bb_5u_a1_adapter.log
