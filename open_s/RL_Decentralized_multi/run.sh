source activate torch
CUDA_VISIBLE_DEVICES=0 nohup python3 distributed_train.py \
--train \
--node_num  10 \
--flow_num 10 \
--traffic_id_begin  0 \
--traffic_id_end 15 \
--traffic_exp_name manual_flow_10_10_meta/group_1 \
--model_suffix meta_reward_2 \
--exp_prefix gen_ \
--random_seed 13 &
