source activate torch
CUDA_VISIBLE_DEVICES=0 nohup python3 distributed_train.py \
--train \
--node_num 4 \
--flow_num 4 \
--traffic_id_begin 0 \
--traffic_id_end 5 \
--model_suffix training_time_compare \
--traffic_exp_name flow_4_flow_compare &


#source activate torch
#CUDA_VISIBLE_DEVICES=2 nohup python3 distributed_train.py \
#--train \
#--node_num  10 \
#--flow_num 10 \
#--traffic_id_begin  3 \
#--traffic_id_end 3 \
#--traffic_exp_name manual_flow_10_10_meta/group_1 \
#--model_suffix bug_test \
#--exp_prefix gen_ &