source activate torch
CUDA_VISIBLE_DEVICES=0 nohup python3 main.py \
--node_count 6 \
--traffic_begin_id 0 \
--traffic_end_id 699 \
--traffic_prefix c_gen_ \
--model_suffix hidd_60_2 \
--traffic_exp_name flow_6_p_1 &
