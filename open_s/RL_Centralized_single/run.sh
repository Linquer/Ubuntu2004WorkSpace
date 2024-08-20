source activate torch
CUDA_VISIBLE_DEVICES=2 nohup python3 main.py \
--flow_num 6 \
--traffic_id_begin  700 \
--traffic_id_end 729 \
--model_suffix risk_case \
--traffic_prefix c_gen_ \
--traffic_exp_name flow_6_risk_deadline &