import numpy as np

d3qn = np.load("/home/hjh/pythonProject/packet_schedule_v2/centralized_gen_torch_2/c_gen_model/flow_8_p_1_hidd_60/flow_8_traffic_0_to_1499/plot_ME_D3QN_throughput_8.npy")
theory = np.load("/home/hjh/pythonProject/packet_schedule_v2/centralized_gen_torch_2/c_gen_model/flow_8_p_1_hidd_60/flow_8_traffic_0_to_1499/plot_theory_throughput_8.npy")

for i in range(len(d3qn)):
    print(f"{d3qn[i]} {theory[i]}")