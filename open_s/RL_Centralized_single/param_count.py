from agent import Agent
import torch
from agent import MLP

# model = MLP(20, 11)
# model.load_state_dict(torch.load('/home/hjh/pythonProject/packet_schedule_v2/centralized_advance/c_model/0_to_99_platform_model/flow_10/traffic_0/model.pth'))
#
# total_params = sum(p.numel() for p in model.parameters())
# total_params += sum(p.numel() for p in model.buffers())
# print(f'{total_params:,} total parameters.')
# print(f'{total_params/(1024*1024):.2f}M total parameters.')
# total_trainable_params = sum(
#     p.numel() for p in model.parameters() if p.requires_grad)
# print(f'{total_trainable_params:,} training parameters.')
# print(f'{total_trainable_params/(1024*1024):.2f}M training parameters.')
#
# print("================================================================")
# _dict = {}
# for _,param in enumerate(model.named_parameters()):
#     # print(param[0])
#     # print(param[1])
#     total_params = param[1].numel()
#     # print(f'{total_params:,} total parameters.')
#     k = param[0].split('.')[0]
#     if k in _dict.keys():
#         _dict[k] += total_params
#     else:
#         _dict[k] = 0
#         _dict[k] += total_params
#     # print('----------------')
# for k,v in _dict.items():
#     print(k)
#     print(v)
#     print("%3.5fM parameters" %  (v / (1024*1024)))
#     print('--------')

k = 50
mb = (3*k+3)*20 + k + 2;
mb *= 4

print("%.5f", mb/1024/1024)