# 定义环境类
import math

from node import Node


class Environment:
    # 初始化环境变量
    def __init__(self, config):
        # global information initialize
        self.node_pair_number = config.FLOW_NUMBER
        # node entities
        Node.total_pair_count = 0
        self.nodes = [Node(config=config) for x in range(config.FLOW_NUMBER)]

    '''初始化环境信息'''

    def initialize(self, config, traffic_id):
        self.LAMBDA_VALUE = config.LAMBDA_VALUE
        self.INITIAL_PHASE_TIME = config.INITIAL_PHASE_TIME
        self.STEADY_PHASE_TIME = config.STEADY_PHASE_TIME
        for node in self.nodes:
            node.initialize(config=config, traffic_id=traffic_id)
            node.generate_new_packet(time_slot=1)
        return self.nodes

    '''环境运行，返回下一状态,回报值,结束标志'''

    def step(self, time_slot, action):
        # two level reward definition
        if 1 <= action <= self.node_pair_number:
            packet_have, packet_sent, probability, \
            mac_delay_from_queue_to_send, \
            mac_delay_from_head_to_send = self.nodes[action - 1].remove_buffer_expiration(time_slot=time_slot)

            if packet_have == 1:
                reward = probability
            else:
                reward = 0
            # 注意：奖励函数3有两处地方设置
            # 奖励函数3：
            # 在指定的及时吞吐量下达到最小MAC时延(被目的节点接收的数据包)
            # if packet_have == 1:
            #     reward_temp = probability
            # else:
            #     reward_temp = 0
            # reward = -self.LAMBDA_VALUE * reward_temp - mac_delay_from_queue_to_send
        else:
            reward = 0

        # remove expired packet
        expire_list = []
        for node in self.nodes:
            packet_dealloc,\
            mac_delay_from_queue_to_send, \
            mac_delay_from_head_to_send = node.check_buffer_expiration(time_slot=time_slot)
            expire_list.append(packet_dealloc)
            # 奖励函数3：--统计信息用
            # 在指定的及时吞吐量下达到最小MAC时延(被丢弃的数据包)
            # reward += -mac_delay_from_queue_to_send

        # generate new packet
        generate_list = []
        for node in self.nodes:
            generate_list.append(node.generate_new_packet(time_slot=time_slot + 1))

        # check the end label
        # if time_slot >= self.INITIAL_PHASE_TIME and (time_slot - self.INITIAL_PHASE_TIME) % self.STEADY_PHASE_TIME == 0:
        #     end = True
        # else:
        #     end = False
        end = False

        # get the next state
        next_state = self.get_state(time_slot=time_slot + 1)

        return next_state, reward, end, expire_list, generate_list

    ''' get the environment state'''

    def get_state(self, time_slot):
        state_in_buffer = []
        state_in_channel_prob = []
        state_in_packet_arrival_indicator = []
        for node in self.nodes:
            # version 1 : state transform（this paper propose）
            state_in_buffer.append(node.get_node_state(time_slot=time_slot))
            if (time_slot + 1 - self.nodes[node.pair_id - 1].offset - 1 >= 0) and \
                    (time_slot + 1 - self.nodes[node.pair_id - 1].offset - 1) % self.nodes[
                node.pair_id - 1].inter_period == 0:
                state_in_packet_arrival_indicator.append(1)
            else:
                state_in_packet_arrival_indicator.append(0)
            # version 2 : binary string representation （compared version）
            # state_in_buffer.append(node.get_node_true_state(time_slot=time_slot))
            # channel info
            state_in_channel_prob.append(node.channel_prob)
        # state = state_in_buffer + state_in_channel_prob
        # for version 1
        return state_in_buffer + state_in_packet_arrival_indicator
        # for version 2
        # return state_in_buffer

    def get_true_state(self, time_slot):
        state_in_buffer = []
        for node in self.nodes:
            state_in_buffer.append(node.get_node_true_state(time_slot=time_slot))
        return state_in_buffer

    def get_stats(self):
        total_generate_packet_count = 0
        total_send_before_expiration = 0
        total_mac_delay_from_queue_to_send = 0
        total_mac_delay_from_head_to_send = 0
        total_mac_delay_from_queue_to_send_success = 0
        total_mac_delay_from_head_to_send_success = 0
        for node in self.nodes:
            total_generate_packet_count += node.generate_packet_count
            total_send_before_expiration += node.send_before_expiration
            total_mac_delay_from_queue_to_send += node.mac_delay_from_queue_to_send
            total_mac_delay_from_head_to_send += node.mac_delay_from_head_to_send
            total_mac_delay_from_queue_to_send_success += node.mac_delay_from_queue_to_send_success
            total_mac_delay_from_head_to_send_success += node.mac_delay_from_head_to_send_success

        # success
        average_mac_delay_from_queue_to_send_success = total_mac_delay_from_queue_to_send_success / total_send_before_expiration if total_send_before_expiration != 0 else 0
        average_mac_delay_from_head_to_send_success = total_mac_delay_from_head_to_send_success / total_send_before_expiration if total_send_before_expiration != 0 else 0
        # total
        average_mac_delay_from_queue_to_send = total_mac_delay_from_queue_to_send / total_generate_packet_count if total_generate_packet_count != 0 else 0
        average_mac_delay_from_head_to_send = total_mac_delay_from_head_to_send / total_generate_packet_count if total_generate_packet_count != 0 else 0
        return total_send_before_expiration, total_generate_packet_count, \
               average_mac_delay_from_queue_to_send, average_mac_delay_from_head_to_send, \
               average_mac_delay_from_queue_to_send_success, average_mac_delay_from_head_to_send_success
