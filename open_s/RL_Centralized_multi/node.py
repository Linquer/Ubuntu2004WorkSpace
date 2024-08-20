# 定义节点类(此处以一对节点为单位，因为接收节点不对网络环境产生影响，因此不进行建模)
from packet import Packet
from packet_buffer import PacketBuffer
import copy
import random

# 数据流
class Node:
    total_pair_count = 0

    def __init__(self, config):
        Node.total_pair_count += 1
        self.pair_id = Node.total_pair_count
        # traffic information
        self.offset = 0
        self.inter_period = 0
        self.deadline = 0
        self.arrival_prob = 0
        self.channel_prob = 0
        # packet buffer
        self.packet_buffer = PacketBuffer(self.pair_id)
        # 统计信息
        self.generate_packet_count = 0
        self.send_before_expiration = 0
        self.dealloc_for_expiration = 0
        # total
        self.mac_delay_from_head_to_send = 0
        self.mac_delay_from_queue_to_send = 0
        # success
        self.mac_delay_from_head_to_send_success = 0
        self.mac_delay_from_queue_to_send_success = 0
        self.stats_slot_begin = config.INITIAL_PHASE_TIME + 1

    # 初始化节点信息
    def initialize(self, config, traffic_id):
        # 统计信息
        self.dealloc_for_expiration = 0
        self.send_before_expiration = 0
        self.generate_packet_count = 0
        # total
        self.mac_delay_from_head_to_send = 0
        self.mac_delay_from_queue_to_send = 0
        # success
        self.mac_delay_from_head_to_send_success = 0
        self.mac_delay_from_queue_to_send_success = 0
        # 缓存队列
        self.packet_buffer = PacketBuffer(self.pair_id)
        self.stats_slot_begin = config.INITIAL_PHASE_TIME + 1
        # 流量信息
        self.offset = config.OFFSET[traffic_id][self.pair_id - 1]
        self.inter_period = config.INTER_PERIOD[traffic_id][self.pair_id - 1]
        self.deadline = config.DEADLINE[traffic_id][self.pair_id - 1]
        self.arrival_prob = config.ARRIVAL_PROB[traffic_id][self.pair_id - 1]
        self.channel_prob = config.CHANNEL_PROB[traffic_id][self.pair_id - 1]

    # 根据流量获取下一个数据包
    def generate_new_packet(self, time_slot):
        if (time_slot - self.offset - 1) >= 0 and (time_slot - self.offset - 1) % self.inter_period == 0:
            # 应用层到mac层产生丢包，到达模型B_k
            random_number = random.random()
            if random_number <= self.arrival_prob:
                packet = Packet(channel_probability=self.channel_prob,
                                arrival_time=time_slot,
                                expiration_time=time_slot + self.deadline - 1)
                self.packet_buffer.add_packet(packet=packet)
                # 更新数据包队头信息
                self.packet_buffer.update_packet_arrive_head_time(time_slot=time_slot)
                # stats information
                self.generate_packet_count = self.generate_packet_count + 1 \
                    if time_slot >= self.stats_slot_begin \
                    else self.generate_packet_count

    # 检查缓存中过期的数据包进行删除
    def check_buffer_expiration(self, time_slot):
        packet_dealloc, \
        mac_delay_from_queue_to_send,\
        mac_delay_from_head_to_send = self.packet_buffer.check_expiration(time_slot=time_slot)
        # 更新数据包队头信息
        self.packet_buffer.update_packet_arrive_head_time(time_slot=time_slot)
        # stats information
        # self.dealloc_for_expiration = self.dealloc_for_expiration + packet_dealloc \
        #     if time_slot >= self.stats_slot_begin and self.generate_packet_count >= 1 \
        #     else self.dealloc_for_expiration
        # ------  安全检查  --------
        if packet_dealloc >= 2:
            exit('there just can have only one packet dealloc in one timeslot, present is : ' + str(packet_dealloc))
        if packet_dealloc == 1 and mac_delay_from_queue_to_send != self.deadline + 1:
            exit('a packet is dealloc but its mac_delay_from_queue_to_send is not equal to (deadline+1) '
                 + str(mac_delay_from_queue_to_send))
        if time_slot >= self.stats_slot_begin and self.generate_packet_count >= 1:
            self.dealloc_for_expiration = self.dealloc_for_expiration + packet_dealloc
            # 将因为超过截止期限而丢弃的数据包也算进MAC时延里
            # 因为超过截止期限而丢弃的数据包其MAC时延计为deadline + 1
            self.mac_delay_from_queue_to_send += mac_delay_from_queue_to_send
            self.mac_delay_from_head_to_send += mac_delay_from_head_to_send
        return mac_delay_from_queue_to_send, mac_delay_from_head_to_send

    # 检查缓存中最快过期的数据包进行删除(send packet)
    def remove_buffer_expiration(self, time_slot):
        packet_have, packet_sent, probability, \
        mac_delay_from_queue_to_send, \
        mac_delay_from_head_to_send = self.packet_buffer.remove_expiration(time_slot=time_slot)
        # 更新数据包队头信息
        self.packet_buffer.update_packet_arrive_head_time(time_slot=time_slot)
        # stats information
        if time_slot >= self.stats_slot_begin and self.generate_packet_count >= 1:
            self.send_before_expiration = self.send_before_expiration + packet_sent
            self.mac_delay_from_queue_to_send += mac_delay_from_queue_to_send
            self.mac_delay_from_head_to_send += mac_delay_from_head_to_send
            # success
            self.mac_delay_from_queue_to_send_success += mac_delay_from_queue_to_send
            self.mac_delay_from_head_to_send_success += mac_delay_from_head_to_send
        return packet_have, packet_sent, probability, mac_delay_from_queue_to_send, mac_delay_from_head_to_send

    # 获取节点状态(用于神经网络输入)
    def get_node_state(self, time_slot):
        return self.packet_buffer.get_buffer_state_v2(time_slot=time_slot)

    # 获取节点真实状态(直观获取buffer信息)
    def get_node_true_state(self, time_slot):
        return self.packet_buffer.get_buffer_state_v1(time_slot=time_slot)
