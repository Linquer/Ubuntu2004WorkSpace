# 节点缓存
import copy
import random


class PacketBuffer:

    def __init__(self, id):
        self.buffer = []
        self.id = id

    # 向缓存中添加单个数据包
    def add_packet(self, packet):
        self.buffer.append(packet)

    # 检查缓存中过期的数据包进行删除
    def check_expiration(self, time_slot):
        expiration_packet_count = 0
        mac_delay_from_queue_to_send = 0
        mac_delay_from_head_to_send = 0
        # 遍历缓存判断每个数据包是否过期
        for i in range(len(self.buffer) - 1, -1, -1):
            if self.buffer[i].get_expiration_time() == time_slot:
                mac_delay_from_queue_to_send = time_slot - self.buffer[i].arrival_time + 2
                if self.buffer[i].arrive_in_head_time == -1:
                    exit('a packet which is not in head of the queue is dealloc')
                mac_delay_from_head_to_send = time_slot - self.buffer[i].arrive_in_head_time + 2
                del self.buffer[i]
                expiration_packet_count += 1
        return expiration_packet_count, mac_delay_from_queue_to_send, mac_delay_from_head_to_send

    # 概率删除最快过期的数据包
    def remove_expiration(self, time_slot):
        mac_delay_from_queue_to_send = 0
        mac_delay_from_head_to_send = 0
        min_lead_time = 0
        min_lead_time_id = 0
        if len(self.buffer) == 0:
            packet_have = 0
            packet_sent = 0
            probability = 0
            return packet_have, packet_sent, probability, \
                   mac_delay_from_queue_to_send, mac_delay_from_head_to_send
        # 遍历缓存找到最快过期的数据包
        for packet_id in range(len(self.buffer) - 1, -1, -1):
            lead_time = self.buffer[packet_id].get_expiration_time() - time_slot
            if packet_id == len(self.buffer) - 1:
                min_lead_time = lead_time
                min_lead_time_id = packet_id
            else:
                if lead_time < min_lead_time:
                    min_lead_time_id = packet_id
                    min_lead_time = lead_time
        probability = self.buffer[min_lead_time_id].get_channel_probability()
        if random.random() > self.buffer[min_lead_time_id].get_channel_probability():
            packet_have = 1
            packet_sent = 0  # 未成功发出，该数据包不能删除，即：不能从队头丢掉
        else:
            arrival_queue_time_of_sent_packet = self.buffer[min_lead_time_id].arrival_time
            if self.buffer[min_lead_time_id].arrive_in_head_time == -1:
                exit('a packet which is not in head of the queue is sent')
            arrival_head_time_of_sent_packet = self.buffer[min_lead_time_id].arrive_in_head_time
            del self.buffer[min_lead_time_id]
            packet_have = 1
            packet_sent = 1  # 成功发出，将该数据包从缓存中删除，即：从队头丢掉
            mac_delay_from_queue_to_send = time_slot - arrival_queue_time_of_sent_packet + 1 # 从进入队列到被目的节点接收的MAC时延
            mac_delay_from_head_to_send = time_slot - arrival_head_time_of_sent_packet + 1 # 从到达队列首部到被目的节点成功接收的MAC时延
        return packet_have, packet_sent, probability, \
               mac_delay_from_queue_to_send, mac_delay_from_head_to_send

    # 更新数据包到达队列头部的信息
    def update_packet_arrive_head_time(self, time_slot):
        # 当前队列中没有数据包，直接返回
        if len(self.buffer) == 0:
            return
        # 当前队列中有数据包，找到队首数据包更新其到达队首的时间
        # 当前队列中lead time值最小的数据包是在队首的数据包
        min_lead_time = -1
        min_lead_time_id = -1
        for packet_id in range(len(self.buffer)):
            lead_time = self.buffer[packet_id].get_expiration_time() - time_slot
            if packet_id == 0:
                min_lead_time = lead_time
                min_lead_time_id = packet_id
            else:
                if lead_time < min_lead_time:
                    min_lead_time_id = packet_id
                    min_lead_time = lead_time
        # 该数据包是首次到达队列首部
        if self.buffer[min_lead_time_id].arrive_in_head_time == -1:
            self.buffer[min_lead_time_id].arrive_in_head_time = time_slot

    # 获取当前缓存当中状态信息：(索引表示距离过期的剩余时间，值表示对应数据包数量)
    # binary string based representation
    def get_buffer_state_v1(self, time_slot):
        if len(self.buffer) == 0:
            return 0
        # 寻找最大距离过期剩余时间
        max_lead_time = 0
        for packet in self.buffer:
            # 数据包距离过期剩余时间=过期时间-当前时间
            lead_time = packet.get_expiration_time() - time_slot + 1
            max_lead_time = lead_time if lead_time > max_lead_time else max_lead_time
        # 根据最大距离过期剩余时间确定状态长度
        state_in_list = [0 for i in range(max_lead_time)]
        for packet in self.buffer:
            # 数据包距离过期剩余时间=过期时间-当前时间
            lead_time = packet.get_expiration_time() - time_slot + 1
            state_in_list[lead_time - 1] += 1
        # 将状态形式转换为整数
        multiply_number = 1
        state_in_number = 0
        for number in state_in_list:
            state_in_number += number * multiply_number
            multiply_number *= 10
        return state_in_number

    # equivalent ratio sequence base representation
    def get_buffer_state_v2(self, time_slot):
        state_in_number = 0
        for packet in self.buffer:
            lead_time = packet.get_expiration_time() - time_slot + 1
            state_in_number += 10 / pow(2, lead_time - 1)
        return state_in_number
