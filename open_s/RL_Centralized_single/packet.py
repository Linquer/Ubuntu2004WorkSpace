
class Packet:
    def __init__(self, channel_probability, arrival_time, expiration_time):
        # 数据包的到达时间，即：数据包到达队列的时间
        self.arrival_time = arrival_time
        # 数据包到达队列头部的时间
        # 如果是-1，表明该数据包尚未到达队列头部，否则表明该数据已经在队列头部
        self.arrive_in_head_time = -1

        self.expiration_time = expiration_time
        self.channel_probability = channel_probability

    def set_arrival_time(self, arrival_time):
        self.arrival_time = arrival_time

    def set_expiration_time(self, expiration_time):
        self.expiration_time = expiration_time

    def get_arrival_time(self):
        return self.arrival_time

    def get_expiration_time(self):
        return self.expiration_time

    def get_channel_probability(self):
        return self.channel_probability
