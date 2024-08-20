import random
import numpy as np
from itertools import count
from collections import  deque

tiebreaker = count()


class Replay_buffer:
    def __init__(self, capacity):
        self.buffer = []
        self.capacity = capacity
        self.cur = 0

    def push(self, state_1, state_2, action, reward, next_state_1, next_state_2):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)

        self.buffer[self.cur] = (state_1, state_2, action, reward, next_state_1, next_state_2)
        # print(f"exp push : {self.buffer[self.cur]}")
        self.cur = (self.cur + 1) % self.capacity

    def sample_batch(self, batch_size):
        # print(f"now buffer : {self.buffer}")
        batch = random.sample(self.buffer, batch_size)
        state_1, state_2, action, reward, next_state_1, next_state_2 = zip(*batch)
        return state_1, state_2, action, reward, next_state_1, next_state_2

    def __len__(self):
        return len(self.buffer)


# SumTree
# a binary tree data structure where the parent’s value is the sum of its children
class SumTree:
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0

    # update to the root node
    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    # find sample on leaf node
    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    # store priority and sample
    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    # update priority
    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    # get priority and sample
    def get(self, s):
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.data[data_idx])


class Priority_replay_buffer:
    e = 0.01
    a = 0.6
    beta = 0.4
    beta_increment_per_sampling = 0.001

    def __init__(self, capacity):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.size = 0

    def _get_priority(self, error):
        return (np.abs(error) + self.e) ** self.a

    def add(self, error, sample):
        p = self._get_priority(error)
        self.tree.add(p, sample)
        self.size += 1

    def sample_batch(self, n):
        batch = []
        idxs = []
        segment = self.tree.total() / n
        priorities = []

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            while data == 0:
                s = random.uniform(a, b)
                (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        sampling_probabilities = priorities / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()

        state_1, state_2, action, reward, next_state_1, next_state_2 = zip(*batch)

        return state_1, state_2, action, reward, next_state_1, next_state_2, idxs, is_weight

    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)
