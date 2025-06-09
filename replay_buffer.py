import random
from collections import deque, namedtuple

import numpy as np

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size: int):
        samples = random.sample(self.buffer, batch_size)
        batch = Transition(*zip(*samples))
        return batch
