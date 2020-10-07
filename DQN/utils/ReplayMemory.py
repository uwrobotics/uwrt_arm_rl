import random
from collections import namedtuple

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = int(capacity)
        self.memory = []
        self.position = int(0)
        self.transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = self.transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)