from collections import deque
from model import Model


MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR= 0.001

class Agent:
    def __init__(self):
        super(Agent, self).__init__()
        self.n_game = 0
        self.epsilon = 0
        self.gamma = 0.9
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Model()
        # Todo : create and import trainer class

    