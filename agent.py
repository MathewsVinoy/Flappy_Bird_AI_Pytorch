from collections import deque
from model import Model
from game import GamePlay
import random
from Qtrainer import QTrainer
import torch

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR= 0.001

class Agent:
    def __init__(self):
        self.n_game = 0
        self.epsilon = 0
        self.gamma = 0.9
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Model()
        # Todo : create and import trainer class
        self.trainer =QTrainer(self.model, lr=LR, gamma=self.gamma)


    def remember(self,state, action, reward, next_state, done):
        self.memory.append((state,action,reward,next_state,done))

    def train_long_memory(self):
        if len(self.memory)>BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample=self.memory

        states,actions,rewards,next_states,dones=zip(*mini_sample)
        self.trainer.train_step(states,actions,rewards,next_states,dones)

    def train_short_memory(self,state, action, reward, next_state, done):
        self.trainer.train_step(state,action,reward,next_state,done)

    def get_action(self, state):
        self.epsilon = 80 - self.n_games
        if random.randint(0, 200) < self.epsilon:

            move = random.randint(0, 2)
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()

        return 1 if move == 0 else 0

    
def train():
    total_score =0
    record =0
    agent = Agent()
    game = GamePlay()
    while True:
        state_old = game.get_state()
        final_move = agent.get_action(state_old)
        print(final_move)
        reward, done, score = game.play_step(final_move)
        state_new = game.get_state()
        
    

if __name__ == "__main__":
    train()