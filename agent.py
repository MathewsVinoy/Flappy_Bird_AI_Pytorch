import torch
import random
from model import Model
from Qtrainer import QTrainer
from collections import deque
from gameplay import GamePlay

# Constants
MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:
    def __init__(self):
        self.n_game = 0
        self.epsilon = 0 
        self.gamma = 0.9  
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Model(3, 1024, 2) 
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step([state], [action], [reward], [next_state], [done])

    def get_action(self, state):
        output = []
        for i in range(len(state)):
            self.epsilon = max(10, 80 - self.n_game) 
            final_move = [0, 0]

            if random.randint(0, 200) < self.epsilon:
                move = random.randint(0, 1)
                final_move[move] = 1
            else:
                state0 = torch.tensor(state[i], dtype=torch.float)
                prediction = self.model(state0)
                move = torch.argmax(prediction).item()
                final_move[move] = 1

            output.append(final_move)
        return output


def train():
    record = 0
    agent = Agent()
    game = GamePlay()

    try:
        while True:

            state_old = game.get_state()

            final_move = agent.get_action(state_old)

            score, birds = game.play_step(final_move)
            state_new = game.get_state()

            for idx, bird in enumerate(birds):
                if idx >= len(state_old):
                    continue
                agent.train_short_memory(state_old[idx], final_move[idx], bird.reward, state_new[idx], bird.game_over)
                agent.remember(state_old[idx], final_move[idx], bird.reward, state_new[idx], bird.game_over)

            if len(birds) == 0:
                game.reset()
                agent.n_game += 1
                agent.train_long_memory()

                if score > record:
                    record = score
                    agent.model.save()

                print(f"Game {agent.n_game} | Score: {score} | Record: {record}")

    except KeyboardInterrupt:
        print("\nTraining manually interrupted. Exiting...")

if __name__ == "__main__":
    train()
