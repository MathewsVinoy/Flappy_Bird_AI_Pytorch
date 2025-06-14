import  torch  
from torch import nn, optim 
from model import Model

class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.float)
        reward = torch.tensor(reward, dtype=torch.float)

        if len(state.shape)==1:
            state= torch.unsqueeze(state,0)
            next_state= torch.unsqueeze(next_state,0)
            action= torch.unsqueeze(action,0)
            reward= torch.unsqueeze(reward,0)
            done = (done, )
        
        # 1: predicted Q Values with current state
        pred = self.model(state)

        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] +self.gamma * torch.max(self.model(next_state[idx]))

            action_idx = torch.argmax(action).item()
            if action_idx >= pred.shape[1]:
                print(f"Action index {action_idx} is out of bounds for pred shape {pred.shape}")
                action_idx = pred.shape[1] - 1  # Clamp to the last valid index

            target[idx][action_idx] = Q_new

        # 2: Q_new = r+y* max(next_predicted Q value)
        # pred.clone()
        # pred[argmax(action)] = Q_new
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()