import os
import torch
from torch import nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self,input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size-100)
        self.linear3 = nn.Linear(hidden_size, output_size)

    def forward(self,x):
        x= F.sigmoid(self.linear1(x))
        x=self.linear3(x)
        return x

    def save(self, file_name='model1.pth'):
        model_folder_path='./model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)
