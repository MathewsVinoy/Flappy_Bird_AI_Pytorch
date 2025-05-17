import os
import torch
from torch import nn

class Model(nn.Model):
    def __init__(self,input_size, output_size, hidden_size):
        super(Model, self).__init__()
        self.seq = nn.Sequential(
            nn.Linear(in_features=input_size, out_features=hidden_size),
            nn.Linear(in_features=hidden_size, out_features=output_size),
        )
        
    def forward(self,x):
        return self.seq(x)
    
    def save(self, file_name='model.pth'):
        model_folder_path='./model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)