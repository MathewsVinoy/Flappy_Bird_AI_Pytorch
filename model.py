import torch
from torch import nn

class Model(nn.Model):
    def __init__(input_size, output_size, hidden_size):
        super(Model, self).__init__()
        self.seq = nn.Sequential(
            nn.Linear(in_features=input_size, out_features=hidden_size),
            nn.Linear(in_features=hidden_size, out_features=output_size),
        )
        
    def forward(self,x):
        return self.seq(x)