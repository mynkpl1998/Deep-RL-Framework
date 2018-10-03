import torch.nn as nn
import torch

# Define your network here
# input output format should be same as provided here.
class q_network(nn.Module):

    def __init__(self,input_size,out_size):
        super(q_network,self).__init__()

        self.input_size = input_size
        self.out_size = out_size

        self.layer1 = nn.Linear(out_features=24,in_features=self.input_size)
        self.layer2 = nn.Linear(out_features=48,in_features=24)
        self.layer3 = nn.Linear(out_features=self.out_size,in_features=48)

        self.relu = nn.ReLU()

    def forward(self,x,bsize):

        x = x.view(bsize,self.input_size)
        q_out = self.relu(self.layer1(x))
        q_out = self.relu(self.layer2(q_out))
        q_out = self.layer3(q_out)

        return q_out
