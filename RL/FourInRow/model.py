import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN_FCN(nn.Module):
    def __init__(self):
        super(DQN_FCN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, stride=1)  #7X6
        self.conv2 = nn.Conv2d(32, 128, 3, stride=1)  #5X4
        #self.conv3 = nn.Conv2d(128, 512, 3, stride=1)  #3X2
        self.linear4 = nn.Linear(128*3*2, 2048)  #4X4X16
        self.linear5 = nn.Linear(2048, 512)
        self.linear6 = nn.Linear(512, 7)

    def forward(self, X):
        #print(X.shape)
        o1 = F.relu(self.conv1(X))
        #print(o1.shape)
        o2 = F.relu(self.conv2(o1))
        #print(o2.shape)
        #   o3 = F.relu(self.conv3(o2))
        #   print(o3.shape)
        o4 = o2.view(-1, 128*3*2)
        #print(o4.shape)
        o5 = F.relu(self.linear4(o4))
        o6 = F.relu(self.linear5(o5))
        #print(o5.shape)
        o7 = F.sigmoid(self.linear6(o6))
        #print(o6.shape)
        return o6

