import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy

class DQN_FCN(nn.Module):
    def __init__(self):
        super(DQN_FCN, self).__init__()
        self.conv1 = nn.Conv2d(2, 32, 3, stride=1)  # 7x6 -> 5x4
        self.conv2 = nn.Conv2d(32, 128, 3, stride=1)  # 5x4 -> 3x2
        self.linear4 = nn.Linear(128*3*2, 2048)
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
        o7 = self.linear6(o6)
        #print(o6.shape)
        return o7


class DQN_FCN_WIDE(nn.Module):
    def __init__(self):
        super(DQN_FCN_WIDE, self).__init__()
        self.conv1 = nn.Conv2d(2, 64, 3, stride=1)  # 7x6 -> 5x4
        self.conv2 = nn.Conv2d(64, 256, 3, stride=1)  # 5x4 -> 3x2
        self.linear4 = nn.Linear(256*3*2, 2048)
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
        o4 = o2.view(-1, 256*3*2)
        #print(o4.shape)
        o5 = F.relu(self.linear4(o4))
        o6 = F.relu(self.linear5(o5))
        #print(o5.shape)
        o7 = self.linear6(o6)
        #print(o6.shape)
        return o7


class DQN_LINEAR(nn.Module):
    def __init__(self):
        super(DQN_LINEAR, self).__init__()
        self.linear1 = nn.Linear(2*7*6, 1024)
        self.linear2 = nn.Linear(1024, 2048)
        self.linear3 = nn.Linear(2048, 2048)
        self.linear4 = nn.Linear(2048, 512)
        self.linear5 = nn.Linear(512, 128)
        self.linear6 = nn.Linear(128, 7)

    def forward(self, X):
        o0 = X.view(-1, 2*7*6)
        o1 = F.relu(self.linear1(o0))
        o2 = F.relu(self.linear2(o1))
        o3 = F.relu(self.linear3(o2))
        o4 = F.relu(self.linear4(o3))
        o5 = F.relu(self.linear5(o4))
        o6 = self.linear6(o5)
        return o6


class DQN_SKIP(nn.Module):
    def __init__(self):
        super(DQN_SKIP, self).__init__()
        self.conv1 = nn.Conv2d(2, 32, 3, stride=1)  # 7x6 -> 5x4
        self.conv2 = nn.Conv2d(32, 128, 3, stride=1)  # 5x4 -> 3x2
        self.linear3 = nn.Linear(128*3*2 + 7*6*2, 2048)
        self.linear4 = nn.Linear(2048, 512)
        self.linear5 = nn.Linear(512, 7)

    def forward(self, X):
        #print(X.shape)
        o1 = F.relu(self.conv1(X))
        #print(o1.shape)
        o2 = F.relu(self.conv2(o1))
        #print(o2.shape)
        #   o3 = F.relu(self.conv3(o2))
        #   print(o3.shape)
        o2_l = o2.view(-1, 128*3*2)
        X_l = X.view(-1, 7*6*2)
        X_o2_concat = torch.cat((o2_l, X_l), dim=1)
        #print(o4.shape)
        o3 = F.relu(self.linear3(X_o2_concat))
        o4 = F.relu(self.linear4(o3))
        #print(o5.shape)
        o5 = self.linear5(o4)
        #print(o6.shape)
        return o5
