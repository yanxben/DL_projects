import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN_CNN(nn.Module):
    def __init__(self):
        super(DQN_CNN, self).__init__()
        self.conv1 = nn.Conv2d(2, 32, 3, stride=1)  # 7x6 -> 5x4
        self.conv2 = nn.Conv2d(32, 128, 3, stride=1)  # 5x4 -> 3x2
        self.linear4 = nn.Linear(128*3*2, 2048)
        self.linear5 = nn.Linear(2048, 512)
        self.linear6 = nn.Linear(512, 7)

    def forward(self, X):
        o1 = F.relu(self.conv1(X))
        o2 = F.relu(self.conv2(o1))
        o4 = o2.view(-1, 128*3*2)
        o5 = F.relu(self.linear4(o4))
        o6 = F.relu(self.linear5(o5))
        o7 = self.linear6(o6)
        return o7


class DQN_CNN_WIDE(nn.Module):
    def __init__(self):
        super(DQN_CNN_WIDE, self).__init__()
        self.conv1 = nn.Conv2d(2, 64, 3, stride=1)  # 7x6 -> 5x4
        self.conv2 = nn.Conv2d(64, 256, 3, stride=1)  # 5x4 -> 3x2
        self.linear4 = nn.Linear(256*3*2, 2048)
        self.linear5 = nn.Linear(2048, 512)
        self.linear6 = nn.Linear(512, 7)

    def forward(self, X):
        o1 = F.relu(self.conv1(X))
        o2 = F.relu(self.conv2(o1))
        o4 = o2.view(-1, 256*3*2)
        o5 = F.relu(self.linear4(o4))
        o6 = F.relu(self.linear5(o5))
        o7 = self.linear6(o6)
        return o7


class DQN_CNN_WIDE_PREDICTION(nn.Module):
    def __init__(self):
        super(DQN_CNN_WIDE_PREDICTION, self).__init__()
        self.conv1 = nn.Conv2d(2, 64, 3, stride=1)  # 7x6 -> 5x4
        self.conv2 = nn.Conv2d(64, 256, 3, stride=1)  # 5x4 -> 3x2
        self.linear4 = nn.Linear(256*3*2, 2048)
        self.linear5 = nn.Linear(2048, 512)
        self.linear6 = nn.Linear(512, 7)
        self.linear_done = nn.Linear(512, 7)
        self.linear_reward = nn.Linear(512, 7)

    def forward(self, X, prediction=False):
        o1 = F.relu(self.conv1(X))
        o2 = F.relu(self.conv2(o1))
        o4 = o2.view(-1, 256*3*2)
        o5 = F.relu(self.linear4(o4))
        o6 = F.relu(self.linear5(o5))
        o7 = self.linear6(o6)

        if prediction:
            done = F.sigmoid(self.linear_done(o6))
            reward = self.linear_reward(o6)
            return o7, done, reward

        return o7


class DQN_CNN_VERY_WIDE(nn.Module):
    def __init__(self):
        super(DQN_CNN_VERY_WIDE, self).__init__()
        self.conv1 = nn.Conv2d(2, 128, 3, stride=1)  # 7x6 -> 5x4
        self.conv2 = nn.Conv2d(128, 512, 3, stride=1)  # 5x4 -> 3x2
        self.linear4 = nn.Linear(512*3*2, 2048)
        self.linear5 = nn.Linear(2048, 2048)
        self.linear6 = nn.Linear(2048, 512)
        self.linear7 = nn.Linear(512, 7)

    def forward(self, X):
        o1 = F.relu(self.conv1(X))
        o2 = F.relu(self.conv2(o1))
        o4 = o2.view(-1, 512*3*2)
        o5 = F.relu(self.linear4(o4))
        o6 = F.relu(self.linear5(o5))
        o7 = F.relu(self.linear6(o6))
        o8 = self.linear7(o7)
        return o8


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
        o1 = F.relu(self.conv1(X))
        o2 = F.relu(self.conv2(o1))
        o2_flat = o2.view(-1, 128*3*2)
        X_flat = X.view(-1, 7*6*2)
        X_o2_concat = torch.cat((o2_flat, X_flat), dim=1)
        o3 = F.relu(self.linear3(X_o2_concat))
        o4 = F.relu(self.linear4(o3))
        o5 = self.linear5(o4)
        return o5


class DQN_SKIP_WIDE(nn.Module):
    def __init__(self):
        super(DQN_SKIP_WIDE, self).__init__()
        self.conv1 = nn.Conv2d(2, 64, 3, stride=1)  # 7x6 -> 5x4
        self.conv2 = nn.Conv2d(64, 256, 3, stride=1)  # 5x4 -> 3x2
        self.linear3 = nn.Linear(256*3*2 + 7*6*2, 2048)
        self.linear4 = nn.Linear(2048, 512)
        self.linear5 = nn.Linear(512, 7)

    def forward(self, X):
        o1 = F.relu(self.conv1(X))
        o2 = F.relu(self.conv2(o1))
        o2_flat = o2.view(-1, 256*3*2)
        X_flat = X.view(-1, 7*6*2)
        X_o2_concat = torch.cat((o2_flat, X_flat), dim=1)
        o3 = F.relu(self.linear3(X_o2_concat))
        o4 = F.relu(self.linear4(o3))
        o5 = self.linear5(o4)
        return o5
