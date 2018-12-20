import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

from torch.autograd import Variable

import numpy as np
import matplotlib.pyplot as plt


transform = transforms.ToTensor()

# Set parameters
BATCH_SIZE = 16

# Load Dataset
trainset = torchvision.datasets.MNIST('/tmp', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=False)

testset = torchvision.datasets.MNIST('/tmp', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)

# Visualize data
def show_batch(batch):
    im = torchvision.utils.make_grid(batch)
    plt.imshow(np.transpose(im.numpy(), (1, 2, 0))) # Original shape is (N, C, H, W)

dataiter = iter(testloader)
images, labels = dataiter.next()

print('Labels:', labels)
print('Batch shape:', images.size())
show_batch(images)
plt.show()

# Create the model
class SequentialMNIST(nn.Module):
    def __init__(self):
        super(SequentialMNIST, self).__init__()
        self.linear1 = nn.Linear(28*28, 256)
        self.lstm1 = nn.LSTM(input_size=256, hidden_size=256, num_layers=1)
        self.linear2 = nn.Linear(256, 10)

    def forward(self, x, hidden):
        o1_relu = F.relu(self.linear1(x.view(-1, 28*28)))
        o1_lstm, hidden = self.lstm1(o1_relu.view(1, -1, 256), hidden)
        o2_relu = F.relu(o1_lstm.view(-1, 256))
        y_pred = self.linear2(o2_relu)
        return y_pred, hidden


model = SequentialMNIST()

def train(model, trainloader, criterion, optimizer, n_epochs=1):
    Loss = []
    hidden_state = (torch.zeros([1, BATCH_SIZE, 256], dtype=torch.float),
                    torch.zeros([1, BATCH_SIZE, 256], dtype=torch.float))
    labels_hist = -1*torch.ones([BATCH_SIZE, 9], dtype=torch.float)
    tracks = torch.zeros([BATCH_SIZE, 10], dtype=torch.float)
    for t in range(n_epochs):
        for i, data in enumerate(trainloader):
            inputs, labels = data
            inputs, labels = Variable(inputs), Variable(labels)

            for b in range(labels.shape[0]):
                tracks[b, 0] = 0
                for n in range(len(labels_hist[b])):
                    if labels[b].numpy() == labels_hist[b, n].numpy():
                        tracks[b, 0] = 1
                tracks[b, 1:] = torch.FloatTensor([k.numpy() == labels[b].numpy() for k in labels_hist[b, :]])

            optimizer.zero_grad()

            outputs, hidden_state = model(inputs, hidden_state)
            print(outputs.size(0), tracks.size(0))
            loss = criterion(outputs.long(), Variable(tracks).long()) # Compute the loss
            loss.backward() # Compute the gradient for each variable
            optimizer.step() # Update the weights according to the computed gradient

            labels_hist = [labels, labels_hist[2:]]

            if not i % 500:
                print('Epoch: ', t, ' Batch: ', i, 'Loss: ', loss.data.item())

            Loss.append(loss.data.item())

    return Loss

#outputs = model(Variable(images))

def predict(model, images):
    outputs = model(Variable(images))
    _, predicted = torch.max(outputs.data, 1)
    return predicted


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
Loss = train(model, testloader, criterion, optimizer, n_epochs=1)

plt.plot(range(0, len(Loss)), Loss)
plt.show()

def test(model, tloader, n):
    correct = 0
    for data in testloader:
        inputs, labels = data
        pred = predict(model, inputs)
        correct += (pred == labels).sum()

        #print('Prediction: ', pred)
        #print('Labels: ', labels)
        #print('Size: ', len(labels), 'Error: ', (pred == labels).sum().numpy())
    return 100 * correct / n

print('Accuracy: ', test(model, testloader, len(testset)).numpy())

