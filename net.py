import os
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

if torch.cuda.is_available():
    device = torch.device('cuda:0')
    print('running on GPU')
else:
    device = torch.device('cpu')
    print('running on CPU')

class TrainingDataGetter():
    TEXT = os.path.join('Training Data', 'Text')
    NOTEXT = os.path.join('Training Data', 'No Text')
    LABELS = {TEXT: 0, NOTEXT: 1}

    training_data = []

    def make_training_data(self):
        for label in self.LABELS:
            dir = os.listdir(label)
            for fileIndex in tqdm(range(0, len(os.listdir(label)), 1)):
                file = dir[fileIndex]
                try:
                    path = os.path.join(label, file)

                    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                    img = cv2.resize(img, (128, 64))

                    self.training_data.append([np.array(img), np.eye(2)[self.LABELS[label]]])

                except Exception as e:
                    pass

        np.random.shuffle(self.training_data)
        np.save("training_data.npy", self.training_data)

class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.convLayers = nn.ModuleList([nn.Conv2d(1, 32, 4), nn.Conv2d(32, 64, 4), nn.Conv2d(64, 128, 4), nn.Conv2d(128, 256, 4)])

        x = torch.randn(128, 64).view(-1, 1, 128, 64)

        self._to_linear = None
        self.convs(x)

        self.fullyConnectedLayers = nn.ModuleList([nn.Linear(self._to_linear, 512), nn.Linear(512, 2)])

    def convs(self, x):
        for layer in self.convLayers:
            x = F.max_pool2d(F.relu(layer(x)), (2, 2))

        if self._to_linear is None:
            self._to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]

        return x

    def forward(self, x):
        x = self.convs(x)

        x = x.view(-1, self._to_linear)

        for i in range(len(self.fullyConnectedLayers) - 1):
            x = F.relu(self.fullyConnectedLayers[i](x))

        x = self.fullyConnectedLayers[-1:][0](x)
        return F.softmax(x, dim = 1)

    def load_weights(self, file):
        trained_data = np.load('trained_data.npy', allow_pickle = True)
        for layerIndex in range(len(self.convLayers)):
            self.convLayers[layerIndex].weight = trained_data[layerIndex][0]
            self.convLayers[layerIndex].bias = trained_data[layerIndex][1]

        for layerIndex in range(len(self.fullyConnectedLayers)):
            self.fullyConnectedLayers[layerIndex].weight = trained_data[len(self.convLayers) + layerIndex][0]
            self.fullyConnectedLayers[layerIndex].bias = trained_data[len(self.convLayers) + layerIndex][1]

if __name__ == '__main__':
    net = Net().to(device)

def train(net):
    epoch = 0
    accuracy = test(net)
    print('Initial Accuracy:', int(accuracy * 100))
    while epoch < EPOCHS:
        epoch += 1
        for i in tqdm(range(0, len(train_X), BATCH_SIZE)):
            batch_X = train_X[i:i+BATCH_SIZE].view(-1, 1, 128, 64).to(device)
            batch_y = train_y[i:i+BATCH_SIZE].to(device)

            optimizer.zero_grad()

            outputs = net(batch_X)

            loss = loss_function(outputs, batch_y)
            loss.backward()
            optimizer.step()

        print(f'Epoch: {epoch}. Loss:{loss}')

        if epoch % 5 == 0:
            accuracy = test(net)
            print('Accuracy:', int(accuracy * 100))

        trained_data = []
        for layer in net.convLayers:
            trained_data.append([layer.weight, layer.bias])
        for layer in net.fullyConnectedLayers:
            trained_data.append([layer.weight, layer.bias])

        np.save('trained_data.npy', trained_data)


def test(net):
    print('testing')
    correct = 0
    total = 0

    with torch.no_grad():
        for i in tqdm(range(len(test_X.to(device)))):
            real_class = torch.argmax(test_y[i].to(device))
            net_out = net(test_X[i].view(-1, 1, 128, 64).to(device))[0]
            predicted_class = torch.argmax(net_out)
            if predicted_class == real_class:
                correct += 1
            total += 1

    return correct/total

if __name__ == '__main__':
    try:
        with torch.no_grad():
            net.load_weights('trained_data.npy')

        try:
            print('getting training data')
            training_data = np.load('training_data.npy', allow_pickle = True)

        except:
            print('making training data')
            textornotext = TrainingDataGetter()
            textornotext.make_training_data()
            training_data = textornotext.training_data
            del(textornotext)

        optimizer = optim.Adam(net.parameters(), lr=0.001)
        loss_function = nn.MSELoss()

        VAL_PCT = 0.1

        print('getting inputs')
        X = torch.Tensor(np.array([i[0] for i in training_data])).view(-1, 128, 64)/255.0
        print('getting outputs')
        y = torch.Tensor(np.array([i[1] for i in training_data]))

        del(training_data)

        val_size = int(len(X) * VAL_PCT)

        train_X = X[:-val_size]
        train_y = y[:-val_size]

        test_X = X[-val_size:]
        test_y = y[-val_size:]

        BATCH_SIZE = 32
        EPOCHS = 30

        train(net)

    except Exception as e:
        print(e)
        try:
            print('getting training data')
            training_data = np.load('training_data.npy', allow_pickle = True)

        except:
            print('making training data')
            textornotext = TrainingDataGetter()
            textornotext.make_training_data()
            training_data = textornotext.training_data
            del(textornotext)

        optimizer = optim.Adam(net.parameters(), lr=0.001)
        loss_function = nn.MSELoss()

        VAL_PCT = 0.1

        print('getting inputs')
        X = torch.Tensor(np.array([i[0] for i in training_data])).view(-1, 128, 64)/255.0
        print('getting outputs')
        y = torch.Tensor(np.array([i[1] for i in training_data]))

        del(training_data)

        val_size = int(len(X) * VAL_PCT)

        train_X = X[:-val_size]
        train_y = y[:-val_size]

        test_X = X[-val_size:]
        test_y = y[-val_size:]

        BATCH_SIZE = 32
        EPOCHS = 30

        train(net)
