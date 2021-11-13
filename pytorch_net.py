import torch ## попробуем применить backprop к нашей, вдург прокатит
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt # графики
from sklearn.metrics import accuracy_score, classification_report, log_loss


data = pd.read_csv("data.csv", header=None)
target = torch.tensor([1 if i == 'M' else 0 for i in data[1]])
data = data.drop([0, 1], axis=1)
data = StandardScaler().fit_transform(data)
data = torch.tensor(data).float()


class ClassificationNet(torch.nn.Module):
    def __init__(self, n_hidden_neurons):
        super(ClassificationNet, self).__init__()# инициируем родительский объект.
        self.fc1 = torch.nn.Linear(30, n_hidden_neurons)
        self.fc2 = torch.nn.Linear(n_hidden_neurons, n_hidden_neurons)
        self.act1 = torch.nn.Sigmoid()
        self.fc3 = torch.nn.Linear(n_hidden_neurons, n_hidden_neurons)
        self.act2 = torch.nn.Sigmoid()
        self.fc4 = torch.nn.Linear(n_hidden_neurons, n_hidden_neurons)
        self.act3 = torch.nn.Softmax(dim=1)
        # self.act3 = torch.nn.Sigmoid()


    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.act1(x)
        x = self.fc3(x)
        x = self.act2(x)
        x = self.fc4(x)
        x = self.act3(x)
        return x


net = ClassificationNet(2)
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)


loss = torch.nn.CrossEntropyLoss()

for epoch_index in range(2500):
    optimizer.zero_grad()
    y_pred = net.forward(data)
    loss_value = loss(y_pred, target)
    loss_value.backward()
    optimizer.step()


ans = net.forward(data)
print('value: ', loss(net.forward(data), target))
pred = [1 if i > 0.5 else 0for i in ans[:, 1]]
print("accuracy_score ", accuracy_score(target, pred))

print(net.forward(data))



