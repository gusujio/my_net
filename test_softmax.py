import torch
import numpy as np
# loss = torch.nn.CrossEntropyLoss()
# input = torch.randn(1, 5, requires_grad=True)
# target = torch.randn(1, 5).softmax(dim=1)
# output = loss(input, target)
# output.backward()
# print(input.grad)
# print(output)
# print(target)
def il_log_loss(y_true, y_pred):
    """Если y_pred масив из двух вероятностей, то первая вероятность - вероятность о нуле?"""
    return np.mean(-(y_true * np.log(y_pred) + (1 - y_true) * np.log((1 - y_pred))))

softmax = torch.nn.Softmax(dim=-1)
print(softmax(torch.tensor([0, 1]).float()))
"""почему я получаю такое распредеоение, как будто у меня три класса, а не 2??"""
CE = torch.nn.CrossEntropyLoss()
# CE()
import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.metrics import accuracy_score, classification_report, log_loss

"""output линейное предсказание, так что может быть и больше 1 и тогда loss будет меньше
"""
output = Variable(torch.FloatTensor([3,  0])).unsqueeze(dim=0)
target = Variable(torch.LongTensor([1, 0])).unsqueeze(dim=0)
print(output, target)
criterion = nn.CrossEntropyLoss()
""" CrossEntropyLoss внутри себя уже использует Softmax
Input: первый параментр, просто линейное предсказание
я понял, так как софт макс и CE нужны для оценки выходного слоя, то значений  по количеству там столько,
сколько выходных нейронов, а я подавал несколько значений одного класса, думая, что он сработает как BCE
"""
loss = criterion(output, target.float())
print(loss)
print(-torch.log(torch.exp(output[0][0]) / (torch.sum(torch.exp(output)))))
print(log_loss([1, 0], [0.2689, 0.7311]))

print(torch.sum(torch.exp(output)), torch.exp(output), output)
# loss = torch.nn.CrossEntropyLoss()
# input = torch.randn(1, 5, requires_grad=True)
# target = torch.empty(1, 5, dtype=torch.long).random_(5)
# output = loss(input, target)
# output.backward()
# print(input.grad)
