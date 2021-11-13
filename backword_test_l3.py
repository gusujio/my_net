import torch ## попробуем применить backprop к нашей, вдург прокатит
from torch.autograd import Variable
from torch import autograd
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt # графики
from sklearn.metrics import accuracy_score, classification_report, log_loss



# print(il_loss.grad_fn)
# print(il_loss.grad_fn.next_functions[0][0])
# print(il_loss.grad_fn.next_functions[0][0].next_functions[0][0])
# print(il_loss.grad_fn.next_functions[0][0].next_functions[0][0].next_functions[0][0])
# print(il_loss.grad_fn.next_functions[0][0].next_functions[0][0].next_functions[0][0].next_functions[1][0])
# print(il_loss.grad_fn.next_functions[0][0].next_functions[0][0].next_functions[0][0].next_functions[1][0].next_functions[0][0])
# print(il_loss.grad_fn.next_functions[0][0].next_functions[0][0].next_functions[0][0].next_functions[1][0].next_functions[0][0].next_functions[0][0])


def il_softmax_2(pr1, pr2):
    """максимально простой софт макс"""
    pr1, pr2 = torch.exp(pr1), torch.exp(pr2)
    sum_exp = pr1 + pr2
    return pr1 / sum_exp, pr2 / sum_exp

def tr_log_loss_2(y_true1, y_pred1, y_pred2):
    """Если мы подаем одну вероятность -то это вероятность пренадлежать к 1"""
    """Если y_pred масив из двух вероятностей, то первая вероятность - вероятность к 0, а вторая к 1"""
    return y_true1 * torch.log(y_pred1) + (1-y_true1) * torch.log(y_pred2)


y_true = torch.tensor([0, 1]).unsqueeze(dim=0)

weight1 = torch.tensor([0.87, 0.4736, 0.8009], requires_grad=True)
input_x1 = torch.tensor([0.5229, 0.5605, 1])
pr_1 = torch.sum(weight1 * input_x1)

weight2 = torch.tensor([0.1182, -0.36, 0.1433], requires_grad=True)
input_x2 = torch.tensor([0.5229, 0.5605, 1])
pr_2 = torch.sum(weight2 * input_x2)

CE = torch.nn.CrossEntropyLoss()
inputs = torch.stack([pr_1, pr_2]).unsqueeze(dim=0)
inputs.requires_grad_(True)
tr_log = CE(inputs.float(), y_true.float())
print("Log_loss: ", tr_log.detach().numpy())
tr_log.backward()

sf_2, sf_1 = il_softmax_2(pr_2, pr_1)
softmax = torch.nn.Softmax(dim=-1)
true_softmax = softmax(torch.tensor([pr_1, pr_2]))
print(f"pr_1: {round(pr_1.tolist(), 4)} sf_1: {round(sf_1.tolist(), 4)}, soft_max {true_softmax[0]}\n\
pr_2: {round(pr_2.tolist(), 4)} sf_2: {round(sf_2.tolist(), 4)}, soft_max {true_softmax[1]}")


# y_false = torch.tensor(list(map(lambda x: int(not x), y_true)))
y_false = torch.tensor(list(map(lambda x: int(not x), y_true[0]))).unsqueeze(dim=0)
print(f"градиент от pytorch: {weight1.grad}, мой градиент {(sf_1-y_true[0][0])  * input_x1}, (sf_1-y_true): {sf_1-y_true[0][0]}")
print(f"градиент от pytorch: {weight2.grad}, мой градиент {(sf_2-y_false[0][0]) * input_x2}, (sf_2-y_false): {(sf_2-y_false[0][0])}")

"""все Ок, проверил как первый, так и второй случай"""

# y_true = torch.tensor([1, 0]).unsqueeze(dim=0)
# Log_loss:  0.19814573
# pr_1: 1.5213 soft_max_pr_1: 0.8202, soft_max 0.8202362656593323
# pr_2: 0.0033 soft_max_pr_2: 0.1798, soft_max 0.17976374924182892
# градиент от pytorch: tensor([-0.0940, -0.1008, -0.1798]), мой градиент tensor([-0.0940, -0.1008, -0.1798], grad_fn=<MulBackward0>), (sf_1-y_true): -0.17976373434066772
# градиент от pytorch: tensor([0.0940, 0.1008, 0.1798]), мой градиент tensor([0.0940, 0.1008, 0.1798], grad_fn=<MulBackward0>), (sf_2-y_false): 0.17976373434066772
# sf_1 0.8202362656593323, sf_2 0.17976373434066772

# y_true = torch.tensor([0, 1]).unsqueeze(dim=0)
# Log_loss:  1.7161119
# pr_1: 1.5213 sf_1: 0.8202, soft_max 0.8202362656593323
# pr_2: 0.0033 sf_2: 0.1798, soft_max 0.17976374924182892
# градиент от pytorch: tensor([0.4289, 0.4597, 0.8202]), мой градиент tensor([0.4289, 0.4597, 0.8202], grad_fn=<MulBackward0>), (sf_1-y_true): 0.8202362656593323
# градиент от pytorch: tensor([-0.4289, -0.4597, -0.8202]), мой градиент tensor([-0.4289, -0.4597, -0.8202], grad_fn=<MulBackward0>), (sf_2-y_false): -0.8202362656593323
