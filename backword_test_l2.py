import torch ## попробуем применить backprop к нашей, вдург прокатит

def il_sigma(x):
    return 1 / (1 + torch.exp(-x))

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
""""Второй слой"""
weight_l2_n1 = torch.tensor([0.81216873, 0.47997717, -0.6072152], requires_grad=True)
input_l2_n1 = torch.tensor([0.65335918, 0.35121428, 1.])
pr_l2_n1 = il_sigma(torch.sum(weight_l2_n1 * input_l2_n1)).unsqueeze(dim=0)
pr_l2_n1.requires_grad_(True)

weight_l2_n2 = torch.tensor([-0.9128707, 0.0202184, 0.83261985], requires_grad=True)
input_l2_n2 = torch.tensor([0.65335918, 0.35121428, 1.])
pr_l2_n2 = il_sigma(torch.sum(weight_l2_n2 * input_l2_n2)).unsqueeze(dim=0)
pr_l2_n2.requires_grad_(True)

""""Третий слой"""
weight1 = torch.tensor([0.87008726, 0.47360805, 0.80091075])
input_x1 = torch.stack([pr_l2_n1, pr_l2_n2, torch.tensor([1])])
pr_1 = torch.sum(weight1 * input_x1.T)

weight2 = torch.tensor([0.11827443, -0.36007898, 0.14335329])
input_x2 = torch.stack([pr_l2_n1, pr_l2_n2, torch.tensor([1])])
pr_2 = torch.sum(weight2 * input_x2.T)

"""Поиск ошибки"""
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


y_false = torch.tensor(list(map(lambda x: int(not x), y_true[0]))).unsqueeze(dim=0)
w9, w10 = weight1[0], weight1[1]
w11, w12 = weight2[0], weight2[1]
grad_l2_n1 = ((sf_1-y_true[0][0]) * w9 + (sf_2-y_false[0][0]) * w11) * pr_l2_n1 * (1-pr_l2_n1)
grad_l2_n2 = ((sf_1-y_true[0][0]) * w10 + (sf_2-y_false[0][0]) * w12) * pr_l2_n2 * (1-pr_l2_n2)
print(f"градиент от pytorch: {weight_l2_n1.grad}, мой градиент {grad_l2_n1 * input_l2_n1}, grad: {grad_l2_n1}")
print(f"градиент от pytorch: {weight_l2_n2.grad}, мой градиент {grad_l2_n2 * input_l2_n2}, grad: {grad_l2_n2}")
"""Все ок, два случая проверил"""

# y_true = torch.tensor([1, 0]).unsqueeze(dim=0)
# Log_loss:  0.19814573
# pr_1: 1.5214 sf_1: 0.8203, soft_max 0.8202503323554993
# pr_2: 0.0034 sf_2: 0.1797, soft_max 0.17974966764450073
# градиент от pytorch: tensor([-0.0220, -0.0118, -0.0337]), мой градиент tensor([-0.0220, -0.0118, -0.0337], grad_fn=<MulBackward0>), grad: tensor([-0.0337], grad_fn=<MulBackward0>)
# градиент от pytorch: tensor([-0.0241, -0.0130, -0.0369]), мой градиент tensor([-0.0241, -0.0130, -0.0369], grad_fn=<MulBackward0>), grad: tensor([-0.0369], grad_fn=<MulBackward0>)

# y_true = torch.tensor([0, 1]).unsqueeze(dim=0)
# Log_loss:  1.7161902
# pr_1: 1.5214 sf_1: 0.8203, soft_max 0.8202503323554993
# pr_2: 0.0034 sf_2: 0.1797, soft_max 0.17974966764450073
# градиент от pytorch: tensor([0.1005, 0.0540, 0.1538]), мой градиент tensor([0.1005, 0.0540, 0.1538], grad_fn=<MulBackward0>), grad: tensor([0.1538], grad_fn=<MulBackward0>)
# градиент от pytorch: tensor([0.1101, 0.0592, 0.1685]), мой градиент tensor([0.1101, 0.0592, 0.1685], grad_fn=<MulBackward0>), grad: tensor([0.1685], grad_fn=<MulBackward0>)
