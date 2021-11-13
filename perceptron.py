import numpy as np
import pandas as pd
import pickle
import torch
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, log_loss

np.random.seed(0)

def il_softmax(mas_pred):
    mas_exp = np.exp(mas_pred)
    sum_exp = np.sum(mas_exp, axis=1)
    return mas_exp / [[i] for i in sum_exp]


def il_sigma(x):
    return 1 / (1 + np.exp(-x))

def tr_log_loss_2(y_true1, y_pred1, y_pred0):
    """Если мы подаем одну вероятность -то это вероятность пренадлежать к 1"""
    """Если y_pred масив из двух вероятностей, то первая вероятность - вероятность к 0, а вторая к 1"""
    return y_true1 * torch.log(y_pred1) + (1-y_true1) * torch.log(y_pred0)

def il_log_loss(y_true, y_pred):
    """Если y_pred масив из двух вероятностей, то первая вероятность - вероятность о нуле?"""
    return np.mean(-(y_true * np.log(y_pred) + (1 - y_true) * np.log((1 - y_pred))))
#     return np.mean(-((y_true * np.log(y_pred[:, 0]))[:,0] + (y_true * np.log(y_pred[:, 1]))[:,1]))


class Neuron:
    def __init__(self, sigma=False):
        self.sigma = sigma
        self.weights = 0
        self.result = None
        self.x = None
        self.weights_grad = None

    def __repr__(self):
        return 'Neuron'

    def pred(self, args):
        self.x = np.array([np.append(i, 1) for i in args])
        if isinstance(self.weights, int):
            self.weights = np.random.rand(self.x.shape[1]) + np.random.randint(-1, 1, self.x.shape[1]) # от -1 до 1

        if self.sigma:
            self.result = il_sigma(np.dot(self.x, self.weights))
        else:
            self.result = np.dot(self.x, self.weights)
        return self.result


class Layers:
    def __init__(self, n_neuron=None, sigma=False):
        self.sigma = sigma
        self.layer = n_neuron

    def __repr__(self):
        return f"Layer with {len(self.layer)} n"

    @property
    def layer(self):
        return self.__layer

    @layer.setter
    def layer(self, n_neuron):
        self.__layer = []
        for i in range(n_neuron):
            self.__layer.append(Neuron(self.sigma))

    def __getitem__(self, item):
        if item < len(self.layer):
            return self.layer[item]
        else:
            raise IndexError('Индекс за пределами слоя')

    def __iter__(self):
        return iter(self.layer)

    def __len__(self):
        return len(self.layer)

    def pred(self, args):
        output = []
        for neuron in self:
            output.append(neuron.pred(args))
        return np.array(output).T


class Net:
    table_param = {}
    mas_logloss = []

    def __init__(self, n_hidden_layers, n_neurons=2, debug=False, **kwargs):
        self.net = n_hidden_layers, n_neurons
        self.debug = debug
        self.classification = kwargs.get('classification', True)
        self.function_error = kwargs.get('function_error', il_log_loss)
        self.function_activation = kwargs.get('function_activation', il_softmax)
        self.n_epochs = kwargs.get('n_epochs', 2000)
        self.lr = kwargs.get('lr', 0.01)
        self.min_err = kwargs.get('min_err', 0.05)

    def __iter__(self):
        """можем итерироваться по слоям"""
        return iter(self.net)

    @property
    def net(self):
        return self.__net

    @net.setter
    def net(self, value: tuple):
        """ Создаем нашу сетку
        Первый слой содержит всегда один нейрон, с линейной функцией
        На последний нейрон пока не навешиваем функцию активации, так как будет применем softmax
        :value - содержит кол-во скрытых слоев и кол-во нейронов в слоях
        """
        n_hidden_layers, n_neurons = value
        if n_hidden_layers != 2:
            raise ValueError("Недопустимое значение для n_hidden_layers")

        self.__net = [Layers(1)]
        for i in range(n_hidden_layers):
            self.__net.append(Layers(n_neurons, True))

        self.__net.append(Layers(2))

    def forward(self, vec_data):
        """ Нужно для прямого распространения ошибки
        Реализована взаимосвь между слоями
        по сути, это функция predict
        :return vec_data - вероятность принадлежности к одному из классов
        vec_data[0] - принадлежность к 1, vec_data[1] -  принадлежность к 0
        """
        for layer in self:
            vec_data = layer.pred(vec_data)

        if self.classification:
            vec_data = self.function_activation(vec_data)
            self.net[-1][0].result = vec_data[:, 0]
            self.net[-1][1].result = vec_data[:, 1]

        return vec_data

    def on_in_man(self, mas):
        """ переводим одномерный масив в двухмерные, нужно для удобного переумножения"""
        return np.array([[i] for i in mas])

    def create_table(self, neuron, name):
        if name not in self.table_param:
            self.table_param[name] = {'weights': [], 'x': [], 'res': [], 'res_m_tar': [], 'dir_res': []}
        self.table_param[name]['weights'].append(neuron.weights)
        self.table_param[name]['x'].append(neuron.x)
        self.table_param[name]['res'].append(neuron.result)

    def grad_l3(self, target, indx_neuron):
        """Беру производну по функции ошибки кросс энтропии
        Функция активация- на последнем слое - Softmax"""
        if indx_neuron == 0:
            res_m_tar = self.table_param['l3']['res'][-1] - target
        else:
            un_target = np.array(list(map(lambda x: int(not x), target)))
            res_m_tar = self.table_param['l3']['res'][-1] - un_target
        self.table_param['l3']['res_m_tar'].append(res_m_tar)
        return res_m_tar

    def grad_l2(self, indx_neuron):
        weights = self.on_in_man(np.array(self.table_param['l3']['weights'])[:, indx_neuron])
        sums = np.sum(weights * self.table_param['l3']['res_m_tar'], axis=0)
        res_neur = np.array(self.table_param['l2']['res'])
        dir_res = res_neur[indx_neuron] * (1 - res_neur[indx_neuron])
        self.table_param['l2']['dir_res'].append(dir_res)
        return sums * dir_res

    def grad_l1(self, indx_neuron):
        weights_l3 = np.array([[[j] for j in i] for i in np.array(self.table_param['l3']['weights'])[:, :-1]])
        weights_l2 = np.array([self.on_in_man(np.array(self.table_param['l2']['weights'])[:, indx_neuron]).tolist()] * 2)
        res_m_tar = weights_l3 * self.table_param['l2']['dir_res']
        self.table_param['l1']['res_m_tar'].append(res_m_tar) ### ??? то что голубое
        ko = np.sum(weights_l2 * res_m_tar, axis=1)
        sums = np.sum(ko * self.table_param['l3']['res_m_tar'], axis=0)

        res_neur = np.array(self.table_param['l1']['res'])
        dir_res = res_neur[indx_neuron] * (1 - res_neur[indx_neuron])
        self.table_param['l1']['dir_res'].append(dir_res)
        return sums * dir_res

    def grad_l0(self):
        gr_1 = np.array(self.table_param['l2']['x'])[:, :, :-1]
        gr_1 = gr_1 * (1 - gr_1)
        gr = self.on_in_man(np.array(self.table_param['l2']['weights'])[:, :-1]) * gr_1## зеленное
        rd = np.array(self.table_param['l1']['weights'])[:, 0] ## красное
        or_skob = [[j[0] + j[1] for j in i] for i in gr * rd]  ## орандж скобки
        kr_skob_0 = (np.array([or_skob] * 2) * self.table_param['l1']['res_m_tar'][0]).T
        kr_skob = [[j[0] + j[1] for j in i.T] for i in kr_skob_0]
        br = np.array(self.table_param['l3']['res_m_tar']).T ## коричневое
        return np.sum(br * kr_skob, axis=1) * 10

    def backprop(self, target):
        """ Обратное распространение оишбки, находим производные по каждому слою
        """
        name_layers = ['l3', 'l2', 'l1', 'l0']
        for layer, name in zip(reversed(self.net), name_layers):
            for indx_neuron in range(len(layer)):
                self.create_table(layer[indx_neuron], name)
                if name == 'l3':
                    res_grad = self.grad_l3(target, indx_neuron)
                elif name == 'l2':
                    res_grad = self.grad_l2(indx_neuron)
                elif name == 'l1':
                    res_grad = self.grad_l1(indx_neuron)
                else:
                    res_grad = self.grad_l0()
                layer[indx_neuron].weights_grad = np.mean(layer[indx_neuron].x.T * res_grad, axis=1)

    def step(self, lr=0.01):
        for layer in self:
            for indx_neuron in range(len(layer)):
                layer[indx_neuron].weights = layer[indx_neuron].weights - lr * layer[indx_neuron].weights_grad
        self.table_param.clear()

    def fit(self, data, target):
        if isinstance(data, pd.DataFrame):
            data = data.values
        for epoch in range(1, self.n_epochs + 1):
            pred = self.forward(data)
            loss = self.function_error(target, pred[:, 0])
            self.backprop(target)
            self.step(self.lr)
            if self.debug:
                print(f"epoch {epoch}/{self.n_epochs} - loss: {loss}")
            """#An historic of the metric obtained during training."""
            self.mas_logloss.append(loss)
            """ Early stopping"""
            if loss <= self.min_err:
                break

    def predict(self, data):
        pred = self.forward(data)[:, 0]
        return [1 if i >= 0.5 else 0 for i in pred]

    def predict_proba(self, data):
        return self.forward(data)

    def dump_net(self):
        with open('my_net.pickle', 'wb') as f:
            pickle.dump(self, f)

    def load_net(self):
        with open('my_net.pickle', 'rb') as f:
            model = pickle.load(f)
        return model








# # print(np.random.rand(18))
#
# data = pd.read_csv("data.csv", header=None)
# tagter = np.array([1 if i == 'M' else 0 for i in data[1]])
# tagter[0] = 1
# data = data.drop([0, 1], axis=1)
# # data = StandardScaler().fit_transform(data)
# data = StandardScaler().fit_transform(data)[:, 2:4]

# data = data # временно
# data = np.array([np.array([10, 2]), np.array([30, 4]), np.array([5, 6])])
# tagter = np.array([1, 1, 0])
"""тестирование Net fit"""
# net = Net(2, 2, True, lr=0.001, min_err=0.05, n_epochs=100)
# net.fit(data, tagter)
# print(classification_report(tagter, net.predict(data)))
# print(net.forward(data), tagter)
# pd.DataFrame(net.mas_logloss).plot()
# plt.xlabel('times')
# plt.ylabel('log loss values')
# plt.show()





# data = np.array([np.array([1, 2]), np.array([5, 6]), np.array([9, 10])])
# k = [[[j] for j in i] for i in data]
# print(k[0])

# """тестирование Net"""
# data = np.array([np.array([1, 2]), np.array([5, 6]), np.array([9, 10])])
# # print(data)
# net = Net(2, 2)
# res = net.forward(data)
# # bc = net.backprop([1,1,0])
# print(res)

# net = Net(2, 2)
# data = np.array([np.array([1, 2]), np.array([5, 6]), np.array([9, 10])])
# net.fit(data, np.array([1,1,0]))
# pd.DataFrame(net.mas_logloss).plot(title='th0')
# plt.xlabel('times')
# plt.ylabel('values')
# plt.show()

# print(res.shape, data.shape, res, sep='\n')
# print(il_log_loss(np.array([1,0,1]), res))
# print(np.concatenate(([1,2], [3,4])))
# print(data * np.array([1, 2]))

# """тестирование Layers"""
# data = np.array([np.array([1, 2, 3, 4]), np.array([5, 6, 7, 8])])
# la1 = Layers(1)
# la2 = Layers(4)
# la3 = Layers(4)
# la4 = Layers(2)
# print(la3.pred(data))

# print(la.pred(data, True))

# """тестирование Neuron.pred"""

# ner = Neuron()
# # arg = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
# arg = np.array([np.array([1, 2, 3, 4]), np.array([5, 6, 7, 8])])
# print(ner.pred(arg))
# print(ner.weights)
# # [11. 27.]

# """тестирование il_log_loss"""
# y_true = np.array([1, 0, 1, 0])
# y_pred = np.array([0.8, .5, .2, .1])
# orig_log_loss = log_loss(y_true, y_pred)
# pred_log_loss = il_log_loss(y_true, y_pred)
# print(f" orig_log_loss: {orig_log_loss}\n pred_log_loss: {pred_log_loss}")

# """тестирование softmax"""

# print(softmax(arg), sum(softmax(arg)))




# loss = torch.nn.CrossEntropyLoss()
# input = torch.randn(2, 5, requires_grad=True)
# target = torch.randn(2, 5).softmax(dim=1)
# output = loss(input, target)
# output.backward()
# print('input: ', input,'\n', 'target', target)
# print(output)
# print(cross_entropy(input, target))



