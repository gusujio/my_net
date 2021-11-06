import numpy as np
import pandas as pd


def il_softmax(mas_pred):
    mas_exp = np.exp(mas_pred)
    sum_exp = sum(mas_exp)
    return mas_exp / sum_exp


def il_sigma(x):
    return 1 / (1 + np.exp(-x))


def il_log_loss(y_true, y_pred):
    return np.mean(-y_true * np.log(y_pred[:, 0]) - (1-y_true) * np.log(1-y_pred[:, 1]))


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
            self.weights = np.random.randint(1, 20, self.x.shape[1])
            # self.weights = np.ones(self.x.shape[1])
            # self.weights = np.random.rand(self.x.shape[1])
            # self.weights = np.random.rand(args2.shape[1]) от -1 до 1

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
        return f"Layers with {len(self.layer)} n"

    @property
    def layer(self):
        return self.__layer

    @layer.setter
    def layer(self, n_neuron):
        self.__layer = []
        for i in range(n_neuron):
            self.__layer.append(Neuron(self.sigma))

    def __getitem__(self, item):
        if 0 <= item < len(self.layer):
            return self.layer[item]
        else:
            raise IndexError('Индекс за пределами слоя')

    def __iter__(self):
        return iter(self.layer)

    def __len__(self):
        return len(self.layer)

    def pred(self, args):
        output = []
        for neuron in self.layer:
            output.append(neuron.pred(args))
        return np.array(output).T


class Net:
    table_param = {}

    def __init__(self, n_hidden_layers, n_neurons=2, **kwargs):
        self.net = n_hidden_layers, n_neurons
        self.classification = kwargs.get('classification', True)
        self.function_error = kwargs.get('function_error', il_log_loss)
        self.function_activation = kwargs.get('function_activation', il_softmax)

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
            self.__net.append(Layers(n_neurons)) # """временно!!! """
            # self.__net.append(Layers(n_neurons, True))

        self.__net.append(Layers(2))

    def feedforward(self, vec_data):
        """ Нужно для прямого распространения ошибки
        Реализована взаимосвь между слоями
        по сути, это функция predict
        """
        if isinstance(vec_data, pd.DataFrame):
            vec_data = vec_data.values

        for layer in self.net:
            vec_data = layer.pred(vec_data)

        if self.classification:
            for i in range(len(vec_data)):
                vec_data[i] = self.function_activation(vec_data[i])
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

    def grad_l3(self, target):
        res_m_tar = self.table_param['l3']['res'][-1] - target
        self.table_param['l3']['res_m_tar'].append(res_m_tar)
        return res_m_tar

    def grad_l2(self, indx_neuron):
        weights = self.on_in_man(np.array(self.table_param['l3']['weights'])[:, indx_neuron])
        sums = np.sum(weights * self.table_param['l3']['res_m_tar'], axis=0)
        res_neur = np.array(self.table_param['l2']['res'])
        dir_res = res_neur[indx_neuron] * (1 - res_neur[indx_neuron])
        self.table_param['l2']['dir_res'].append(dir_res)
        return sums[indx_neuron] * dir_res

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
        return sums[indx_neuron] * dir_res

    def grad_l0(self):
        gr_1 = np.array(self.table_param['l2']['x'])[:, :, :-1]
        gr_1 = gr_1 * (1 - gr_1)
        gr = self.on_in_man(np.array(self.table_param['l2']['weights'])[:, :-1]) * gr_1## зеленное
        rd = np.array(self.table_param['l1']['weights'])[:, 0] ## красное
        or_skob = [[j[0] + j[1] for j in i] for i in gr * rd]  ## орандж скобки
        kr_skob_0 = (np.array([or_skob] * 2) * self.table_param['l1']['res_m_tar'][0]).T
        kr_skob = [[j[0] + j[1] for j in i.T] for i in kr_skob_0]
        br = np.array(self.table_param['l3']['res_m_tar']).T ## коричневое
        return np.sum(br * kr_skob, axis=1)

    def backprop(self, target):
        """ Ошибка в том, что софт макс применяется после выполнений последнего ответа
        """
        name_layers = ['l3', 'l2', 'l1', 'l0']
        for layer, name in zip(reversed(self.net), name_layers):
            for indx_neuron in range(len(layer)):
                self.create_table(layer[indx_neuron], name)
                if name == 'l3':
                    res_grad = self.grad_l3(target)
                elif name == 'l2':
                    res_grad = self.grad_l2(indx_neuron)
                elif name == 'l1':
                    res_grad = self.grad_l1(indx_neuron)
                else:
                    res_grad = self.grad_l0()
                layer[indx_neuron].weights_grad = np.mean(layer[indx_neuron].x.T * res_grad, axis=1)

    def step(self):
        pass

    def fit(self, data, target):
        self.n_epochs = 2000
        for epoch in range(self.n_epochs):
            pred = self.feedforward(data)
            loss = self.function_error(target, pred)
            self.backprop(target)







np.random.seed(0)
# print(np.random.rand(18))

# data = pd.read_csv("data.csv", header=None)
# tagter = data[1]
# data = data.drop([0, 1], axis=1)
# print(type(data))
# print(data.values)
#

# data = np.array([np.array([1, 2]), np.array([5, 6]), np.array([9, 10])])
# k = [[[j] for j in i] for i in data]
# print(k[0])

"""тестирование Net"""
data = np.array([np.array([1, 2]), np.array([5, 6]), np.array([9, 10])])
# print(data)
net = Net(2, 2)
res = net.feedforward(data)
bc = net.backprop([1,1,0])
# print(res)
print(bc)

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



