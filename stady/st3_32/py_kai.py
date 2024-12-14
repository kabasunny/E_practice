import numpy as np

class MultiLayerNet:
    def __init__(self, input_size, hidden_size_list, output_size, weight_decay_lambda=0):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size_list = hidden_size_list
        self.hidden_layer_num = len(hidden_size_list)
        self.weight_decay_lambda = weight_decay_lambda

        # パラメータの初期化
        self.params = {}
        self.__init_weights()

        # レイヤの生成
        self.layers = {}
        self.__init_layers()

        # 出力層（ソフトマックスなど）の生成
        self.last_layer = SoftmaxWithLoss()

    def __init_weights(self):
        # 各層の重みとバイアスの初期化
        all_size_list = [self.input_size] + self.hidden_size_list + [self.output_size]
        for idx in range(1, len(all_size_list)):
            self.params['W' + str(idx)] = np.random.randn(all_size_list[idx-1], all_size_list[idx]) * 0.01
            self.params['b' + str(idx)] = np.zeros(all_size_list[idx])

    def __init_layers(self):
        # 各レイヤを生成
        for idx in range(1, self.hidden_layer_num + 1):
            self.layers['Affine' + str(idx)] = Affine(self.params['W' + str(idx)], self.params['b' + str(idx)])
            self.layers['ReLU' + str(idx)] = ReLU()
        idx = self.hidden_layer_num + 1
        self.layers['Affine' + str(idx)] = Affine(self.params['W' + str(idx)], self.params['b' + str(idx)])

    def loss(self, x, t):
        # 順伝播処理
        for layer in self.layers.values():
            x = layer.forward(x)
        loss = self.last_layer.forward(x, t)
        return loss

    def gradient(self, x, t):
        # forward: 順伝播を行い損失を計算
        self.loss(x, t)

        # backward: 逆伝播を行い勾配を計算
        dout = 1
        dout = self.last_layer.backward(dout)

        # すべてのレイヤに対して逆伝播を行う
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 勾配を設定
        grads = {}
        for idx in range(1, self.hidden_layer_num + 2):
            # 重み減衰（L2正則化）の勾配を計算
            weight_decay_gradient = self.weight_decay_lambda * self.params['W' + str(idx)]
            # 重みの勾配に重み減衰項を加算
            grads['W' + str(idx)] = self.layers['Affine' + str(idx)].dW + weight_decay_gradient
            # バイアスの勾配を設定
            grads['b' + str(idx)] = self.layers['Affine' + str(idx)].db

        return grads

# 以下に使用するレイヤや関数の定義を追加
class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b

    def forward(self, x):
        self.x = x
        out = np.dot(x, self.W) + self.b
        return out

    def backward(self, dout):
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        dx = np.dot(dout, self.W.T)
        return dx

class ReLU:
    def __init__(self):
        pass

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx

class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size
        return dx

def softmax(x):
    if x.ndim == 2:
        x = x - np.max(x, axis=1, keepdims=True)
        x = np.exp(x)
        x /= np.sum(x, axis=1, keepdims=True)
    elif x.ndim == 1:
        x = x - np.max(x)
        x = np.exp(x) / np.sum(np.exp(x))
    return x

def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    if t.size == y.size:
        t = t.argmax(axis=1)
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t])) / batch_size
