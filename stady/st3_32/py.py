class MultiLayerNet:
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
