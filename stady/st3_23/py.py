import numpy as np

class RMSprop:
    def __init__(self, lr=0.01, decay_rate=0.99):
        # 学習率（learning rate）を設定
        self.lr = lr
        # 減衰率（decay rate）を設定
        self.decay_rate = decay_rate
        # 勾配の平方和を格納する変数（初期はNone）
        self.h = None

    def update(self, params, grads):
        # 初回の更新時に勾配の平方和の辞書を初期化
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                # パラメータと同じ形状のゼロ配列を作成
                self.h[key] = np.zeros_like(val)

        # 各パラメータを更新
        for key in params.keys():
            # 過去の勾配の平方和に減衰率を掛ける
            self.h[key] *= self.decay_rate
            # 現在の勾配の平方を加えて移動平均を更新
            self.h[key] += (1 - self.decay_rate) * grads[key] * grads[key]
            # パラメータを更新。勾配を勾配の平方和のルートで割り、学習率を掛ける
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)
