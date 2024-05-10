import numpy as np


# 最適化手法 : AdaGrad
class AdaGrad:
    # 学習率を0.01で初期化
    def __init__(self, lr=0.01):
        self.lr = lr
        self.h = None  # 二乗和を保持する変数

    # パラメータの更新、パラメータと勾配を受け取る
    def update(self, params, grads):
        # 初回、過去の二乗和が無い状態
        if self.h is None:
            # 空の辞書を代入
            self.h = {}
            # 各パラメータのキーと値を順に取得、.items()は辞書のすべてのキーと値のペアを返すメソッド
            for key, val in params.items():
                # valと同じ形状のゼロ配列を生成
                self.h[key] = np.zeros_like(val)

        # パラメータの各キーを順に取得、.keys()は辞書のすべてのキーを返すメソッド
        for key in params.keys():
            # 二乗和
            self.h[key] += grads[key] * grads[key]
            # 学習率をhの平方根を用いてパラメータを更新
            # 1e-7はルートの中か外か、どちらでもよい
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)
