import numpy as np


# 最適化手法 : SGD : 確率的勾配降下法
class SGD:
    # 学習率を0.01で初期化
    def __init__(self, lr=0.01):
        self.lr = lr

    # パラメータと勾配を受け取り、パラメータの更新
    def update(self, params, grads):
        # パラメータの各キーを順に取得、.keys()は辞書のすべてのキーを返すメソッド
        for key in params.keys():
            # 各パラメータを各勾配で更新
            params[key] -= self.lr * grads[key]
