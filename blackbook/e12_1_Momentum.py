import numpy as np


# 最適化手法 : モーメンタム
class Momentum:
    # 学習率を0.01、モーメンタム項を0.9で初期化
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None  # ベロシティ用のインスタンス変数の初期化

    # パラメータの更新、パラメータと勾配を受け取る
    def update(self, params, grads):
        # 初回ベロシティが無い状態
        if self.v is None:
            # 空の辞書を代入
            self.v = {}
            # 各パラメータのキーと値を順に取得、.items()は辞書のすべてのキーと値のペアを返すメソッド
            for key, val in params.items():
                # valと同じ形状のゼロ配列を生成
                self.v[key] = np.zeros_like(val)
        # パラメータの各キーを順に取得、.keys()は辞書のすべてのキーを返すメソッド
        for key in params.keys():
            # ベロシティを各勾配で更新
            self.v[key] = self.momentum * self.v[key] - self.lr * grads[key]
            # 各パラメータを各勾配で更新
            params[key] += self.v[key]
