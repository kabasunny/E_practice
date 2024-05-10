import numpy as np


# 最適化手法 : ネステロフのモーメンタム : NAG
class Nesterov:
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
            # 各パラメータを各勾配で更新
            params[key] += self.momentum * self.momentum * self.v[key]
            params[key] -= (1 + self.momentum) * self.lr * grads[key]
            # params[key] = params[key] + self.momentum * self.momentum * self.v[key] - (1 + self.momentum) * self.lr * grads[key]

            # 次回のベロシティを各勾配で更新
            self.v[key] *= self.momentum
            self.v[key] -= self.lr * grads[key]
            # self.v[key] = self.momentum * self.v[key] - self.lr * grads[key]
        # ネステロフのモーメンタムでは、「先読み」を行い、現在のベロシティ方向に少し「先読み」した位置での勾配を計算し、その情報を用いてパラメータを更新する。
        # 上記の実装では、次回のベロシティを今回の最後に更新する。初回の更新については、ネステロフのモーメンタムも通常のモーメンタムも同じ。
