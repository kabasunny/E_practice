class ReLU:
    def __init__(self):
        # マスクを初期化
        self.mask = None

    def forward(self, x):
        # 入力xが0以下の場所を示すマスクを作成
        self.mask = (x <= 0)
        # 入力xのコピーを作成
        out = x.copy()
        # マスクがTrueの位置を0に設定
        out[self.mask] = 0
        # 出力を返す
        return out
    
    def backward(self, dout):
        # マスクがTrueの位置の勾配doutを0に設定
        dout[self.mask] = 0
        # 変換された勾配をdxに設定
        dx = dout
        # 変換された勾配を返す
        return dx
