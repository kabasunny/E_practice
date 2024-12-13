class Network:
    def __init__(self, input_size, hidden_size):
        # 入力サイズと隠れ層サイズに基づいてパラメータを初期化
        self.x = None
        self.params = {
            "w": np.random.randn(input_size, hidden_size),  # 重み行列の初期化（ランダム）
            "b": np.zeros(hidden_size)                      # バイアスベクトルの初期化（ゼロ）
        }
        self.grads = {
            "w": None,  # 重み勾配の初期化
            "b": None   # バイアス勾配の初期化
        }

    def forward(self, x):
        # 順伝播計算
        self.x = x  # 入力データを保存
        return np.dot(x, self.params["w"]) + self.params["b"]
        # 入力データ x と重み w のドット積にバイアス b を加えて出力を計算

    def backward(self, dout): # dout（デルタ・アウト）は、出力層から伝達される誤差情報で、ある層の出力に対する損失関数の勾配
        # 逆伝播計算
        self.grads["w"] = np.dot(self.x.T, dout)
        # 入力データ x の転置行列と出力誤差 dout のドット積を計算して重み勾配を求める
        self.grads["b"] = np.sum(dout, axis=0)
        # 出力誤差 dout をすべてのサンプルに対して合計してバイアス勾配を求める
        return np.dot(dout, self.params["w"].T)
        # 出力誤差 dout と重み w の転置行列のドット積を計算して前の層に伝達する誤差を求める
