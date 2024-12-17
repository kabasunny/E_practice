import numpy as np
import matplotlib.pyplot as plt

# 自己情報量についての説明
print("自己情報量（Self-Information）は、あるイベントが発生したときの情報の量を示す尺度で、")
print("イベントの発生確率 P(x) が低いほど、そのイベントが発生したときの驚きの度合いが大きくなり、自己情報量も高くなり、")
print("具体的には、自己情報量は次の式で定義される： I(x) = -log2(P(x))")
print("このグラフでは、横軸に発生確率 P(x)、縦軸に自己情報量 I(x) を取り、確率が高いほど自己情報量が低く、確率が低いほど自己情報量が高くなる")

# 発生確率の範囲を定義
p = np.linspace(0.01, 1, 100)

# 自己情報量を計算
I = -np.log2(p)

# グラフを描く
plt.figure(figsize=(8, 6))
plt.plot(p, I, label='Self-Information I(x)')
plt.xlabel('Probability P(x)')
plt.ylabel('Self-Information I(x)')
plt.title('Self-Information as a Function of Probability')
plt.grid(True)
plt.legend()
plt.show()
