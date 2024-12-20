import numpy as np
import matplotlib.pyplot as plt

# エントロピーについての説明
print("エントロピー（Entropy）は、システムの不確実性や情報の不確実性を測る尺度で、")
print("具体的には、確率分布における情報の平均量を示し、次の式で定義される：")
print("H(X) = -sum(P(x_i) * log2(P(x_i)))")
print("このグラフでは、横軸に発生確率 P(x)、縦軸にエントロピー H(X) を取り、")
print("確率が均等な場合（0.5）にエントロピーが最大となり、確率が0または1に近づくとエントロピーは0に近づく")


# 発生確率の範囲を定義（0を含めないように微小な値に変更）
p = np.linspace(0.0001, 0.9999, 100)

# エントロピーを計算
entropy = -p * np.log2(p) - (1 - p) * np.log2(1 - p)

# グラフを描く
plt.figure(figsize=(8, 6))
plt.plot(p, entropy, label='Entropy H(X)')
plt.xlabel('Probability P(x)')
plt.ylabel('Entropy H(X)')
plt.title('Entropy as a Function of Probability')
plt.grid(True)
plt.legend()
plt.show()
