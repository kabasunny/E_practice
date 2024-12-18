import numpy as np
import matplotlib.pyplot as plt

# グラフの解釈に関する説明
print("このグラフは、各イベントの発生確率と対応する自己情報量を示し")
print("横軸（Event）は各イベントを表し、縦軸（Self-Information）は自己情報量を示し")
print("自己情報量は、イベントの発生確率が低いほど高くなり")
print("エントロピーは、これらの自己情報量を各イベントの発生確率で重み付けして合計したもの")
print("計算されたエントロピーはシステム全体の不確実性を示す")

# 各イベントの確率
probabilities = np.array([0.01, 0.1, 0.3, 0.35, 0.5, 0.65, 0.7, 0.9, 0.99])

# エントロピーの計算
epsilon = 1e-10  # ゼロ割り回避のための微小値
entropy_terms = -probabilities * np.log2(probabilities + epsilon)  # 各イベントのエントロピーへの寄与
entropy = np.sum(entropy_terms)

# 結果を表示
print(f"システム全体のエントロピー: {entropy:.4f} ビット")

# グラフを描くための準備
labels = ['0.01', '0.1', '0.3', '0.35', '0.5', '0.65', '0.7', '0.9', '0.99']

# エントロピーの寄与度を可視化
plt.figure(figsize=(8, 6))
plt.bar(labels, entropy_terms, color='skyblue')
plt.xlabel('Event')
plt.ylabel('Contribution to Entropy')
plt.title('Entropy Contribution by Each Event')
plt.grid(True)
plt.show()

