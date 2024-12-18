import numpy as np
import matplotlib.pyplot as plt

# グラフの解釈に関する説明
print("このグラフは、2つの確率変数の結合エントロピーを示す")
print("横軸（Event）は各イベントを表し、縦軸（Joint Entropy）は結合エントロピーへの寄与を示す")
print("結合エントロピーは、2つの確率変数の共同分布に基づいて計算され、システム全体の不確実性を示す")

# 2つの確率変数の共同分布
joint_probabilities = np.array([
    [0.1, 0.2, 0.1],
    [0.2, 0.1, 0.1],
    [0.1, 0.05, 0.05]
])

# 結合エントロピーの計算
epsilon = 1e-10  # ゼロ割り回避のための微小値
joint_entropy_terms = -joint_probabilities * np.log2(joint_probabilities + epsilon)  # 各イベントの結合エントロピーへの寄与
joint_entropy = np.sum(joint_entropy_terms)

# 結果を表示
print(f"システム全体の結合エントロピー: {joint_entropy:.4f} ビット")

# グラフを描くための準備
# ラベル生成（行列のインデックスを使用）
labels = [f'P({i},{j})' for i in range(joint_probabilities.shape[0]) for j in range(joint_probabilities.shape[1])]

# 結合エントロピーの寄与度を可視化
plt.figure(figsize=(10, 6))
plt.bar(labels, joint_entropy_terms.flatten(), color='skyblue')
plt.xlabel('Event')
plt.ylabel('Contribution to Joint Entropy')
plt.title('Joint Entropy Contribution by Each Event')
plt.grid(True)
plt.show()
