
# k-meansクラスタリングアルゴリズム

import numpy as np

# クラスタ数を設定
k = 4  # クラスタ数（整数）

# データポイント数を設定
n = 200  # データポイント数（整数）

# n個の2次元正規分布のランダムデータを生成
data = np.random.randn(n, 2)  # 200個のデータポイントからなる2次元のデータ配列（形状: (200, 2)）

# 初期セントロイドをランダムに選択
centroids = data[np.random.choice(np.arange(n), size=(k,))]  # 4つの初期セントロイドで2次元のデータ配列（形状: (4, 2)）
# 範囲 0 から n-1 の整数から k 個の値を選ぶ

# k-meansクラスタリングの反復回数を設定
for l in range(10):
    # 各データポイントが属するクラスタを格納する配列を初期化
    indexes = np.zeros(data.shape[0])  # 形状: (200,)
    # 各データポイントに最も近いセントロイドを見つける
    for centroid in centroids:
        for i, x in enumerate(data):
            # 各セントロイドとの距離の二乗の和を計算し、最も小さいインデックスを取得
            indexes[i] = np.argmin(np.sum((x - centroids) ** 2, axis=1))  # 形状: (200,)
    # 各クラスタのセントロイドを再計算する
    for i in range(k):  # 各クラスタに対して
        # クラスタに属するデータポイントの平均を計算してセントロイドを更新
        centroids[i] = data[indexes == i].mean(axis=0)  # 各クラスタの新しいセントロイド（形状: (4, 2)）



# 修正後
import numpy as np

# クラスタ数を設定
k = 4

# データポイント数を設定
n = 200

# n個の2次元正規分布のランダムデータを生成
data = np.random.randn(n, 2)

# 初期セントロイドをランダムに選択
# np.random.choice()の中にnp.arange(n)を入れる
indices = np.random.choice(np.arange(n), size=k, replace=False)# 初期セントロイドのインデックスをランダムに選択（形状: (4,))
# 範囲 0 から n-1 の整数から k 個の値を選ぶ　replace=False を指定することで、選択は重複を許可しない
centroids = data[indices] # 4つの初期セントロイドで2次元のデータ配列（形状: (4, 2)）

# k-meansクラスタリングの反復回数を設定
for l in range(10):
    # 各データポイントが属するクラスタを格納する配列を初期化
    indexes = np.zeros(data.shape[0], dtype=int)

    # 各データポイントに最も近いセントロイドを見つける
    for i, x in enumerate(data):
        # 各セントロイドとの距離を計算
        distances = np.linalg.norm(x - centroids, axis=1)
        # 最も近いセントロイドのインデックスを見つける
        indexes[i] = np.argmin(distances)

    # 各クラスタのセントロイドを再計算する
    for i in range(k):
        # クラスタに属するデータポイントを抽出
        points_in_cluster = data[indexes == i]
        # クラスタが空でない場合、セントロイドを更新
        if len(points_in_cluster) > 0:
            centroids[i] = points_in_cluster.mean(axis=0)

# 最終的なセントロイドを出力
print("最終的なセントロイド:", centroids)
