import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from sklearn.datasets import make_blobs

# データセットの生成
def generate_data(n_samples, n_features, n_clusters):
    data, _ = make_blobs(n_samples=n_samples, n_features=n_features, centers=n_clusters, random_state=42)
    # n_samples:生成するデータポイントの数, n_features:各データポイントの特徴量の数, centers:クラスタの数, random_state:乱数のシードを42に設定することで、毎回同じデータセットが生成
    return data

# k-means++ アルゴリズムによる初期セントロイドの選択
def initialize_centroids(data, k):
    n = data.shape[0]  # データポイントの数を取得
    probabilities = np.repeat(1/n, n)  # 各データポイントが選ばれる確率を均等に設定
    centroids = np.zeros((k, 2))  # k個のセントロイドを格納する配列を初期化（2次元データ）
    distances = np.zeros((n, k))  # 各データポイントから各セントロイドまでの距離を格納する配列を初期化

    for i in range(k):
        # 各セントロイドを確率に基づいて選択
        centroids[i] = data[np.random.choice(np.arange(n), p=probabilities, size=(1))]
        
        # 各データポイントから選ばれたセントロイドまでの距離を計算
        distances[:, i] = np.sum((data - centroids[i]) ** 2, axis=1)
        
        # 新しいセントロイドを選ぶための確率を更新
        # 距離が長いデータポイントが選ばれる確率を高くする
        probabilities = np.sum(distances, axis=1) / np.sum(distances)

    return centroids
# 各ステップの詳細:
# 1. n = data.shape[0] - データポイントの総数を取得
# 2. probabilities = np.repeat(1/n, n) - 各データポイントが選ばれる確率を均等に設定
# 3. centroids = np.zeros((k, 2)) - k個のセントロイドを格納するための配列を初期化します。ここでは2次元データを扱っていると仮定
# 4. distances = np.zeros((n, k)) - 各データポイントから各セントロイドまでの距離を格納するための配列を初期化
# 5. for i in range(k) - k個のセントロイドを選択するためのループを開始
# 6. centroids[i] = data[np.random.choice(np.arange(n), p=probabilities, size=(1))] - 確率に基づいてセントロイドを1つ選択
# 7. distances[:, i] = np.sum((data - centroids[i]) ** 2, axis=1) - 各データポイントから選ばれたセントロイドまでの距離を計算
# 8. probabilities = np.sum(distances, axis=1) / np.sum(distances) - 新しいセントロイドを選ぶための確率を更新（この確率は、距離が大きいデータポイントが選ばれる確率を高くする）

# サンプルデータの作成と関数の実行例
# np.random.seed(42)  # 再現性のためのシード設定
# data = np.random.rand(100, 2)  # 100個の2次元データポイントを作成
# k = 3  # クラスタ数
# centroids = initialize_centroids(data, k)
# print(centroids)

# k-means アルゴリズム
def kmeans(data, k, max_iters=100):
    centroids = initialize_centroids(data, k)  # 初期セントロイドを設定
    for _ in range(max_iters):
        # 各データポイントを最近傍のセントロイドに割り当て
        distances = np.array([np.sum((data - centroid) ** 2, axis=1) for centroid in centroids]).T
        clusters = np.argmin(distances, axis=1)  # 各データポイントに最も近いセントロイドのインデックスを割り当て

        # 新しいセントロイドの計算
        new_centroids = np.array([data[clusters == i].mean(axis=0) for i in range(k)])

        # セントロイドが収束した場合、終了
        if np.all(centroids == new_centroids):
            break

        centroids = new_centroids  # セントロイドを更新
    return centroids, clusters  # 最終的なセントロイドとクラスタ割り当てを返す

# プロット関数
def plot_clusters(data, centroids, clusters):
    plt.scatter(data[:, 0], data[:, 1], c=clusters, cmap='viridis', marker='o', edgecolor='k', s=50)  # データポイントのプロット
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=200)  # セントロイドのプロット
    plt.title('K-means++ Clustering')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

# メイン処理
def main():
    n_samples = 10  # サンプル数
    n_features = 2  # 特徴量の数
    n_clusters = 2  # クラスタ数

    # データセットの生成
    data = generate_data(n_samples, n_features, n_clusters)

    # k-means クラスタリングの実行
    centroids, clusters = kmeans(data, n_clusters)

    # クラスタのプロット
    plot_clusters(data, centroids, clusters)

if __name__ == "__main__":
    main()
