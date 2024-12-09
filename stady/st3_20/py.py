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

