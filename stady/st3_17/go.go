package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"
)

// クラスタ数とデータポイント数の定数
const (
	k = 4   // クラスタ数
	n = 200 // データポイント数
)

// 2次元ポイントを表す構造体
type Point struct {
	x, y float64
}

// データポイントの配列を生成する関数
func generateData(n int) []Point {
	data := make([]Point, n)
	for i := range data {
		data[i] = Point{rand.NormFloat64(), rand.NormFloat64()}
	}
	return data
}

// 初期セントロイドをランダムに選択する関数
func initializeCentroids(data []Point, k int) []Point {
	centroids := make([]Point, k)
	rand.Seed(time.Now().UnixNano())
	for i := range centroids {
		centroids[i] = data[rand.Intn(len(data))]
	}
	return centroids
}

// ポイント間の距離を計算する関数
func distance(a, b Point) float64 {
	return math.Sqrt((a.x-b.x)*(a.x-b.x) + (a.y-b.y)*(a.y-b.y))
}

// データポイントが最も近いセントロイドのインデックスを返す関数
func closestCentroid(point Point, centroids []Point) int {
	minDist := distance(point, centroids[0])
	minIndex := 0
	for i := 1; i < len(centroids); i++ {
		dist := distance(point, centroids[i])
		if dist < minDist {
			minDist = dist
			minIndex = i
		}
	}
	return minIndex
}

// セントロイドを更新する関数
func updateCentroids(data []Point, indexes []int, k int) []Point {
	newCentroids := make([]Point, k)
	counts := make([]int, k)

	for i, idx := range indexes {
		newCentroids[idx].x += data[i].x
		newCentroids[idx].y += data[i].y
		counts[idx]++
	}

	for i := range newCentroids {
		if counts[i] > 0 {
			newCentroids[i].x /= float64(counts[i])
			newCentroids[i].y /= float64(counts[i])
		}
	}

	return newCentroids
}

func main() {
	// データポイントを生成
	data := generateData(n)

	// 初期セントロイドを設定
	centroids := initializeCentroids(data, k)

	// クラスタリングの反復回数
	for l := 0; l < 10; l++ {
		indexes := make([]int, len(data))
		// 各データポイントに最も近いセントロイドを見つける
		for i, point := range data {
			indexes[i] = closestCentroid(point, centroids)
		}
		// 各クラスタのセントロイドを再計算する
		centroids = updateCentroids(data, indexes, k)
	}

	// 最終的なセントロイドを出力
	fmt.Println("最終的なセントロイド:")
	for _, centroid := range centroids {
		fmt.Printf("[%.2f, %.2f]\n", centroid.x, centroid.y)
	}
}
